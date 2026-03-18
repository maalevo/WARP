# experiment_batcher.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import json
from datetime import datetime, timezone

from experiment.experiment import Experiment, ExperimentResult


@dataclass
class ExperimentBatcher:
    """
    Orchestrates running multiple experiments with different configurations.
    """

    def __init__(
        self,
        cfgs: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]],
        verbose: int = 1,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.cfgs: List[Dict[str, Any]] = self._normalize_cfgs(cfgs)
        self.verbose = verbose
        self.logger = logger if logger is not None else print

    # ---------------- public API ----------------

    def run_all(self) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []
        n = len(self.cfgs)

        if n == 0:
            raise ValueError("No experiments to run (empty config list).")

        
        for i, cfg in enumerate(self.cfgs, start=1):
            try:
                exp_name = self._experiment_name(cfg, default=f"experiment_{i}")
                cfg = self._ensure_output_name(cfg, exp_name)

                self._log(f"[{i}/{n}] Running '{exp_name}'")
                res = Experiment(cfg, verbose=self.verbose, logger=self.logger).run()
                results.append(res)

                metrics = res.metrics or {}
                m_str = ", ".join(f"{k}={v:.6g}" for k, v in metrics.items()) if metrics else "no metrics"
                self._log(f"[{i}/{n}] Finished '{exp_name}' → {m_str}\n")
            except Exception as e:
                self._log(f"[{i}/{n}] Execution of experiment '{exp_name}' halted due to error: {e}")
        self._save_experiments_results(results)
        return results

    # ---------------- logging ----------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            formatted = f"[Batcher]\t    {msg}"
            if self.logger:
                self.logger(formatted)
            else:
                print(formatted)

    # ---------------- persistence ----------------

    def _save_experiments_results(self, results: List[ExperimentResult]) -> None:
        """
        Writes a single batch-level JSON summary.

        Output path priority:
          1) top-level batch output.path (if present)
          2) artifacts/batches
        """
        if not results:
            return

        out_dir = self._infer_batch_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        batch_out = getattr(self, "_batch_output", {}) or {}
        batch_name = batch_out.get("name")

        if batch_name:
            filename = f"{batch_name}_{ts}.json"
        else:
            filename = f"batch_summary_{ts}.json"

        json_path = out_dir / filename

        payload = {
            "timestamp_utc": ts,
            "num_experiments": len(results),
            "experiments": [
                {
                    "experiment_name": self._experiment_name(r.config or {}, default="(unknown)"),
                    "metrics": r.metrics or {},
                    "artifact_path": r.artifact_path,
                    "config": r.config,
                }
                for r in results
            ],
        }

        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self._log(f"Saved batch summary JSON: {json_path}")

    # ---------------- config normalization ----------------

    def _normalize_cfgs(
        self,
        cfgs: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if isinstance(cfgs, (str, Path)):
            loaded = self._load_config_file(Path(cfgs))
            return self._extract_experiment_list(loaded)

        if isinstance(cfgs, dict):
            return self._extract_experiment_list(cfgs)

        if isinstance(cfgs, list):
            if not all(isinstance(x, dict) for x in cfgs):
                raise ValueError("cfgs list must contain only dict experiment configs.")
            self._batch_output = {}
            return cfgs

        raise TypeError(f"Unsupported cfgs type: {type(cfgs)}")

    def _extract_experiment_list(self, loaded: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "experiments" in loaded:
            exps = loaded.get("experiments")
            if not isinstance(exps, list):
                raise ValueError("'experiments' must be a list.")
            self._batch_output = loaded.get("output") or {}
            return exps

        self._batch_output = {}
        return [loaded]

    def _load_config_file(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()

        if suffix in (".yml", ".yaml"):
            try:
                import yaml
            except Exception as e:
                raise RuntimeError("YAML config provided but PyYAML is not installed.") from e
            return yaml.safe_load(text) or {}

        if suffix == ".json":
            return json.loads(text)

        try:
            import yaml
            return yaml.safe_load(text) or {}
        except Exception:
            return json.loads(text)

    # ---------------- helpers ----------------

    def _experiment_name(self, cfg: Dict[str, Any], default: str) -> str:
        out = cfg.get("output") or {}
        return out.get("name") or cfg.get("experiment_name") or default

    def _ensure_output_name(self, cfg: Dict[str, Any], exp_name: str) -> Dict[str, Any]:
        cfg = dict(cfg)
        out = dict(cfg.get("output") or {})
        if not out.get("name"):
            out["name"] = exp_name
        cfg["output"] = out
        return cfg

    def _infer_batch_output_dir(self) -> Path:
        batch_out = getattr(self, "_batch_output", None) or {}
        batch_path = batch_out.get("path")
        if batch_path:
            return Path(batch_path)
        return Path("artifacts") / "batches"
