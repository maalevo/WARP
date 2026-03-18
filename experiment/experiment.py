# experiment.py

from __future__ import annotations
from dataclasses import dataclass
import dataclasses
import hashlib
import joblib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf

# --- package imports ---
from load.loaders import LoadSpec, load_events
from split.splitters import TemporalSplitSpec, temporal_split_by_case_end
from worktime_mapping.worktime_identifiers import (
    WorktimeIdentificationSpec,
    identify_discrete_worktime_intervals
)
from worktime_mapping.worktime_mappers import (
    WorktimeMappingSpec,
    map_clocktime_to_worktime_timestamp_log,
    map_worktime_to_clocktime_interval_series,
)
from prefix.prefixers import PrefixSpec, build_prefix_log
from feature.feature_builder import (
    FeatureSpec,
    NormState,
    apply_stateful_norm,
    invert_stateful_norm,
)
from feature.dataframe_builder import build_feature_dataframe
from transform.transformers import (
    TransformerSpec,
    TransformerState,
    build_transformation
)
from model.models import (
    ModelSpec, 
    LSTMModel, 
    XGBoostModel
)


# ============================
# Stage I/O payloads
# ============================

@dataclass(frozen=True)
class LoadInput:
    load_cfg: Dict[str, Any]

@dataclass
class LoadOutput:
    log: pd.DataFrame
    case_col: str
    act_col: str
    ts_col: str


@dataclass(frozen=True)
class SplitInput:
    log: pd.DataFrame
    case_col: str
    ts_col: str
    split_cfg: Dict[str, Any]

@dataclass
class SplitOutput:
    log_tr: pd.DataFrame
    log_va: pd.DataFrame
    log_te: pd.DataFrame
    case_col: str

@dataclass(frozen=True)
class WorktimeIdentifyInput:
    log_tr: pd.DataFrame
    ts_col: str
    worktime_id_cfg: Dict[str, Any]

@dataclass(frozen=True)
class WorktimeIdentifyOutput:
    worktime_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]

@dataclass(frozen=True)
class WorktimeTransformInput:
    log_tr: pd.DataFrame
    log_va: pd.DataFrame
    log_te: pd.DataFrame
    worktime_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ts_col: str
    worktime_id_cfg: Dict[str, Any]
    worktime_mapping_cfg: Dict[str, Any]

@dataclass(frozen=True)
class WorktimeTransformOutput:
    log_tr: pd.DataFrame
    log_va: pd.DataFrame
    log_te: pd.DataFrame
    clocktime_col: str
    worktime_col: str

@dataclass(frozen=True)
class PrefixInput:
    log_tr: pd.DataFrame
    log_va: pd.DataFrame
    log_te: pd.DataFrame
    case_col: str
    act_col: str
    ts_col: str
    prefix_cfg: Dict[str, Any]

@dataclass
class PrefixOutput:
    pref_tr: pd.DataFrame
    pref_va: pd.DataFrame
    pref_te: pd.DataFrame
    case_col: str

@dataclass(frozen=True)
class TargetInput:
    prefix_out: PrefixOutput
    target_cfg: str

@dataclass
class TargetOutput:
    y_tr: pd.Series
    y_va: pd.Series
    y_te: pd.Series
    target_key: str
    case_col: str
    target_state: Optional[List[NormState]] = None


@dataclass(frozen=True)
class FeatureInput:
    prefix_out: PrefixOutput
    features_cfg: List[Dict[str, Any]]
    worktime_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
    worktime_id_cfg: Dict[str, Any]
    worktime_mapping_cfg: Dict[str, Any]
    ts_col: str
    

@dataclass
class FeatureOutput:
    X_tr: pd.DataFrame
    X_cases_tr: pd.Series
    X_va: pd.DataFrame
    X_cases_va: pd.Series
    X_te: pd.DataFrame
    X_cases_te: pd.Series
    case_col: str

@dataclass(frozen=True)
class TransformInput:
    feature_out: FeatureOutput
    target_out: TargetOutput
    transformer_cfg: Dict[str, Any]

@dataclass(frozen=True)
class TransformOutput:
    X_tr_t: pd.DataFrame
    y_tr_t: pd.Series
    case_ids_tr: pd.Index

    X_va_t: pd.DataFrame
    y_va_t: pd.Series
    case_ids_va: pd.Index

    X_te_t: pd.DataFrame
    y_te_t: pd.Series
    case_ids_te: pd.Index

    transformer: TransformerState

@dataclass(frozen=True)
class TrainInput:
    X_tr: pd.DataFrame
    y_tr: pd.Series
    X_va: pd.DataFrame
    y_va: pd.Series
    model_cfg: Dict[str, Any]

@dataclass(frozen=True)
class TrainOutput:
    model: Any

@dataclass(frozen=True)
class PredictInput:
    train_out: TrainOutput
    transformer_out: TransformOutput
    target_out: TargetOutput 
    
    log_te: pd.DataFrame
    target_cfg: Dict[str, Any]
    worktime_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ts_col: str
    worktime_id_cfg: Dict[str, Any]
    worktime_mapping_cfg: Dict[str, Any]
    
    X_te: pd.DataFrame
    y_te: pd.Series
    

@dataclass(frozen=True)
class PredictOutput:
    y_pred: pd.Series
    y_true: pd.Series
    metrics: Dict[str, float]


# ============================
# Stage implementations (pure logic)
# ============================

class LoadStage:
    VERSION = "v1"

    def run(self, inp: LoadInput) -> LoadOutput:
        ds = inp.load_cfg
        path = ds.get("path")
        case_col = ds.get("case_id_col", "case:concept:name")
        act_col = ds.get("activity_col", "concept:name")
        ts_col = ds.get("timestamp_col", "time:timestamp")

        spec = LoadSpec(
            path=path,
            case_id_col=case_col,
            activity_col=act_col,
            timestamp_col=ts_col,
            rename=ds.get("rename"),
        )
        log = load_events(spec)
        return LoadOutput(log=log, case_col=case_col, act_col=act_col, ts_col=ts_col)

class SplitStage:
    VERSION = "v1"

    def run(self, inp: SplitInput) -> SplitOutput:
        split_cfg = inp.split_cfg or {}
        split_type = (split_cfg.get("type") or "temporal").lower()
        params = split_cfg.get("params", {})

        if split_type == "temporal":
            sp = TemporalSplitSpec(
                test_ratio=float(params.get("test_ratio", 0.2)),
                val_ratio=float(params.get("val_ratio", 0.0)),
                shuffle_within_same_timestamps=bool(params.get("shuffle_within_same_timestamps", False)),
                seed=int(params.get("seed", 42)),
                case_id_col=inp.case_col,
                timestamp_col=inp.ts_col,
            )
            res = temporal_split_by_case_end(inp.log, sp)
            return SplitOutput(res.log_tr, res.log_va, res.log_te, inp.case_col)

        raise ValueError(f"Unknown split.type '{split_type}'")

class WorktimeIdentifyStage:
    VERSION = "v1"

    def run(self, inp: WorktimeIdentifyInput) -> WorktimeIdentifyOutput:
        log_tr = inp.log_tr
        ts_col = inp.ts_col
        worktime_id_cfg = inp.worktime_id_cfg or {}

        spec = WorktimeIdentificationSpec(
            bucket_len=worktime_id_cfg.get("bucket_len", "3600"),
            period_len=worktime_id_cfg.get("period_len", "week"),
            threshold=float(worktime_id_cfg.get("threshold", 0.2)),
            threshold_ref=worktime_id_cfg.get("threshold_ref", "max"),
            timestamp_col=ts_col,
            week_anchor=worktime_id_cfg.get("week_anchor", "MON"),
            tz=worktime_id_cfg.get("tz", None),
        )
        res = identify_discrete_worktime_intervals(log_tr, spec)
        return WorktimeIdentifyOutput(worktime_intervals=res.worktime_intervals)
    
class WorktimeTransformStage:
    VERSION = "v1"

    def run(self, inp: WorktimeTransformInput) -> WorktimeTransformOutput:
        log_tr = inp.log_tr
        log_va = inp.log_va
        log_te = inp.log_te

        worktime_intervals = inp.worktime_intervals
        ts_col = inp.ts_col
        worktime_id_cfg = inp.worktime_id_cfg or {}
        worktime_mapping_cfg = inp.worktime_mapping_cfg or {}

        spec = WorktimeMappingSpec(
            worktime_intervals=worktime_intervals,
            worktime_anchor=pd.Timestamp(worktime_mapping_cfg.get("worktime_anchor", "1970-01-01T00:00:00Z")),
            period_len=worktime_id_cfg.get("period_len", "week"),
            week_anchor=worktime_id_cfg.get("week_anchor", "MON"),
            tz=worktime_id_cfg.get("tz", None),
            clocktime_col=ts_col,
            worktime_col=worktime_mapping_cfg.get("worktime_col", "worktime:timestamp"),
        )
        res_tr = map_clocktime_to_worktime_timestamp_log(log_tr, spec)
        res_va = map_clocktime_to_worktime_timestamp_log(log_va, spec)
        res_te = map_clocktime_to_worktime_timestamp_log(log_te, spec)
        return WorktimeTransformOutput(
            log_tr=res_tr,
            log_va=res_va,
            log_te=res_te,
            clocktime_col=spec.clocktime_col,
            worktime_col=spec.worktime_col
        )

class PrefixStage:
    VERSION = "v1"

    def run(self, inp: PrefixInput) -> PrefixOutput:
        case_col, act_col, ts_col = inp.case_col, inp.act_col, inp.ts_col

        log_tr = inp.log_tr
        log_va = inp.log_va
        log_te = inp.log_te

        px = inp.prefix_cfg or {}
        spec = PrefixSpec(
            min_prefix_len=int(px.get("min_prefix_len", 1)),
            max_prefix_len=px.get("max_prefix_len"),
            drop_last_event=bool(px.get("drop_last_event", False)),
            case_id_col=case_col,
            activity_col=act_col,
            time_col=ts_col,
        )
        
        pref_tr = build_prefix_log(log_tr, spec)
        pref_va = build_prefix_log(log_va, spec)
        pref_te = build_prefix_log(log_te, spec)
        return PrefixOutput(pref_tr=pref_tr, pref_va=pref_va, pref_te=pref_te, case_col=case_col)

class FeatureStage:
    VERSION = "v1"

    def run(self, inp: FeatureInput) -> FeatureOutput:
        prefixes = inp.prefix_out
        worktime_intervals = inp.worktime_intervals
        worktime_mapping_cfg = inp.worktime_mapping_cfg or {}
        worktime_id_cfg = inp.worktime_id_cfg or {}
        ts_col = inp.ts_col
        
        feature_cfgs = inp.features_cfg or []
        feature_specs = []
        
        # 1. Prepare Feature Specifications
        for fc in feature_cfgs:
            feature_key = fc.get("feature_key", "")
            # Copy params to prevent mutation of the original config object
            params = fc.get("params", {}).copy()
            
            # 2. Handle One-Hot Encoding Consistency
            # Determine categories strictly from the training set to prevent 
            # data leakage or shape mismatches in test data.
            if params.get("encoding", "") == "onehot":
                source_col_name = params.get("source_col_name", "")
                # Sort uniques to ensure deterministic column ordering
                unique_vals = sorted(
                    prefixes.pref_tr.prefix_log[source_col_name].dropna().unique().tolist()
                )
                params["categories"] = unique_vals

            # 3. handle worktime mapping spec for worktime-based features
            if feature_key in ("worktime_in_day", "worktime_in_week"):
                worktime_mapping_spec = WorktimeMappingSpec(
                    worktime_intervals=worktime_intervals,
                    worktime_anchor=pd.Timestamp(worktime_mapping_cfg.get("worktime_anchor", "1970-01-01T00:00:00Z")),
                    period_len=worktime_id_cfg.get("period_len", "week"),
                    week_anchor=worktime_id_cfg.get("week_anchor", "MON"),
                    tz=worktime_id_cfg.get("tz", None),
                    clocktime_col=ts_col,
                    worktime_col=worktime_mapping_cfg.get("worktime_col", "worktime:timestamp"),
                )
                params["worktime_mapping_spec"] = worktime_mapping_spec
                
                
            feature_specs.append(
                FeatureSpec(feature_key=feature_key, params=params)
            )
        
        if feature_specs:
            # 3. Build Training Data & Learn States
            # Returns X_tr and the learned normalization parameters (train_states)  
            X_tr, train_states = build_feature_dataframe(
                prefixes.pref_tr.prefix_log, 
                feature_specs,
            )

            # 4. Build Validation Data
            # Pass fit_states=train_states to apply the learned scaling (transform only)
            X_va, _ = build_feature_dataframe(
                prefixes.pref_va.prefix_log, 
                feature_specs, 
                fit_states=train_states
            )
            
            # 5. Build Test Data
            # Pass fit_states=train_states to apply the learned scaling (transform only)
            X_te, _ = build_feature_dataframe(
                prefixes.pref_te.prefix_log, 
                feature_specs, 
                fit_states=train_states
            )

            return FeatureOutput(
                X_tr=X_tr,
                X_cases_tr=prefixes.pref_tr.prefix_log[prefixes.case_col],
                X_va=X_va, 
                X_cases_va=prefixes.pref_va.prefix_log[prefixes.case_col],
                X_te=X_te,
                X_cases_te=prefixes.pref_te.prefix_log[prefixes.case_col],
                case_col=prefixes.case_col
            )
        else:
            # Fallback: No features configured
            empty_tr = pd.DataFrame(index=prefixes.pref_tr.prefix_log.index)
            empty_va = pd.DataFrame(index=prefixes.pref_va.prefix_log.index)
            empty_te = pd.DataFrame(index=prefixes.pref_te.prefix_log.index)
            
            return FeatureOutput(
                X_tr=empty_tr, 
                X_va=empty_va, 
                X_te=empty_te,
                X_cases_tr=prefixes.pref_tr.prefix_log[prefixes.case_col],
                X_cases_va=prefixes.pref_va.prefix_log[prefixes.case_col],
                X_cases_te=prefixes.pref_te.prefix_log[prefixes.case_col],
                case_col=prefixes.case_col
            )
        
class TargetStage:
    VERSION = "v1"

    def run(self, inp: TargetInput) -> TargetOutput:
        px = inp.prefix_out
        
        # 1. Parse Configuration
        target_key = inp.target_cfg.get("target", "").lower()
        time_mode = inp.target_cfg.get("time", "").lower()
        # Get normalization steps (e.g., ["log", "standard"])
        norm_steps = inp.target_cfg.get("normalization", []) 

        # 2. Select Raw Target Data
        if target_key == "remaining_time":
            if time_mode == "clocktime":
                y_tr = px.pref_tr.ct_remaining_time.astype(float)
                y_va = px.pref_va.ct_remaining_time.astype(float)
                y_te = px.pref_te.ct_remaining_time.astype(float)
            elif time_mode == "worktime":
                y_tr = px.pref_tr.wt_remaining_time.astype(float)
                y_va = px.pref_va.wt_remaining_time.astype(float)
                y_te = px.pref_te.wt_remaining_time.astype(float)
            else:
                raise ValueError(f"Unknown time mode '{time_mode}'. Must be 'clocktime' or 'worktime'.")
                
        elif target_key == "time_until_next_event":
            if time_mode == "clocktime":
                y_tr = px.pref_tr.ct_time_until_next_event.astype(float)
                y_va = px.pref_va.ct_time_until_next_event.astype(float)
                y_te = px.pref_te.ct_time_until_next_event.astype(float)
            elif time_mode == "worktime":
                y_tr = px.pref_tr.wt_time_until_next_event.astype(float)
                y_va = px.pref_va.wt_time_until_next_event.astype(float)
                y_te = px.pref_te.wt_time_until_next_event.astype(float)
            else:
                raise ValueError(f"Unknown time mode '{time_mode}'. Must be 'clocktime' or 'worktime'.")
                
        else:
            raise ValueError(f"Unknown target_key '{target_key}'. Must be 'remaining_time' or 'time_until_next_event'.")

        # 3. Apply Stateful Normalization
        target_states: List[NormState] = []

        if norm_steps:
            # Step A: Train (Fit & Transform)
            # We iterate through steps, updating y_tr and capturing the state
            for strategy in norm_steps:
                y_tr, state = apply_stateful_norm(y_tr, strategy, state=None)
                target_states.append(state)

            # Step B: Validation (Transform using Train State)
            for i, strategy in enumerate(norm_steps):
                state = target_states[i]
                y_va, _ = apply_stateful_norm(y_va, strategy, state=state)

            # Step C: Test (Transform using Train State)
            for i, strategy in enumerate(norm_steps):
                state = target_states[i]
                y_te, _ = apply_stateful_norm(y_te, strategy, state=state)

        return TargetOutput(
            y_tr=y_tr, 
            y_va=y_va, 
            y_te=y_te, 
            target_key=target_key,
            case_col=px.case_col,
            # Pass the states so we can inverse_transform predictions later
            target_state=target_states 
        )

class TransformStage:
    VERSION = "v1"

    def run(self, inp: TransformInput) -> TransformOutput:
        X_tr = inp.feature_out.X_tr
        X_cases_tr = inp.feature_out.X_cases_tr
        X_va = inp.feature_out.X_va
        X_cases_va = inp.feature_out.X_cases_va
        X_te = inp.feature_out.X_te
        X_cases_te = inp.feature_out.X_cases_te

        y_tr = inp.target_out.y_tr
        y_va = inp.target_out.y_va
        y_te = inp.target_out.y_te

        # 1. Normalize Key
        # Strip '_transformer' suffix to match the new API keys (e.g., "lstm_transformer" -> "lstm")
        raw_key = inp.transformer_cfg.get("transformer_key", "").lower()
        key = raw_key.replace("_transformer", "")
        
        spec = TransformerSpec(transformer_key=key, params=inp.transformer_cfg.get("params", {}))

        # 2. Train: Fit & Transform
        # Passing state=None triggers "fit" mode (learns schema like feature names/counts)
        X_tr_t, y_tr_t, case_ids_tr, state = build_transformation(
            X_tr, y_tr, X_cases_tr, spec, state=None
        )

        # 3. Validation: Transform using Learned State
        # Passing the state ensures X_va is reshaped exactly like X_tr
        X_va_t, y_va_t, case_ids_va, _ = build_transformation(
            X_va, y_va, X_cases_va, spec, state=state
        )

        # 4. Test: Transform using Learned State
        X_te_t, y_te_t, case_ids_te, _ = build_transformation(
            X_te, y_te, X_cases_te, spec, state=state
        )

        # 5. Return Output
        # Note: 'transformer' field in TransformOutput now holds the 'TransformerState' object
        return TransformOutput(
            X_tr_t=X_tr_t, y_tr_t=y_tr_t, case_ids_tr=case_ids_tr,
            X_va_t=X_va_t, y_va_t=y_va_t, case_ids_va=case_ids_va,
            X_te_t=X_te_t, y_te_t=y_te_t, case_ids_te=case_ids_te,
            transformer=state 
        )

class TrainStage:
    VERSION = "v2"

    def run(self, inp: TrainInput) -> TrainOutput:
        model_cfg = inp.model_cfg or {}
        model_type = (model_cfg.get("type") or "lstm").lower()
        params = model_cfg.get("params", {})

        if model_type == "lstm":
            spec = ModelSpec(model_key="lstm", params=params)
            model = LSTMModel(spec)
            model.fit(inp.X_tr, inp.y_tr, inp.X_va, inp.y_va)
        elif model_type in ("xgboost"):
            spec = ModelSpec(model_key="xgboost", params=params)
            model = XGBoostModel(spec)
            model.fit(inp.X_tr, inp.y_tr, inp.X_va, inp.y_va)
        else:
            raise ValueError(f"Unknown model.type '{model_type}'")

        return TrainOutput(model=model)

class PredictStage:
    VERSION = "v1"

    def run(self, inp: PredictInput) -> PredictOutput:        
        
        model = inp.train_out.model
        case_ids = inp.transformer_out.case_ids_te
        
        # Ensure you are passing target_out to PredictInput
        # This contains the 'mean', 'std', 'min', 'max' needed for inversion
        target_states = inp.target_out.target_state 
        
        log_te = inp.log_te
        target_cfg = inp.target_cfg

        worktime_intervals = inp.worktime_intervals
        worktime_id_cfg = inp.worktime_id_cfg or {}
        worktime_mapping_cfg = inp.worktime_mapping_cfg or {}
        ts_col = inp.ts_col

        # 1. Model Prediction (Output is Normalized)
        y_pred_normalized = model.predict(inp.X_te)
        if y_pred_normalized.ndim > 1:
            y_pred_normalized = y_pred_normalized.flatten()

        y_pred_normalized_series = pd.Series(y_pred_normalized.values, index=case_ids, name="y_pred")
        y_true_normalized_series = pd.Series(inp.y_te, index=case_ids, name="y_true")
        

        # 3. Value Inversion (De-Normalization) 
        # We use the states from TargetStage to convert Z-scores/Logs back to Seconds.
        y_pred = invert_stateful_norm(y_pred_normalized_series, target_states)
        y_true = invert_stateful_norm(y_true_normalized_series, target_states)

        # 4. Worktime / Clocktime Logic
        if target_cfg.get("time") == "worktime":
            worktime_mapping_spec = WorktimeMappingSpec(
                worktime_intervals=worktime_intervals,
                worktime_anchor=pd.Timestamp(worktime_mapping_cfg.get("worktime_anchor", "1970-01-01T00:00:00Z")),
                period_len=worktime_id_cfg.get("period_len", "week"),
                week_anchor=worktime_id_cfg.get("week_anchor", "MON"),
                tz=worktime_id_cfg.get("tz", None),
                clocktime_col=ts_col,
                worktime_col=worktime_mapping_cfg.get("worktime_col", "worktime:timestamp"),
            )

            
            y_true = pd.Series(y_true)
            y_pred = pd.Series(y_pred)

            log_te = log_te.sort_values("time:timestamp")
            last_worktime_per_case = log_te.groupby("case:concept:name")["worktime:timestamp"].last()

            y_pred = pd.to_timedelta(y_pred, unit='s')
            y_true = pd.to_timedelta(y_true, unit='s')

            mae_wt = float(np.mean(np.abs(y_true.dt.total_seconds() - y_pred.dt.total_seconds())))
            y_pred = map_worktime_to_clocktime_interval_series(y_pred, last_worktime_per_case, worktime_mapping_spec)
            y_true = map_worktime_to_clocktime_interval_series(y_true, last_worktime_per_case, worktime_mapping_spec)
            
            y_pred = y_pred.dt.total_seconds()
            y_true = y_true.dt.total_seconds()

            err = y_true - y_pred
            mae_ct = float(np.mean(np.abs(err)))
            metrics = {"mae": mae_ct, "mae_wt": mae_wt}
        else:
            mae = float(np.mean(np.abs(y_true - y_pred)))
            metrics = {"mae": mae}
        
        return PredictOutput(y_pred=y_pred, y_true=y_true, metrics=metrics)
    

# ============================
# Caching utilities
# ============================

def fingerprint_df(df: pd.DataFrame) -> str:
    """
    Create a deterministic hash for a DataFrame, ignoring object types.
    """
    # Use pandas built-in hash function
    # hash_pandas_object returns Series of int64 hashes per row
    row_hashes = pd.util.hash_pandas_object(df, index=True)
    # Convert to bytes via string concatenation
    combined = row_hashes.values.tobytes()
    return hashlib.sha256(combined).hexdigest()

def fingerprint_series(s: pd.Series) -> str:
    return fingerprint_df(s.to_frame())


def serialize_input(obj):
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        return {k: serialize_input(v) for k, v in d.items()}

    elif isinstance(obj, pd.DataFrame):
        return {"__df_hash__": fingerprint_df(obj), "shape": obj.shape}

    elif isinstance(obj, pd.Series):
        return {"__series_hash__": fingerprint_series(obj), "shape": obj.shape}

    # --- Timedelta handling ---
    elif isinstance(obj, pd.Timedelta):
        # Handle NaT-like values (rare here, but safe)
        if pd.isna(obj):
            return {"__timedelta__": None}
        # Stable + precise: total nanoseconds
        return {"__timedelta_ns__": int(obj.value)}

    elif isinstance(obj, timedelta):
        # Store microseconds exactly (datetime.timedelta precision)
        total_us = int(obj.total_seconds() * 1_000_000)
        return {"__timedelta_us__": total_us}

    elif isinstance(obj, np.timedelta64):
        # Normalize to nanoseconds for consistency
        ns = obj.astype("timedelta64[ns]").astype(np.int64)
        return {"__timedelta_ns__": int(ns)}

    # (Optional) catch pandas NaT at top-level
    elif obj is pd.NaT:
        return {"__nat__": True}

    # --- Containers ---
    elif isinstance(obj, (list, tuple)):
        return [serialize_input(x) for x in obj]

    elif isinstance(obj, dict):
        return {k: serialize_input(v) for k, v in obj.items()}

    else:
        return obj

class StageCache:
    def __init__(self, root: Path = Path(".cache")):
        self.root = root

    def get(self, stage: str, version: str, key: str):
        path = self.root / stage / version / f"{key}.pkl"
        if path.exists():
            return joblib.load(path)
        return None

    def set(self, stage: str, version: str, key: str, value):
        path = self.root / stage / version / f"{key}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value, path)

def run_stage_cached(stage, inp, cache: StageCache, enabled: bool = True):
    if not enabled:
        return stage.run(inp)

    key_payload = serialize_input(inp)
    key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode()
    ).hexdigest()

    cached = cache.get(stage.__class__.__name__, stage.VERSION, key)
    if cached is not None:
        return cached

    out = stage.run(inp)
    cache.set(stage.__class__.__name__, stage.VERSION, key, out)
    return out


# ============================
# Results container (public API)
# ============================

@dataclass
class ExperimentResult:
    config: Dict[str, Any]
    pref_te: pd.DataFrame
    working_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
    X_te: pd.DataFrame
    y_pred: pd.Series
    y_true: pd.Series
    metrics: Dict[str, float] = None
    artifact_path: Optional[str] = None


# ============================
# Experiment Orchestrator
# ============================

class Experiment:
    """
    Orchestrates: Load -> Split -> Prefix -> Feature Engineering -> Transform -> Train/Eval.
    """

    def __init__(self, 
                 cfg: Union[str, Path, Dict[str, Any]], 
                 verbose: int = 1,
                 logger: Optional[Callable[[str], None]] = None):
        self.cfg = self._load_config(cfg)
        self.verbose = verbose
        self._logger = logger

    def run(self) -> ExperimentResult:
        cfg = self.cfg
        t0 = time.perf_counter()


        cache = StageCache()
        use_cache = cfg.get("cache", {}).get("enabled", False)

        seed = cfg.get("seed", 42)
        self._set_global_seed(seed)


        # Load
        load_cfg = cfg.get("load", {})
        #load_out = LoadStage().run(LoadInput(load_cfg=load_cfg))
        load_out = run_stage_cached(
            LoadStage(),
            LoadInput(load_cfg=load_cfg),
            cache,
            use_cache
        )
        self._log(f"(1) Loaded log with {len(load_out.log)} events.")

        # Split
        split_cfg = cfg.get("split", {})
        #split_out = SplitStage().run(SplitInput(log=load_out.log, case_col=load_out.case_col, ts_col=load_out.ts_col, split_cfg=split_cfg))
        split_out = run_stage_cached(
            SplitStage(),
            SplitInput(log=load_out.log, case_col=load_out.case_col, ts_col=load_out.ts_col, split_cfg=split_cfg),
            cache,
            use_cache
        )
        self._log(f"(2) Split log into {len(split_out.log_tr)} train, {len(split_out.log_va)} val, {len(split_out.log_te)} test cases.")
        
        # Identify worktime intervals
        worktime_id_cfg = cfg.get("worktime_identification", {})
        #worktime_id_out = WorktimeIdentifyStage().run(WorktimeIdentifyInput(log=load_out.log, ts_col=load_out.ts_col, worktime_id_cfg=worktime_id_cfg))
        worktime_id_out = run_stage_cached(
            WorktimeIdentifyStage(),
            WorktimeIdentifyInput(log_tr=split_out.log_tr, ts_col=load_out.ts_col, worktime_id_cfg=worktime_id_cfg),
            cache,
            use_cache
        )
        self._log(f"(3) Identified {len(worktime_id_out.worktime_intervals)} worktime intervals.")

        # Map to worktime
        worktime_mapping_cfg = cfg.get("worktime_mapping", {})
        worktime_mapping_out = WorktimeTransformStage().run(WorktimeTransformInput(log_tr=split_out.log_tr, log_va=split_out.log_va, log_te=split_out.log_te, worktime_intervals=worktime_id_out.worktime_intervals, ts_col=load_out.ts_col, worktime_id_cfg=worktime_id_cfg, worktime_mapping_cfg=worktime_mapping_cfg))
        """worktime_mapping_out = run_stage_cached(
            WorktimeTransformStage(),
            WorktimeTransformInput(log=load_out.log, worktime_intervals=worktime_id_out.worktime_intervals, ts_col=load_out.ts_col, worktime_id_cfg=worktime_id_cfg, worktime_mapping_cfg=worktime_mapping_cfg),
            cache,
            use_cache
        )"""
        self._log(f"(4) Transformed log to worktime with column '{worktime_mapping_out.worktime_col}'.")

        # Prefix
        prefix_cfg = cfg.get("prefix", {})
        #prefix_out = PrefixStage().run(PrefixInput(log=load_out.log, case_col=load_out.case_col, act_col=load_out.act_col, ts_col=load_out.ts_col, split_out=split_out, prefix_cfg=prefix_cfg))
        prefix_out = run_stage_cached(
            PrefixStage(),
            PrefixInput(log_tr=worktime_mapping_out.log_tr,
                        log_va=worktime_mapping_out.log_va,
                        log_te=worktime_mapping_out.log_te,
                        case_col=load_out.case_col,
                        act_col=load_out.act_col,
                        ts_col=load_out.ts_col,
                        prefix_cfg=prefix_cfg),
            cache,
            use_cache
        )
        self._log(f"(5) Generated prefixes: {len(prefix_out.pref_tr.prefix_log)} train, {len(prefix_out.pref_va.prefix_log)} val, {len(prefix_out.pref_te.prefix_log)} test events.")

        # Features
        feature_cfg = cfg.get("features", [])
        #feature_out = FeatureStage().run(FeatureInput(prefix_out=prefix_out, features_cfg=feature_cfg))
        feature_out = run_stage_cached(
            FeatureStage(),
            FeatureInput(
                prefix_out=prefix_out,
                features_cfg=feature_cfg,
                worktime_intervals=worktime_id_out.worktime_intervals,
                worktime_id_cfg=worktime_id_cfg,
                worktime_mapping_cfg=worktime_mapping_cfg,
                ts_col=load_out.ts_col,),
            cache,
            use_cache
        )
        self._log(f"(6) Engineered features: {feature_out.X_tr.shape[1]} features for {feature_out.X_tr.shape[0]} train, {feature_out.X_va.shape[0]} val, {feature_out.X_te.shape[0]} test events.")

        # Targets
        target_cfg = cfg.get("target", "")
        #target_out = TargetStage().run(TargetInput(prefix_out=prefix_out, target_cfg=target_cfg))
        target_out = run_stage_cached(
            TargetStage(),
            TargetInput(prefix_out=prefix_out, target_cfg=target_cfg),
            cache,
            use_cache
        )
        self._log(f"(7) Prepared target '{target_out.target_key}': {len(target_out.y_tr)} train, {len(target_out.y_va)} val, {len(target_out.y_te)} test labels.")


        # Transform
        transformer_cfg = cfg.get("transformer", {})
        #transform_out = TransformStage().run(TransformInput(feature_out=feature_out, target_out=target_out, transformer_cfg=transformer_cfg))
        transform_out = run_stage_cached(
            TransformStage(),
            TransformInput(feature_out=feature_out, target_out=target_out, transformer_cfg=transformer_cfg),
            cache,
            use_cache
        )

        self._log(f"(8) Transformed data for model input into shape {transform_out.X_tr_t.shape} train, {transform_out.X_va_t.shape} val, {transform_out.X_te_t.shape} test.")

        # Train
        model_cfg = cfg.get("model", {})
        train_out = TrainStage().run(TrainInput(
            X_tr=transform_out.X_tr_t,
            y_tr=transform_out.y_tr_t,
            X_va=transform_out.X_va_t,
            y_va=transform_out.y_va_t,
            model_cfg=model_cfg
        ))
        '''train_out = run_stage_cached(
            TrainStage(),
            TrainInput(
                X_tr=transform_out.X_tr,
                y_tr=transform_out.y_tr,
                X_va=transform_out.X_va,
                y_va=transform_out.y_va,
                model_cfg=model_cfg
            ),
            cache,
            use_cache
        )'''
        self._log(f"(9) Trained model.")

        # Evaluate
        eval_out = PredictStage().run(PredictInput(
            train_out=train_out,
            transformer_out=transform_out,
            target_out=target_out, 
            
            X_te=transform_out.X_te_t,
            y_te=transform_out.y_te_t,
            log_te=prefix_out.pref_te.prefix_log,
            
            target_cfg=target_cfg,
            worktime_intervals=worktime_id_out.worktime_intervals,
            ts_col=load_out.ts_col,
            worktime_id_cfg=worktime_id_cfg,
            worktime_mapping_cfg=worktime_mapping_cfg
        ))
        self._log(f"(10) Evaluated model.")

        # Save results
        if cfg.get("output"):
            artifact_path = self._save_experiment_results(cfg, eval_out.metrics)
            self._log(f"(11) Saved experiment results to: {artifact_path}")
        else:
            artifact_path = None

        t1 = time.perf_counter()
        self._log(f"Experiment completed in {t1 - t0:.2f} seconds.")

        return ExperimentResult(
            config=cfg,
            pref_te=prefix_out.pref_te.prefix_log,
            working_intervals=worktime_id_out.worktime_intervals,
            X_te=transform_out.X_te_t,
            y_pred=eval_out.y_pred,
            y_true=eval_out.y_true,
            metrics=eval_out.metrics,
            artifact_path=str(artifact_path) if artifact_path else None
        )
    
        

    # --------------- helpers ---------------
    @staticmethod
    def _set_global_seed(seed: int = 42) -> None:
        # Python
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # TensorFlow
        tf.random.set_seed(seed)

        # Ensure deterministic ops where possible (TF ≥ 2.9)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()

    @staticmethod
    def _load_config(cfg: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(cfg, dict):
            return cfg
        path = Path(cfg)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yml", ".yaml"):
            try:
                import yaml  # PyYAML
            except Exception as e:
                raise RuntimeError("YAML config provided but PyYAML is not installed.") from e
            return yaml.safe_load(text) or {}
        if suffix == ".json":
            return json.loads(text)
        # Try YAML fallback
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except Exception:
            # Try JSON fallback
            return json.loads(text)

    def _log(self, msg: str) -> None:
        if self.verbose:
            formatted = f"[Experiment]\t    {msg}"
            if self._logger:
                self._logger(formatted)
            else:
                print(formatted)

    def _save_experiment_results(self, cfg: Dict[str, Any], metrics: Dict[str, float]) -> Optional[Path]:
        out_cfg = cfg.get("output", {}) or {}
        save_path = out_cfg.get("path")
        save_name = out_cfg.get("name", "experiment")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if not save_path:
            Path("artifacts").mkdir(parents=True, exist_ok=True)
            save_path = f"artifacts/{save_name}_{ts}.json"

        path = Path(str(f"{save_path}/{save_name}_{ts}.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp_utc": ts, "config": cfg, "metrics": metrics}

        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
