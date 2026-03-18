# split/splitters.py

from __future__ import annotations
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


# ------------ Temporal Splitter Function ------------
@dataclass(frozen=True)
class TemporalSplitSpec:
    test_ratio: float = 0.2
    val_ratio: float = 0.0
    shuffle_within_same_timestamps: bool = False
    seed: int = 42
    case_id_col: str = "case:concept:name"
    timestamp_col: str = "time:timestamp"

@dataclass(frozen=True)
class TemporalSplitResult:
    log_tr: pd.DataFrame
    log_va: pd.DataFrame
    log_te: pd.DataFrame


def temporal_split_by_case_end(
        log: pd.DataFrame,
        split_spec: TemporalSplitSpec) -> TemporalSplitResult:
    """Chronological split by each case's **last event time**.
    Ensures val/test are strictly after train in time.

    Returns lists of case_ids for train/val/test.
    """
    test_ratio = split_spec.test_ratio
    val_ratio = split_spec.val_ratio
    shuffle_within_same_timestamps = split_spec.shuffle_within_same_timestamps
    #seed = split_spec.seed
    case_id_col = split_spec.case_id_col
    timestamp_col = split_spec.timestamp_col

    log = log.copy()
    log[timestamp_col] = pd.to_datetime(log[timestamp_col], utc=True, errors='coerce')

    case_end = log.groupby(case_id_col)[timestamp_col].max().rename('case_end')
    order = case_end.sort_values(kind='mergesort')  # stable sort

    n = len(order)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(0, n - n_val - n_test)

    if shuffle_within_same_timestamps:
        rng = np.random.default_rng()
        # Shuffle only within groups of identical timestamps
        ends = order.values.astype("datetime64[ns]")
        uniq, inv, counts = np.unique(ends, return_inverse=True, return_counts=True)
        new_index = order.index.to_numpy(copy=True)
        start = 0
        for c in counts:
            if c > 1:
                rng.shuffle(new_index[start:start+c])
            start += c
        order = pd.Series(ends, index=new_index).sort_values(kind="mergesort")
        order = case_end.loc[order.index]  # restore case_end values to new index

    s = order.index[:n_train].tolist()
    val_ids   = order.index[n_train:n_train + n_val].tolist()
    test_ids  = order.index[n_train + n_val:].tolist()

    train_ids = order.index[:n_train]
    val_ids   = order.index[n_train:n_train + n_val]
    test_ids  = order.index[n_train + n_val:]

    log_tr = log[log[case_id_col].isin(train_ids)].copy()
    log_va   = log[log[case_id_col].isin(val_ids)].copy()
    log_te  = log[log[case_id_col].isin(test_ids)].copy()

    return TemporalSplitResult(log_tr, log_va, log_te)