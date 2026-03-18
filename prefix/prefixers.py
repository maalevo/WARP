from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PrefixSpec:
    min_prefix_len: int = 1
    max_prefix_len: Optional[int] = None
    drop_last_event: bool = True
    case_id_col: str = "case:concept:name"
    activity_col: str = "concept:name"
    time_col: str = "time:timestamp"
    worktime_col: str = "worktime:timestamp"

@dataclass(frozen=True)
class PrefixResult:
    prefix_log: pd.DataFrame
    ct_remaining_time: pd.Series               
    ct_time_until_next_event: pd.Series   
    wt_remaining_time: pd.Series
    wt_time_until_next_event: pd.Series

def build_prefix_log(log: pd.DataFrame, prefix_spec: PrefixSpec) -> PrefixResult:
    min_prefix_len = prefix_spec.min_prefix_len
    max_prefix_len = prefix_spec.max_prefix_len
    drop_last_event = prefix_spec.drop_last_event
    case_id_col = prefix_spec.case_id_col
    activity_col = prefix_spec.activity_col
    time_col = prefix_spec.time_col
    worktime_col = prefix_spec.worktime_col

    missing = {activity_col, case_id_col, time_col} - set(log.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    log = log.copy()
    log[time_col] = pd.to_datetime(log[time_col], utc=False, errors="coerce")
    if log[time_col].isna().any():
        bad = log[log[time_col].isna()]
        raise ValueError(f"NaT in {time_col}:\n{bad.head()}")

    log = (
        log.sort_values([case_id_col, time_col, activity_col])
           .reset_index(drop=True)
    )

    prefix_parts = []
    ct_rem_time_map = {}   
    ct_next_time_map = {}  
    wt_rem_time_map = {}   
    wt_next_time_map = {}  
    
    for orig_case, g in log.groupby(case_id_col, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        
        ct_ts = g[time_col].to_numpy()
        wt_ts = g[worktime_col].to_numpy()
        ct_case_end = ct_ts[-1]
        wt_case_end = wt_ts[-1]
        
        limit_n = n - 1 if drop_last_event else n
        
        if max_prefix_len is None:
            eff_max = limit_n
        else:
            eff_max = min(int(max_prefix_len), limit_n)

        min_k = max(1, int(min_prefix_len))
        max_k = max(0, eff_max)

        for k in range(min_k, max_k + 1):
            pref_case_id = f"{orig_case}::prefix_{k:04d}"
            pref_rows = g.iloc[:k].copy()
            pref_rows[case_id_col] = pref_case_id
            pref_rows["prefix_length"] = k
            prefix_parts.append(pref_rows)

            last_ct_ts = ct_ts[k - 1]
            last_wt_ts = wt_ts[k - 1]
            
            # Calculate targets using consistent numpy timedelta division for seconds
            ct_rem_time_map[pref_case_id] = (ct_case_end - last_ct_ts) / np.timedelta64(1, "s")
            wt_rem_time_map[pref_case_id] = (wt_case_end - last_wt_ts) / np.timedelta64(1, "s")
            
            # Predict the next event timestamp or default to zero at the end of the trace
            if k < n:
                ct_next_time_map[pref_case_id] = (ct_ts[k] - last_ct_ts) / np.timedelta64(1, "s")
                wt_next_time_map[pref_case_id] = (wt_ts[k] - last_wt_ts) / np.timedelta64(1, "s")
            else:
                ct_next_time_map[pref_case_id] = 0.0
                wt_next_time_map[pref_case_id] = 0.0

    if not prefix_parts:
        empty_df = log.iloc[0:0].assign(prefix_length=pd.Series(dtype="int64"))
        return PrefixResult(
            prefix_log=empty_df,
            ct_remaining_time=pd.Series([], dtype="float64", name="y_ct_remaining_seconds"),
            ct_time_until_next_event=pd.Series([], dtype="float64", name="y_ct_time_until_next_seconds"),
            wt_remaining_time=pd.Series([], dtype="float64", name="y_wt_remaining_seconds"),
            wt_time_until_next_event=pd.Series([], dtype="float64", name="y_wt_time_until_next_seconds"),
        )

    prefix_log = pd.concat(prefix_parts, ignore_index=True)

    original_cols = list(log.columns)
    if "prefix_length" not in prefix_log.columns:
        prefix_log["prefix_length"] = pd.Series(dtype="int64")
    prefix_log = prefix_log[original_cols + ["prefix_length"]]

    ct_rem_per_prefix = pd.Series(ct_rem_time_map, dtype="float64", name="y_ct_remaining_seconds")
    ct_next_per_prefix = pd.Series(ct_next_time_map, dtype="float64", name="y_ct_time_until_next_seconds")
    wt_rem_per_prefix = pd.Series(wt_rem_time_map, dtype="float64", name="y_wt_remaining_seconds")
    wt_next_per_prefix = pd.Series(wt_next_time_map, dtype="float64", name="y_wt_time_until_next_seconds")

    ct_remaining_time_row = prefix_log[case_id_col].map(ct_rem_per_prefix)
    ct_time_until_next_row = prefix_log[case_id_col].map(ct_next_per_prefix)
    wt_remaining_time_row = prefix_log[case_id_col].map(wt_rem_per_prefix)
    wt_time_until_next_row = prefix_log[case_id_col].map(wt_next_per_prefix)

    ct_remaining_time_row.index = prefix_log.index
    ct_time_until_next_row.index = prefix_log.index
    wt_remaining_time_row.index = prefix_log.index
    wt_time_until_next_row.index = prefix_log.index

    return PrefixResult(
        prefix_log=prefix_log,
        ct_remaining_time=ct_remaining_time_row,
        ct_time_until_next_event=ct_time_until_next_row,
        wt_remaining_time=wt_remaining_time_row,
        wt_time_until_next_event=wt_time_until_next_row
    )