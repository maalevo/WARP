# 
#formation/worktime_mappers.py

from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ------------ Specification ------------
@dataclass(frozen=True)
class WorktimeMappingSpec:
    worktime_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
    worktime_anchor: pd.Timestamp | None = pd.Timestamp("1970-01-01", tz="UTC")
    period_len: str = "week"
    week_anchor: str = "MON"
    tz: str | None = None
    clocktime_col: str = "time:timestamp"
    worktime_col: str = "worktime:timestamp"
    
# ------------ Result ------------
@dataclass(frozen=True)
class ToWorktimeTransformationResult:
    wt_log: pd.DataFrame
    worktime_col: str

@dataclass(frozen=True)
class FromWorktimeTransformationResult:
    ct_log: pd.DataFrame
    clocktime_col: str


# ------------ Helper ------------
def start_of_week(ct_ts: pd.Series, week_anchor: str) -> pd.Series:
    weekday_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    anchor_wd = weekday_map[week_anchor.upper()]
    day_start = ct_ts.dt.normalize()
    delta_days = (day_start.dt.dayofweek - anchor_wd) % 7
    return day_start - pd.to_timedelta(delta_days, unit="D")

# ---------------------------------------
# ------------ Mapping logic ------------
# ---------------------------------------

# ------------ Clocktime to Worktime ------------

def map_clocktime_to_worktime_timestamp(ct_timestamp, spec: WorktimeMappingSpec) -> pd.Timestamp:
    worktime_intervals = spec.worktime_intervals
    worktime_anchor = spec.worktime_anchor
    period_len = spec.period_len
    week_anchor = spec.week_anchor
    tz = spec.tz

    wt_timestamp = pd.Timedelta(0)
    total_worktime_per_period = sum(((end - start) for start, end in worktime_intervals), pd.Timedelta(0))

    # Full scope
    if period_len.lower() == "day":
        worktime_anchor_normalized = worktime_anchor.normalize() # start of day
        full_days = int((ct_timestamp - worktime_anchor_normalized) / pd.Timedelta(days=1))
        full_scope_time = pd.Timedelta(days=full_days)
        wt_timestamp += total_worktime_per_period * full_days
        
    elif period_len.lower() == "week":
        worktime_anchor_normalized = worktime_anchor.normalize() - pd.Timedelta(days=worktime_anchor.normalize().weekday()) # start of week
        full_weeks = int((ct_timestamp - worktime_anchor_normalized) / pd.Timedelta(weeks=1))
        full_scope_time = pd.Timedelta(weeks=full_weeks)
        wt_timestamp += total_worktime_per_period * full_weeks
    else:
        raise ValueError(f"Unsupported period_len: {period_len}")
    
    # Last scope
    last_period_start = worktime_anchor_normalized + full_scope_time
    ct_in_last_period = ct_timestamp - last_period_start
    for start, end in worktime_intervals:
        if ct_in_last_period >= end:
            wt_timestamp += (end - start)
        elif ct_in_last_period > start:
            wt_timestamp += (ct_in_last_period - start)

    return wt_timestamp
    
def map_clocktime_to_worktime_interval(ct_interval, ct_interval_start, spec: WorktimeMappingSpec) -> pd.Timedelta:
    ct_1 = ct_interval_start
    ct_2 = ct_interval_start + ct_interval

    wt_start = map_clocktime_to_worktime_timestamp(ct_1, spec)
    wt_end = map_clocktime_to_worktime_timestamp(ct_2, spec)
    return wt_end - wt_start

def map_clocktime_to_worktime_timestamp_log(ct_log: Union[pd.DataFrame, pd.Series], spec: WorktimeMappingSpec) -> Union[pd.DataFrame, pd.Series]:
    ct_col = spec.clocktime_col
    wt_col = spec.worktime_col

    if isinstance(ct_log, pd.Series):
        if ct_log.empty:
            return pd.Series(dtype="timedelta64[ns]", index=ct_log.index, name=wt_col)
            
        wt_series = ct_log.apply(lambda x: map_clocktime_to_worktime_timestamp(x, spec))
        return pd.to_timedelta(wt_series)

    elif isinstance(ct_log, pd.DataFrame):
        wt_log = ct_log.copy()
        
        if wt_log.empty:
            wt_log[wt_col] = pd.Series(dtype="timedelta64[ns]", index=wt_log.index)
            return wt_log
            
        wt_series = wt_log[ct_col].apply(lambda x: map_clocktime_to_worktime_timestamp(x, spec))
        wt_log[wt_col] = pd.to_timedelta(wt_series)
        return wt_log

    else:
        raise TypeError("ct_log must be a pandas DataFrame or Series")    
    
def map_clocktime_to_worktime_interval_series(ct_intervals: pd.Series, ct_interval_starts: pd.Series, spec: WorktimeMappingSpec) -> pd.Series:
    return pd.Series([
        map_clocktime_to_worktime_interval(ct_interval, ct_interval_start, spec)
        for ct_interval, ct_interval_start in zip(ct_intervals, ct_interval_starts)
    ])

# ------------ Worktime to Clocktime ------------
def map_worktime_to_clocktime_timestamp(wt_timestamp, spec: WorktimeMappingSpec) -> pd.Timestamp:
    worktime_intervals = spec.worktime_intervals
    worktime_anchor = spec.worktime_anchor
    period_len = spec.period_len
    week_anchor = spec.week_anchor
    tz = spec.tz
    
    total_worktime_per_period = sum(((end - start) for start, end in spec.worktime_intervals), pd.Timedelta(0))

    # Full scopes
    if period_len.lower() == "day":
        worktime_anchor_normalized = worktime_anchor.normalize() # start of day
        full_days = int(wt_timestamp / total_worktime_per_period)
        full_scope_ct = pd.Timedelta(days=full_days)
        full_scope_wt = total_worktime_per_period * full_days

    elif period_len.lower() == "week":
        worktime_anchor_normalized = worktime_anchor.normalize() - pd.Timedelta(days=worktime_anchor.normalize().weekday()) # start of week
        full_weeks = int(wt_timestamp / total_worktime_per_period)
        full_scope_ct = pd.Timedelta(weeks=full_weeks)
        full_scope_wt = total_worktime_per_period * full_weeks
    else:
        raise ValueError(f"Unsupported period_len: {period_len}")

    # Last scope
    worktime_intervals = sorted(worktime_intervals, key=lambda x: x[0])
    remainder_wt = wt_timestamp - full_scope_wt
    
    last_scope_wt = pd.Timedelta(0)
    last_scope_ct = pd.Timedelta(0)
    for start, end in worktime_intervals:
        wt_interval_duration = end - start
        if remainder_wt < (last_scope_wt + wt_interval_duration):
            last_scope_ct += start + (remainder_wt - last_scope_wt)
            break
        last_scope_wt += wt_interval_duration

    ct_timestamp = worktime_anchor_normalized + full_scope_ct + last_scope_ct
    return ct_timestamp

def map_worktime_to_clocktime_timestamp_series(wt_timestamps: pd.Series, spec: WorktimeMappingSpec) -> pd.Series:
    if wt_timestamps.empty:
        return pd.Series(dtype="datetime64[ns]", index=wt_timestamps.index, name=spec.clocktime_col)
    return wt_timestamps.apply(lambda x: map_worktime_to_clocktime_timestamp(x, spec))

def map_worktime_to_clocktime_interval(wt_interval, wt_interval_start, spec: WorktimeMappingSpec) -> pd.Timedelta:    
    wt_1 = wt_interval_start
    wt_2 = wt_interval_start + wt_interval

    ct_start = map_worktime_to_clocktime_timestamp(wt_1, spec)
    ct_end = map_worktime_to_clocktime_timestamp(wt_2, spec)
    return ct_end - ct_start

def map_worktime_to_clocktime_timestamp_log(wt_log: Union[pd.DataFrame, pd.Series], spec: WorktimeMappingSpec) -> Union[pd.DataFrame, pd.Series]:
    ct_col = spec.clocktime_col
    wt_col = spec.worktime_col

    if isinstance(wt_log, pd.Series):
        # Series → Series
        if wt_log.empty:
            return pd.Series(dtype="datetime64[ns]", index=wt_log.index, name=ct_col)
        return wt_log.apply(
            lambda x: map_worktime_to_clocktime_timestamp(x, spec)
        )

    elif isinstance(wt_log, pd.DataFrame):
        # DataFrame → DataFrame
        ct_log = wt_log.copy()
        if ct_log.empty:
            ct_log[ct_col] = pd.Series(dtype="datetime64[ns]", index=ct_log.index)
            return ct_log
        ct_log[ct_col] = ct_log[wt_col].apply(
            lambda x: map_worktime_to_clocktime_timestamp(x, spec)
        )
        return ct_log

    else:
        raise TypeError(
            "wt_log must be a pandas DataFrame or Series"
        )
    
def map_worktime_to_clocktime_interval_series(wt_intervals: pd.Series, wt_interval_starts: pd.Series, spec: WorktimeMappingSpec) -> pd.Series:
    return pd.Series([
        map_worktime_to_clocktime_interval(wt_interval, wt_interval_start, spec)
        for wt_interval, wt_interval_start in zip(wt_intervals, wt_interval_starts)
    ])