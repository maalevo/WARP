# worktime_transformation/worktime_identifiers.py

from __future__ import annotations
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# ------------ Specification ------------
@dataclass(frozen=True)
class WorktimeIdentificationSpec:
    bucket_len: int | pd.Timedelta = 60*60  # seconds (default: 1 hour)
    period_len: str = "week"
    threshold: float = 0.2
    threshold_ref: str = "max"  # max, mean, median, perc95, perc90
    timestamp_col: str = "time:timestamp"
    week_anchor: str = "MON"
    tz: str | None = None

# ------------ Result ------------
@dataclass(frozen=True)
class WorktimeIdentificationResult:
    worktime_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
    count_per_bucket: pd.DataFrame
    id_period_len: str


# ------------ Identification logic ------------

def identify_discrete_worktime_intervals(log: pd.DataFrame, spec: WorktimeIdentificationSpec) -> WorktimeIdentificationResult:
    """
    Identify worktime intervals based on event density in specified time windows.

    Args:
      log: DataFrame containing event log with timestamps.
      spec: WorktimeIdentificationSpec with configuration parameters.
    """
    bucket_len = spec.bucket_len
    period_len = spec.period_len
    threshold = spec.threshold
    threshold_ref = spec.threshold_ref
    timestamp_col = spec.timestamp_col
    tz = spec.tz
    
    df = log.copy()
    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="raise")
    if tz is not None:
        ts = ts.dt.tz_convert(tz)

    if isinstance(bucket_len, int):
        bucket_len = pd.to_timedelta(bucket_len, unit='s')


    # Count number of events per bucket
    if period_len.lower() == "day":
        period_len = pd.Timedelta(days=1)
        period_starts = ts.dt.floor("D")
    elif period_len.lower() == "week":
        period_len = pd.Timedelta(weeks=1)

        weekday_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
        anchor_wd = weekday_map.get(getattr(spec, "week_anchor", "MON").upper(), 0)

        day_start = ts.dt.normalize()  # keeps tz
        delta_days = (day_start.dt.weekday - anchor_wd) % 7

        period_starts = day_start - pd.to_timedelta(delta_days, unit="D")
    else:
        raise ValueError(f"Unsupported period_len: {period_len}")
    
    within_period = ts - period_starts
    bucket = (within_period // bucket_len) * bucket_len
    counts = bucket.value_counts()

    all_buckets = pd.timedelta_range(start="0s", end=period_len - bucket_len, freq=bucket_len)
    count_per_bucket = pd.DataFrame({"window_start": all_buckets})
    count_per_bucket["window_end"] = count_per_bucket["window_start"] + bucket_len
    count_per_bucket["n_events"] = count_per_bucket["window_start"].map(counts).fillna(0).astype(int)
    
    # Identify worktime intervals based on threshold and bucket counts
    if threshold_ref == "max":
        ref_count = count_per_bucket["n_events"].max()
    elif threshold_ref == "mean":
        ref_count = count_per_bucket["n_events"].mean()
    elif threshold_ref == "median":
        ref_count = count_per_bucket["n_events"].median()
    elif threshold_ref == "perc95":
        ref_count = np.percentile(count_per_bucket["n_events"], 95)
    elif threshold_ref == "perc90":
        ref_count = np.percentile(count_per_bucket["n_events"], 90)
    worktime_intervals = []

    count_per_bucket = count_per_bucket.sort_values("window_start").reset_index(drop=True)

    current_start = None
    current_end = None

    for _, row in count_per_bucket.iterrows():
        is_active = (ref_count > 0) and ((row["n_events"] / ref_count) >= threshold)

        if is_active:
            if current_start is None:
                # start a new run
                current_start = row["window_start"]
                current_end = row["window_end"]
            else:
                # extend run if adjacent; otherwise close and start new
                if row["window_start"] == current_end:
                    current_end = row["window_end"]
                else:
                    worktime_intervals.append((current_start, current_end))
                    current_start = row["window_start"]
                    current_end = row["window_end"]
        else:
            # close any open run
            if current_start is not None:
                worktime_intervals.append((current_start, current_end))
                current_start = None
                current_end = None
    # close trailing run
    if current_start is not None:
        worktime_intervals.append((current_start, current_end))

        
    # Return result
    return WorktimeIdentificationResult(
        worktime_intervals=worktime_intervals,
        count_per_bucket=count_per_bucket,
        id_period_len=period_len
    )
    

# ------------ Result plotting ------------

