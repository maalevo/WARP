# feature/feature_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np

from worktime_mapping.worktime_mappers import  WorktimeMappingSpec, map_worktime_to_clocktime_timestamp_series, map_clocktime_to_worktime_interval_series

# -----------------------------
# Public API
# -----------------------------
@dataclass(frozen=True)
class FeatureSpec:
    feature_key: str  # e.g., "activity", "time_since_case_start", ...
    params: Optional[dict] = None  # feature-specific parameters


def build_feature(
    prefix_log: pd.DataFrame, 
    feature_spec: FeatureSpec,
    fit_states: Optional[Dict[str, List[NormState]]] = None
) -> Tuple[pd.DataFrame, Dict[str, List[NormState]]]:
    """
    Returns:
      - feature_df: The processed features.
      - out_states: Dictionary mapping column names to their list of applied NormStates.
    """
    feature_key = feature_spec.feature_key
    params = feature_spec.params or {}
    normalization_steps = params.get("normalization", [])

    # 1. Generate the base feature
    if feature_key == "raw":
        feature_df = build_raw_feature(prefix_log, params)
    elif feature_key == "time_since_case_start":
        feature_df = build_time_since_case_start_feature(prefix_log, params)
    elif feature_key == "time_since_last_event":
        feature_df = build_time_since_last_event_feature(prefix_log, params)
    elif feature_key == "time_in_day":
        feature_df = build_time_in_day_feature(prefix_log, params)
    elif feature_key == "time_in_week":
        feature_df = build_time_in_week_feature(prefix_log, params)
    elif feature_key == "worktime_in_day":
        feature_df = build_worktime_in_day_feature(prefix_log, params)
    elif feature_key == "worktime_in_week":
        feature_df = build_worktime_in_week_feature(prefix_log, params)
    elif feature_key == "coherence":
        feature_df = build_coherence_feature(prefix_log, params)
    else:
        raise ValueError(f"Unknown feature key: {feature_key}")

    # 2. Apply Normalization sequentially to columns in feature_df
    out_states = fit_states or {}
    
    if normalization_steps:
        for col in feature_df.columns:
            # Skip normalization for onehot or cyclical columns (already normalized-ish)
            # unless explicitly desired. Usually we only normalize 'numeric_' cols.
            if not col.startswith("numeric_"):
                continue

            col_states = []
            for i, strategy in enumerate(normalization_steps):
                existing_state = fit_states[col][i] if (fit_states and col in fit_states) else None
                
                feature_df[col], state = apply_stateful_norm(feature_df[col], strategy, existing_state)
                col_states.append(state)
            
            if not fit_states:
                out_states[col] = col_states

    return feature_df, out_states


# -----------------------------
# Feature implementations
# -----------------------------
def build_raw_feature(prefix_log: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Encodes the current activity per row.
    """
    source_col_name = params.get("source_col_name", None)
    encoding = params.get("encoding", None)
    categories = params.get("categories", None)
    col_prefix = params.get("prefix", "raw")

    _require_cols(prefix_log, [source_col_name])

    s = prefix_log[source_col_name].astype("string")

    if encoding == "onehot":
        return to_onehot_encoding(s, categories=categories, prefix=col_prefix, index=prefix_log.index)
    elif encoding == "numeric":
        return to_numeric_encoding(s, categories=categories, feature_name=col_prefix, index=prefix_log.index)
    else:
        raise ValueError(f"Unknown encoding for raw feature: {encoding}")


def build_time_since_case_start_feature(prefix_log: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Time since the first event of the case (per row)."""
    case_id_col = params.get("case_id_col", "case:concept:name")
    time_col = params.get("time_col", "time:timestamp")
    unit = params.get("unit", "seconds")
    feature_name = params.get("feature_name", "t_since_case_start")

    _require_cols(prefix_log, [case_id_col, time_col])

    ts = _to_datetime(prefix_log[time_col])
    case_start = ts.groupby(prefix_log[case_id_col]).transform("min")
    delta = ts - case_start

    x = _timedelta_to_unit(delta, unit).astype(float)
    feature_name = "numeric_" + feature_name
    return pd.DataFrame({feature_name: x}, index=prefix_log.index)


def build_time_since_last_event_feature(prefix_log: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Time since previous event in the same case (per row)."""
    case_id_col = params.get("case_id_col", "case:concept:name")
    time_col = params.get("time_col", "time:timestamp")
    unit = params.get("unit", "seconds")
    first_event_value = float(params.get("first_event_value", 0.0))
    feature_name = params.get("feature_name", "t_since_last_event")
    assume_sorted = bool(params.get("assume_sorted", False))

    _require_cols(prefix_log, [case_id_col, time_col])

    #print(prefix_log[time_col].head())
    #ts = _to_datetime(prefix_log[time_col])
    ts = prefix_log[time_col]
    #print(ts.head())

    if assume_sorted:
        prev = ts.groupby(prefix_log[case_id_col]).shift(1)
        delta = ts - prev
        x = _timedelta_to_unit(delta, unit)
        x = x.fillna(first_event_value).astype(float)
    else:
        tmp = prefix_log[[case_id_col]].copy()
        tmp["_ts_"] = ts
        tmp["_idx_"] = prefix_log.index
        tmp = tmp.sort_values([case_id_col, "_ts_", "_idx_"])
        tmp["_prev_ts_"] = tmp.groupby(case_id_col)["_ts_"].shift(1)
        tmp["_delta_"] = tmp["_ts_"] - tmp["_prev_ts_"]
        tmp["_x_"] = _timedelta_to_unit(tmp["_delta_"], unit).fillna(first_event_value).astype(float)
        x = pd.Series(tmp["_x_"].to_numpy(), index=tmp["_idx_"]).reindex(prefix_log.index)

    feature_name = "numeric_" + feature_name
    return pd.DataFrame({feature_name: x}, index=prefix_log.index)

def build_time_in_day_feature(prefix_log: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Time-of-day feature.
    
    Cyclical Logic:
      The period is always exactly 1 day (converted to the selected unit).
      e.g., if unit is 'hours', period is 24.
    """
    time_col = params.get("time_col", "time:timestamp")
    unit = params.get("unit", "seconds")
    encoding = params.get("encoding", "numeric")
    feature_name = params.get("feature_name", "time_in_day")
    prefix = params.get("prefix", feature_name)
    categories = params.get("categories", None)

    _require_cols(prefix_log, [time_col])
    ts = _to_datetime(prefix_log[time_col])

    # 1. Calculate the numeric value and the fixed period (1 day)
    unit_l = str(unit).lower()
    
    if unit_l in ("s", "sec", "second", "seconds"):
        value = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).astype(int)
        period = 86400  # 24 * 60 * 60
    elif unit_l in ("m", "min", "minute", "minutes"):
        value = (ts.dt.hour * 60 + ts.dt.minute).astype(int)
        period = 1440   # 24 * 60
    elif unit_l in ("h", "hr", "hour", "hours"):
        value = ts.dt.hour.astype(int)
        period = 24
    else:
        raise ValueError(f"Unknown unit for time_in_day: {unit}. Use hours, minutes, or seconds.")

    # 2. Apply Encoding
    if encoding == "numeric":
        feature_name = "numeric_" + feature_name
        return pd.DataFrame({feature_name: value.astype(float)}, index=prefix_log.index)

    if encoding == "onehot":
        if categories is None:
            # For onehot, we usually range up to the period
            categories = [str(i) for i in range(period)]
        feature_name = "onehot_" + feature_name
        return to_onehot_encoding(
            value.astype("string"), 
            categories=categories, 
            prefix=prefix, 
            index=prefix_log.index
        )
    
    if encoding == "cyclical":
        feature_name = "cyclical_" + feature_name
        # STRICTLY use the theoretical period, ignore data max
        return to_cyclical_encoding(
            value.astype(float), 
            period=period, 
            prefix=feature_name, 
            index=prefix_log.index
        )

    raise ValueError(f"Unknown encoding: {encoding}")


def build_time_in_week_feature(prefix_log: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Time-in-week feature.

    Cyclical Logic:
      The period is always exactly 1 week (converted to the selected unit).
      e.g., if unit is 'days', period is 7.
    """
    time_col = params.get("time_col", "time:timestamp")
    unit = params.get("unit", "seconds")
    encoding = params.get("encoding", "numeric")
    feature_name = params.get("feature_name", "time_in_week")
    prefix = params.get("prefix", feature_name)
    categories = params.get("categories", None)

    _require_cols(prefix_log, [time_col])
    ts = _to_datetime(prefix_log[time_col])

    weekday = ts.dt.weekday.astype(int)

    # 1. Calculate numeric value and fixed period (1 week)
    unit_l = str(unit).lower()
    
    if unit_l in ("s", "sec", "second", "seconds"):
        value = (weekday * 86400 + ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).astype(int)
        period = 604800  # 7 * 24 * 60 * 60
    elif unit_l in ("m", "min", "minute", "minutes"):
        value = (weekday * 1440 + ts.dt.hour * 60 + ts.dt.minute).astype(int)
        period = 10080   # 7 * 24 * 60
    elif unit_l in ("h", "hr", "hour", "hours"):
        value = (weekday * 24 + ts.dt.hour).astype(int)
        period = 168     # 7 * 24
    elif unit_l in ("d", "day", "days"):
        value = weekday.astype(int)
        period = 7
    else:
        raise ValueError(f"Unknown unit: {unit}")

    # 2. Apply Encoding
    if encoding == "numeric":
        feature_name = "numeric_" + feature_name
        return pd.DataFrame({feature_name: value.astype(float)}, index=prefix_log.index)

    if encoding == "onehot":
        if categories is None:
            categories = [str(i) for i in range(period)]
        feature_name = "onehot_" + feature_name
        return to_onehot_encoding(
            value.astype("string"), 
            categories=categories, 
            prefix=prefix, 
            index=prefix_log.index
        )

    if encoding == "cyclical":
        feature_name = "cyclical_" + feature_name
        # STRICTLY use the theoretical period, ignore data max
        return to_cyclical_encoding(
            value.astype(float), 
            period=period, 
            prefix=feature_name, 
            index=prefix_log.index
        )

    raise ValueError(f"Unknown encoding: {encoding}")

def build_worktime_in_day_feature(prefix_log: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Worktime-in-day feature."""
    worktime_col = params.get("worktime_col", "worktime:timestamp")
    worktime_mapping_spec = params.get("worktime_mapping_spec", None)
    unit = params.get("unit", "seconds")
    encoding = params.get("encoding", "numeric")
    feature_name = params.get("feature_name", "worktime_in_day")
    prefix = params.get("prefix", feature_name)
    
    _require_cols(prefix_log, [worktime_col])
    ts = _to_timedelta(prefix_log[worktime_col])

    if worktime_mapping_spec is None:
        raise ValueError("worktime_mapping_spec is required for worktime_in_day feature.")
    
    ct_timestamps = map_worktime_to_clocktime_timestamp_series(ts, worktime_mapping_spec)
    ct_day_starts = ct_timestamps.dt.normalize()

    ct_intervals = ct_timestamps - ct_day_starts
    wt_intervals = map_clocktime_to_worktime_interval_series(ct_intervals, ct_day_starts, worktime_mapping_spec)
    
    values = _timedelta_to_unit(wt_intervals, unit).astype(float)
    
    if encoding == "numeric":
        final_col_name = f"numeric_{prefix}" 
        return pd.DataFrame({final_col_name: values}, index=prefix_log.index)

    raise ValueError(f"Unknown encoding: {encoding}")


def build_worktime_in_week_feature(prefix_log: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Worktime-in-week feature."""
    worktime_col = params.get("worktime_col", "worktime:timestamp")
    worktime_mapping_spec = params.get("worktime_mapping_spec", None)
    week_starts_monday = bool(params.get("week_starts_monday", True))
    unit = params.get("unit", "seconds")
    encoding = params.get("encoding", "numeric")
    feature_name = params.get("feature_name", "worktime_in_week")
    prefix = params.get("prefix", feature_name)
    
    _require_cols(prefix_log, [worktime_col])
    ts = _to_timedelta(prefix_log[worktime_col])

    if worktime_mapping_spec is None:
        raise ValueError("worktime_mapping_spec is required for worktime_in_week feature.")
    
    ct_timestamps = map_worktime_to_clocktime_timestamp_series(ts, worktime_mapping_spec)
    
    ct_timestamps_tz_naive = ct_timestamps.dt.tz_localize(None) if ct_timestamps.dt.tz is not None else ct_timestamps
    
    if week_starts_monday:
        ct_week_starts = ct_timestamps_tz_naive.dt.to_period('W-MON').dt.start_time
    else:
        ct_week_starts = ct_timestamps_tz_naive.dt.to_period('W-SUN').dt.start_time
        
    if ct_timestamps.dt.tz is not None:
         ct_week_starts = ct_week_starts.dt.tz_localize(ct_timestamps.dt.tz)

    ct_intervals = ct_timestamps - ct_week_starts
    wt_intervals = map_clocktime_to_worktime_interval_series(ct_intervals, ct_week_starts, worktime_mapping_spec)
    
    values = _timedelta_to_unit(wt_intervals, unit).astype(float)
    
    if encoding == "numeric":
        final_col_name = f"numeric_{prefix}"
        return pd.DataFrame({final_col_name: values}, index=prefix_log.index)

    raise ValueError(f"Unknown encoding: {encoding}")


def build_coherence_feature(prefix_log: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Coherence per event."""
    attribute_cols = params.get("attribute_cols")
    if not attribute_cols or not isinstance(attribute_cols, (list, tuple)):
        raise ValueError('coherence requires params["attribute_cols"] as a non-empty list of column names.')

    case_id_col = params.get("case_id_col", "case:concept:name")
    time_col = params.get("time_col", "time:timestamp")
    assume_sorted = bool(params.get("assume_sorted", False))
    first_event_value = float(params.get("first_event_value", 1.0))
    feature_name = params.get("feature_name", "coherence")

    _require_cols(prefix_log, [case_id_col] + list(attribute_cols))

    if (not assume_sorted) and (time_col in prefix_log.columns):
        ts = _to_datetime(prefix_log[time_col])
        tmp = prefix_log[[case_id_col] + list(attribute_cols)].copy()
        tmp["_ts_"] = ts
        tmp["_idx_"] = prefix_log.index
        tmp = tmp.sort_values([case_id_col, "_ts_", "_idx_"])
        coh_sorted = _coherence_from_prev(tmp, case_id_col, attribute_cols, first_event_value)
        coh = pd.Series(coh_sorted.to_numpy(), index=tmp["_idx_"]).reindex(prefix_log.index)
    else:
        coh = _coherence_from_prev(prefix_log, case_id_col, attribute_cols, first_event_value)

    feature_name = "numeric_" + feature_name
    return pd.DataFrame({feature_name: coh.astype(float)}, index=prefix_log.index)


def _coherence_from_prev(df: pd.DataFrame, case_id_col: str, attribute_cols: list[str], first_event_value: float) -> pd.Series:
    k = len(attribute_cols)
    changed_counts = pd.Series(0, index=df.index, dtype="int32")
    g = df.groupby(case_id_col, sort=False)
    for col in attribute_cols:
        cur = df[col]
        prev = g[col].shift(1)
        same = (cur == prev) | (cur.isna() & prev.isna())
        changed = ~same
        changed_counts = changed_counts + changed.fillna(False).astype("int32")

    coherence = 1.0 - (changed_counts.astype(float) / float(k))
    is_first = g.cumcount() == 0
    coherence = coherence.where(~is_first, other=first_event_value)
    return coherence.clip(0.0, 1.0)


# -----------------------------
# Encodings
# -----------------------------
def to_numeric_encoding(s: pd.Series, unknown_value: int = -1, feature_name: str = "x", index: Optional[pd.Index] = None) -> pd.DataFrame:
    if index is None: index = s.index
    out = pd.to_numeric(s, errors='coerce')
    out = out.fillna(unknown_value).astype(int)
    col_name = f"numeric_{feature_name}"
    return pd.DataFrame({col_name: out}, index=index)


def to_onehot_encoding(s: pd.Series, categories: Optional[List[str]] = None, prefix: str = "x", index: Optional[pd.Index] = None) -> pd.DataFrame:
    if index is None: index = s.index
    s_str = s.astype("string")
    expected_cols = None
    if categories is not None:
        cats = [str(c) for c in categories]
        s_str = pd.Categorical(s_str, categories=cats)
        expected_cols = [f"{prefix}_{c}" for c in cats]

    d = pd.get_dummies(s_str, prefix=prefix, dtype=np.int8)
    d = d.reindex(index=index, fill_value=0)
    if expected_cols is not None:
        d = d.reindex(columns=expected_cols, fill_value=0)
    return d

def to_cyclical_encoding(
    s: pd.Series, 
    period: float, 
    prefix: str = "x", 
    index: Optional[pd.Index] = None
) -> pd.DataFrame:
    """
    Encodes a numeric series into two cyclical features: sine and cosine.
    
    Formula: 
      sin = sin(2 * pi * x / period)
      cos = cos(2 * pi * x / period)
      
    Args:
      period: The theoretical cycle length (e.g., 24 for hours in a day).
              Do NOT use the max value of the data here.
    """
    if index is None:
        index = s.index
        
    if period == 0:
        period = 1.0

    # Ensure s is numeric and handle NaN as 0 (or you might prefer keeping them NaN)
    s = pd.to_numeric(s, errors='coerce').fillna(0.0)

    val_sin = np.sin(2 * np.pi * s / period)
    val_cos = np.cos(2 * np.pi * s / period)

    return pd.DataFrame({
        f"{prefix}_sin": val_sin,
        f"{prefix}_cos": val_cos
    }, index=index)


# -----------------------------
# Normalization (Stateful)
# -----------------------------
@dataclass
class NormState:
    strategy: str
    params: Dict[str, float]

def apply_stateful_norm(series: pd.Series, strategy: str, state: Optional[NormState] = None) -> Tuple[pd.Series, NormState]:
    if strategy == "minmax":
        if state is None:
            s_min, s_max = series.min(), series.max()
            state = NormState("minmax", {"min": s_min, "max": s_max})
        p = state.params
        denom = p["max"] - p["min"]
        return (series - p["min"]) / (denom if denom != 0 else 1.0), state

    elif strategy == "standard":
        if state is None:
            state = NormState("standard", {"mean": series.mean(), "std": series.std()})
        p = state.params
        return (series - p["mean"]) / (p["std"] if (p["std"] != 0 and not np.isnan(p["std"])) else 1.0), state

    elif strategy == "log":
        return np.log1p(series.clip(lower=0)), NormState("log", {})

    return series, state

def invert_stateful_norm(
    data: Union[pd.Series, np.ndarray], 
    states: List[NormState]
) -> Union[pd.Series, np.ndarray]:
    """
    Reverses the normalization steps stored in 'states'.
    Must be applied in REVERSE order of the forward pass.
    """
    if not states:
        return data

    # Work on a copy to avoid mutating the input
    out = data.copy()

    # Iterate backwards (e.g., if Log -> Standard, we do Un-Standard -> Un-Log)
    for state in reversed(states):
        strategy = state.strategy
        p = state.params
        
        if strategy == "minmax":
            denom = p["max"] - p["min"]
            # Avoid multiplying by 0 if min == max
            scale = denom if denom != 0 else 1.0
            out = out * scale + p["min"]

        elif strategy == "standard":
            std = p["std"]
            # Handle potential 0 std or NaN saved in state
            scale = std if (std != 0 and not np.isnan(std)) else 1.0
            out = out * scale + p["mean"]

        elif strategy == "log":
            # Inverse of log1p is expm1
            out = np.expm1(out)

    return out

# -----------------------------
# Utility
# -----------------------------
def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def _to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s): return s
    return pd.to_datetime(s, errors="coerce")

def _to_timedelta(s: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(s): return s
    return pd.to_timedelta(s, errors="coerce")

def _timedelta_to_unit(delta: pd.Series, unit: str) -> pd.Series:
    if not pd.api.types.is_timedelta64_dtype(delta):
        delta = pd.to_timedelta(delta, errors="coerce")
    seconds = delta.dt.total_seconds()
    unit = unit.lower()
    if unit in ("s", "sec", "second", "seconds"): return seconds
    if unit in ("m", "min", "minute", "minutes"): return seconds / 60.0
    if unit in ("h", "hr", "hour", "hours"): return seconds / 3600.0
    if unit in ("d", "day", "days"): return seconds / 86400.0
    raise ValueError(f"Unknown unit: {unit}")