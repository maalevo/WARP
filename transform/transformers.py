# transformers.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# -----------------------------
# Public API
# -----------------------------

@dataclass
class TransformerSpec:
    transformer_key: str  # "lstm", "last_state", "aggregation", "index_based"
    params: Dict[str, Any]

@dataclass
class TransformerState:
    transformer_key: str
    schema: Dict[str, Any]  # Stores column names, dimensions, etc.

def build_transformation(
    X: pd.DataFrame,
    y: pd.Series,
    cases: pd.Series,
    spec: TransformerSpec,
    state: Optional[TransformerState] = None
) -> Tuple[Any, Any, pd.Index, TransformerState]:  # <--- Typ-Update
    """
    Dispatcher function.
    Returns:
      - X_out: Transformed X
      - y_out: Transformed y
      - case_ids: The index identifying the cases in X_out/y_out
      - state: Schema state
    """
    key = spec.transformer_key
    params = spec.params

    if key == "lstm":
        return transform_lstm(X, y, cases, params, state)
    elif key == "last_state":
        return transform_last_state(X, y, cases, params, state)
    elif key == "aggregation":
        return transform_aggregation(X, y, cases, params, state)
    elif key == "index_based":
        return transform_index_based(X, y, cases, params, state)
    else:
        raise ValueError(f"Unknown transformer key: {key}")


# -----------------------------
# Transformer Implementations
# -----------------------------

def transform_lstm(
    X: pd.DataFrame,
    y: pd.Series,
    cases: pd.Series,
    params: Dict[str, Any],
    state: Optional[TransformerState] = None
) -> Tuple[np.ndarray, np.ndarray, pd.Index, TransformerState]: # <--- Typ-Update
    
    pad_value = params.get("pad_value", -1.0)

    # 1. State Management ... (wie gehabt)
    if state is None:
        feature_names = list(X.columns)
        state = TransformerState("lstm", {"feature_names": feature_names})
    else:
        feature_names = state.schema["feature_names"]

    # 2. Empty Handling
    if X.shape[0] == 0:
        F = len(feature_names)
        X_empty = np.full((0, 0, F), pad_value, dtype=np.float32)
        y_empty = np.full((0,), 0.0, dtype=np.float32)
        # Return empty Index
        return X_empty, y_empty, pd.Index([], name=cases.name), state 

    # 3. Validation ... (wie gehabt)
    _validate_inputs(X, y, cases)

    # 4. Transform
    # WICHTIG: Das hier ist unser Anker für die Ausrichtung
    case_order = pd.Index(cases.drop_duplicates()) 
    
    # ... (Rest der Tensor-Erstellung bleibt identisch) ...
    row_pos = cases.groupby(cases).cumcount().to_numpy(dtype=np.int32)
    case_to_i = {cid: i for i, cid in enumerate(case_order)}
    case_idx = cases.map(case_to_i).to_numpy(dtype=np.int32)

    N = int(len(case_order))
    T = int(cases.groupby(cases).size().max())
    F = int(X.shape[1])

    X_transformed = np.full((N, T, F), pad_value, dtype=np.float32)
    X_vals = X.to_numpy(dtype=np.float32, copy=False)
    X_transformed[case_idx, row_pos, :] = X_vals

    last_row_idx = cases.groupby(cases).tail(1).index
    y_last = y.loc[last_row_idx]
    
    # Hier nutzen wir case_order für das reindex, damit y auch stimmt
    y_last = y_last.groupby(cases.loc[last_row_idx]).first().reindex(case_order)
    y_transformed = y_last.to_numpy(dtype=np.float32, copy=False)

    # RETURN: X, y, case_order (Index), state
    return X_transformed, y_transformed, case_order, state

def transform_last_state(
    X: pd.DataFrame,
    y: pd.Series,
    cases: pd.Series,
    params: Dict[str, Any],
    state: Optional[TransformerState] = None
) -> Tuple[pd.DataFrame, pd.Series, TransformerState]:
    """
    Extracts the last row for each case.
    """
    # 1. Manage State
    if state is None:
        feature_names = list(X.columns)
        state = TransformerState("last_state", {"feature_names": feature_names})
    else:
        feature_names = state.schema["feature_names"]

    # 2. Empty Input Handling
    if X.shape[0] == 0:
        X_empty = pd.DataFrame(columns=feature_names, dtype=float)
        y_empty = pd.Series(dtype=float)
        return X_empty, y_empty, state

    # 3. Validation
    _validate_inputs(X, y, cases)

    # 4. Transform
    last_idx = cases.groupby(cases).tail(1).index
    case_order = cases.loc[last_idx].drop_duplicates()

    X_last = X.loc[last_idx]
    y_last = y.loc[last_idx]

    # Ensure stable case ordering
    X_last = X_last.loc[case_order.index]
    y_last = y_last.loc[case_order.index]

    return X_last, y_last, X_last.index, state


def transform_aggregation(
    X: pd.DataFrame,
    y: pd.Series,
    cases: pd.Series,
    params: Dict[str, Any],
    state: Optional[TransformerState] = None
) -> Tuple[pd.DataFrame, pd.Series, TransformerState]:
    """
    Aggregates features over the entire case duration (min, max, mean, etc.).
    """
    aggregations = params.get(
        "aggregations",
        {"numeric": ["min", "max", "mean", "std", "sum"], "onehot": ["count"]}
    )

    # 1. Manage State
    if state is None:
        # Determine which columns are what type
        numeric_feats = [c for c in X.columns if c.startswith("numeric_")]
        onehot_feats = [c for c in X.columns if c.startswith("onehot_")]
        
        # Precompute output column names for empty handling
        output_cols = []
        for f in numeric_feats:
            for agg in aggregations.get("numeric", []):
                output_cols.append(f"{f}__{agg}")
        for f in onehot_feats:
            for agg in aggregations.get("onehot", []):
                output_cols.append(f"{f}__{agg}")

        state = TransformerState("aggregation", {
            "numeric_features": numeric_feats,
            "onehot_features": onehot_feats,
            "output_columns": output_cols
        })
    
    numeric_features = state.schema["numeric_features"]
    onehot_features = state.schema["onehot_features"]
    output_columns = state.schema["output_columns"]

    # 2. Empty Input Handling
    if X.shape[0] == 0:
        return pd.DataFrame(columns=output_columns), pd.Series(dtype=float), state

    # 3. Validation
    _validate_inputs(X, y, cases)

    # 4. Transform
    agg_frames = []

    if numeric_features:
        agg_num = X[numeric_features].groupby(cases).agg(aggregations.get("numeric", []))
        agg_num.columns = [f"{col}__{func}" for col, func in agg_num.columns]
        agg_frames.append(agg_num)

    if onehot_features:
        agg_oh = X[onehot_features].groupby(cases).agg(aggregations.get("onehot", []))
        agg_oh.columns = [f"{col}__{func}" for col, func in agg_oh.columns]
        agg_frames.append(agg_oh)

    if agg_frames:
        X_agg = pd.concat(agg_frames, axis=1)
    else:
        X_agg = pd.DataFrame(index=cases.unique())

    # y: Last value per case
    last_idx = cases.groupby(cases).tail(1).index
    y_last = y.loc[last_idx].groupby(cases.loc[last_idx]).first().reindex(X_agg.index)

    return X_agg, y_last, X_agg.index, state


def transform_index_based(
    X: pd.DataFrame,
    y: pd.Series,
    cases: pd.Series,
    params: Dict[str, Any],
    state: Optional[TransformerState] = None
) -> Tuple[pd.DataFrame, pd.Series, TransformerState]:
    """
    Flattens the last k events of a case into a single row.
    """
    k = params.get("k", 3)
    pad_value = params.get("pad_value", 0.0)

    # 1. Manage State
    if state is None:
        if X.columns.empty:
            raise ValueError("IndexBasedTransformer requires column information.")
        
        feature_cols = [c for c in X.columns if c.startswith("numeric_") or c.startswith("onehot_")]
        
        if not feature_cols:
             # Fallback to all columns if strict naming isn't found
            feature_cols = list(X.columns)

        # Precompute output column names
        output_cols = []
        for t in range(k):
            suffix = f"t-{k - t}"
            for f in feature_cols:
                output_cols.append(f"{f}__{suffix}")

        state = TransformerState("index_based", {
            "feature_cols": feature_cols,
            "output_columns": output_cols
        })

    feature_cols = state.schema["feature_cols"]
    output_columns = state.schema["output_columns"]

    # 2. Empty Input Handling
    if X.shape[0] == 0:
        return pd.DataFrame(columns=output_columns), pd.Series(dtype=float), state

    # 3. Validation
    _validate_inputs(X, y, cases)

    # 4. Transform
    X_work = X[feature_cols].copy()
    rows = []
    case_order = []

    for case_id, df_case in X_work.groupby(cases):
        last_k = df_case.tail(k).to_numpy(dtype=float)

        # Pad if shorter than k
        if last_k.shape[0] < k:
            pad = np.full(
                (k - last_k.shape[0], last_k.shape[1]),
                pad_value,
                dtype=float
            )
            last_k = np.vstack([pad, last_k])

        rows.append(last_k.flatten())
        case_order.append(case_id)

    X_out = pd.DataFrame(rows, columns=output_columns, index=case_order)

    # y: Last value per case
    last_idx = cases.groupby(cases).tail(1).index
    y_last = (
        y.loc[last_idx]
        .groupby(cases.loc[last_idx])
        .first()
        .reindex(case_order)
    )

    return X_out, y_last, X_out.index, state


# -----------------------------
# Utility
# -----------------------------

def _validate_inputs(X: pd.DataFrame, y: pd.Series, cases: pd.Series):
    if not (X.index.equals(y.index) and X.index.equals(cases.index)):
        raise ValueError("X, y, and cases must have identical indices.")
    if cases.isna().any():
        raise ValueError("cases contains NaNs.")