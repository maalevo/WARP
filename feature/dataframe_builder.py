from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
import pandas as pd

# Make sure to import NormState if you want to use it for type hinting, 
# or just use Any/dict if circular imports are an issue.
from .feature_builder import FeatureSpec, build_feature, NormState

def build_feature_dataframe(
    prefix_log: pd.DataFrame,
    feature_specs: List[FeatureSpec],
    *,
    fit_states: Optional[Dict[str, List[NormState]]] = None, # NEW: Input states (for Test)
    validate_unique_feature_keys: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[NormState]]]: # NEW: Returns Tuple
    """
    Build a complete feature DataFrame from a list of FeatureSpec.
    
    Now supports stateful normalization.
    
    Returns:
      (X, states)
      - X: The concatenated feature DataFrame.
      - states: A dictionary mapping column names to normalization parameters. 
                Save this object to normalize test data later.
    """
    if not isinstance(prefix_log, pd.DataFrame):
        raise TypeError("prefix_log must be a pandas DataFrame.")
    
    if not feature_specs:
        # Return empty DF and empty states
        X = pd.DataFrame(index=prefix_log.index)
        return X, {}

    if validate_unique_feature_keys:
        keys = [fs.feature_key for fs in feature_specs]
        dup = _find_duplicates(keys)
        if dup:
            raise ValueError(
                f"Duplicate feature_key(s) found: {sorted(dup)}. "
                "Set validate_unique_feature_keys=False to allow duplicates."
            )

    feature_dfs: List[pd.DataFrame] = []
    collected_states: Dict[str, List[NormState]] = {}

    for i, fs in enumerate(feature_specs):
        if not isinstance(fs, FeatureSpec):
            raise TypeError(f"feature_specs[{i}] is not a FeatureSpec: {type(fs)}")

        # CALL UPDATED API
        # We pass the full fit_states dictionary; build_feature picks out the specific columns it needs.
        df_i, states_i = build_feature(prefix_log, fs, fit_states=fit_states)

        if not isinstance(df_i, pd.DataFrame):
            raise TypeError(
                f"build_feature returned non-DataFrame for spec {fs.feature_key}: {type(df_i)}"
            )

        # Ensure correct alignment
        if not df_i.index.equals(prefix_log.index):
            df_i = df_i.reindex(prefix_log.index)

        feature_dfs.append(df_i)
        
        # Merge the new states into our master dictionary
        # Since column names must be unique in the final DF, keys here should not overlap 
        # (or will be caught by the duplicate check below).
        collected_states.update(states_i)

    X = pd.concat(feature_dfs, axis=1)

    # Guard against duplicate columns
    dup_cols = X.columns[X.columns.duplicated()].tolist()
    if dup_cols:
        raise ValueError(
            "Duplicate feature column names detected after concatenation. "
            "This typically happens when two FeatureSpecs produce overlapping column names.\n"
            f"Duplicates: {sorted(set(dup_cols))}\n"
            "Fix by using distinct 'prefix' / 'feature_name' values in params."
        )

    # If we are in "Test Mode" (fit_states was provided), we return that same object 
    # (or the updated one, though states shouldn't change in test mode).
    # If "Train Mode", we return the newly collected states.
    final_states = fit_states if fit_states is not None else collected_states

    return X, final_states


def _find_duplicates(items: List[str]) -> set:
    seen = set()
    dup = set()
    for x in items:
        if x in seen:
            dup.add(x)
        else:
            seen.add(x)
    return dup