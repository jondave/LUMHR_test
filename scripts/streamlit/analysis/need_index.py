import numpy as np
import pandas as pd
import streamlit as st

from analysis.common import minmax_scale


WEIGHT_KEYS = ["dep_weight", "smi_weight", "prescribing_weight", "samhi_weight"]


def normalize_weights(weight_values: dict[str, float]) -> dict[str, float]:
    values = np.array([float(weight_values[k]) for k in WEIGHT_KEYS], dtype=float)
    values = np.clip(values, a_min=0.0, a_max=None)
    total = float(values.sum())
    if np.isclose(total, 0.0):
        equal = 1.0 / len(WEIGHT_KEYS)
        return {k: equal for k in WEIGHT_KEYS}
    values = values / total
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, values)}


def rebalance_weight_values(
    weight_values: dict[str, float],
    changed_key: str,
    keys: list[str] | None = None,
    total_points: float = 100.0,
) -> dict[str, float]:
    """Return a balanced copy of the weight values without touching session state.

    This keeps the changed slider value fixed and redistributes the remaining
    points across the other controls, matching the interactive behavior used by
    the original page.
    """

    keys = keys or WEIGHT_KEYS
    if changed_key not in keys:
        return {k: float(weight_values.get(k, 0.0)) for k in keys}

    balanced = {k: max(float(weight_values.get(k, 0.0)), 0.0) for k in keys}
    changed_val = min(max(float(balanced[changed_key]), 0.0), total_points)
    balanced[changed_key] = changed_val

    other_keys = [k for k in keys if k != changed_key]
    other_vals = [float(balanced[k]) for k in other_keys]
    other_sum = sum(other_vals)
    remaining = max(0.0, total_points - changed_val)

    if np.isclose(other_sum, 0.0):
        for k in other_keys:
            balanced[k] = remaining / len(other_keys)
    else:
        scale = remaining / other_sum
        for k in other_keys:
            balanced[k] = float(balanced[k]) * scale

    for k in keys:
        balanced[k] = round(float(balanced[k]), 1)

    total = sum(float(balanced[k]) for k in keys)
    diff = round(total_points - total, 1)
    balanced[other_keys[-1]] = round(float(balanced[other_keys[-1]]) + diff, 1)
    return balanced


def rebalance_weight_points(changed_key: str, keys: list[str] | None = None, total_points: float = 100.0) -> None:
    keys = keys or WEIGHT_KEYS
    if changed_key not in keys:
        return

    if st.session_state.get("_weight_rebalancing", False):
        return

    st.session_state["_weight_rebalancing"] = True
    try:
        balanced = rebalance_weight_values(
            {k: float(st.session_state.get(k, 0.0)) for k in keys},
            changed_key=changed_key,
            keys=keys,
            total_points=total_points,
        )
        for k, value in balanced.items():
            st.session_state[k] = value
    finally:
        st.session_state["_weight_rebalancing"] = False


def apply_need_score(
    lsoa_df: pd.DataFrame,
    dep_weight: float,
    smi_weight: float,
    prescribing_weight: float,
    samhi_weight: float,
    samhi_column: str,
) -> pd.DataFrame:
    normalized_weights = normalize_weights(
        {
            "dep_weight": dep_weight,
            "smi_weight": smi_weight,
            "prescribing_weight": prescribing_weight,
            "samhi_weight": samhi_weight,
        }
    )

    out = lsoa_df.copy()
    out["SAMHI_Selected"] = pd.to_numeric(out.get(samhi_column), errors="coerce")
    out["Depression_Normalized"] = minmax_scale(out["Depression_Prevalence"])
    out["SMI_Normalized"] = minmax_scale(out["SMI_Prevalence"])
    out["Prescribing_Normalized"] = minmax_scale(out["Antidepressant_Items_Per_Patient"])
    out["SAMHI_Normalized"] = minmax_scale(out["SAMHI_Selected"])
    out["Need_Score"] = (
        out["Depression_Normalized"].fillna(0) * normalized_weights["dep_weight"]
        + out["SMI_Normalized"].fillna(0) * normalized_weights["smi_weight"]
        + out["Prescribing_Normalized"].fillna(0) * normalized_weights["prescribing_weight"]
        + out["SAMHI_Normalized"].fillna(0) * normalized_weights["samhi_weight"]
    )
    return out
