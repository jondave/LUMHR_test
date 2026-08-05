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


def rebalance_weight_points(changed_key: str, keys: list[str] | None = None, total_points: float = 100.0) -> None:
    keys = keys or WEIGHT_KEYS
    if changed_key not in keys:
        return

    if st.session_state.get("_weight_rebalancing", False):
        return

    st.session_state["_weight_rebalancing"] = True
    try:
        changed_val = float(st.session_state.get(changed_key, 0.0))
        changed_val = min(max(changed_val, 0.0), total_points)
        st.session_state[changed_key] = changed_val

        other_keys = [k for k in keys if k != changed_key]
        other_vals = [float(st.session_state.get(k, 0.0)) for k in other_keys]
        other_sum = sum(other_vals)
        remaining = max(0.0, total_points - changed_val)

        if np.isclose(other_sum, 0.0):
            for k in other_keys:
                st.session_state[k] = remaining / len(other_keys)
        else:
            scale = remaining / other_sum
            for k in other_keys:
                st.session_state[k] = float(st.session_state.get(k, 0.0)) * scale

        for k in keys:
            st.session_state[k] = round(float(st.session_state[k]), 1)
        total = sum(float(st.session_state[k]) for k in keys)
        diff = round(total_points - total, 1)
        st.session_state[other_keys[-1]] = round(float(st.session_state[other_keys[-1]]) + diff, 1)
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
