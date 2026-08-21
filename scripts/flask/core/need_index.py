from __future__ import annotations

import numpy as np
import pandas as pd

from .common import minmax_scale


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


def apply_need_index(
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
    out["Need_Index"] = (
        out["Depression_Normalized"].fillna(0) * normalized_weights["dep_weight"]
        + out["SMI_Normalized"].fillna(0) * normalized_weights["smi_weight"]
        + out["Prescribing_Normalized"].fillna(0) * normalized_weights["prescribing_weight"]
        + out["SAMHI_Normalized"].fillna(0) * normalized_weights["samhi_weight"]
    )
    return out
