from __future__ import annotations

import numpy as np
import pandas as pd

from .common import minmax_scale

RURAL_RISK_WEIGHT_KEYS = [
    "rural_weight",
    "gp_pt_weight",
    "gp_car_weight",
    "no_car_weight",
    "imd_weight",
    "oac_weight",
    "household_weight",
]

# Supergroup fallback mapping (0 to 1)
LSOAC_SUPERGROUP_RISK_MAP: dict[str, float] = {
    "8": 0.90,  # Legacy / Hard-Pressed Communities
    "7": 0.75,  # Semi- and Un-Skilled Workforce / Blue-Collar
    "1": 0.70,  # Retired Professionals / Ageing Rural
    "4": 0.60,  # Low-Skilled Migrant and Student Communities
    "6": 0.50,  # Baseline UK
    "2": 0.40,  # Suburbanites and Peri-Urbanites
    "5": 0.30,  # Ethnically Diverse Suburban Professionals
    "3": 0.20,  # Multicultural and Educated Urbanites
}

# Granular Subgroup mapping across all 52 national UK LSOAC 2021/2 classifications (0 to 1)
LSOAC_SUBGROUP_RISK_MAP: dict[str, float] = {
    # Supergroup 8 (Legacy Communities)
    "8a1": 0.88,  # Retirement Residences
    "8a2": 0.92,  # Flats and Routine Occupations
    "8b1": 0.95,  # Challenged Families (highest vulnerability)
    "8b2": 0.88,  # Retirement Pockets
    "8b3": 0.90,  # Young Family Flat Renters
    # Supergroup 7 (Semi- and Un-Skilled Workforce)
    "7a1": 0.75,  # Ageing Established Urban Communities
    "7a2": 0.74,  # Industry Associations
    "7b1": 0.78,  # Terraces in Transitional Towns
    "7b2": 0.75,  # Families and Later Life
    # Supergroup 1 (Retired Professionals / Ageing Rural)
    "1a1": 0.68,  # Pre-Retirement Spacious Living (spacious countryside)
    "1a2": 0.76,  # Retirement Spacious Living (deep rural + older demographic)
    "1b1": 0.58,  # Younger Established Suburban Communities (market town fringes)
    "1b2": 0.65,  # Older Established Suburban Communities (market town fringes)
    "1c1": 0.50,  # Affluent Mature Families
    "1c2": 0.52,  # Burgeoning Mature Families
    # Supergroup 4 (Low-Skilled Migrant and Student Communities)
    "4a1": 0.62,  # Semi-Detached, Service Workers and Students
    "4a2": 0.60,  # City Service Workers
    "4a3": 0.64,  # Multi-Child Young Families
    "4b1": 0.65,  # Migrant Families
    "4b2": 0.58,  # European Skilled Workforce
    "4b3": 0.60,  # Inner Suburb Ethnic Group Mix
    "4b4": 0.65,  # Ethnic Minority Routine Service Workers
    "4c1": 0.60,  # African and Asian Influences
    "4c2": 0.58,  # European and Asian Heritage
    # Supergroup 6 (Baseline UK)
    "6a1": 0.45,  # Suburban Housing Starters
    "6a2": 0.48,  # Semi Detached Strivers
    "6a3": 0.55,  # Younger Ethnic Minority Families in Flats
    "6b1": 0.68,  # Retired Seniors (Coastal & Legacy Industrial)
    "6b2": 0.62,  # Traditional Terraces (Coastal & Legacy Industrial)
    "6b3": 0.58,  # EU Singles (Agri-Food and Industry)
    "6c1": 0.56,  # Transient Communities
    "6c2": 0.52,  # Semi-Detached Family Renters
    # Supergroup 2 (Suburbanites and Peri-Urbanites)
    "2a1": 0.45,  # Younger Suburban Family Renters
    "2a2": 0.35,  # Settled Owner-Occupied Suburbs
    "2a3": 0.42,  # Terraced Communities
    "2b1": 0.52,  # Ageing Rural Communities (rural villages)
    "2b2": 0.42,  # Rural Mix
    "2c1": 0.65,  # Communal Retirement Living (care / retirement homes)
    "2c2": 0.48,  # Ageing Independent Living
    # Supergroup 5 (Ethnically Diverse Suburban Professionals)
    "5a1": 0.32,  # Outer Suburb Asian Mix
    "5a2": 0.30,  # Suburban Empty Nesters
    "5a3": 0.32,  # Young Suburban Families
    "5b1": 0.36,  # Families in Multi-Ethnic Terraces
    "5b2": 0.28,  # Established Multi-Ethnic Suburbs
    # Supergroup 3 (Multicultural and Educated Urbanites)
    "3a1": 0.25,  # University Centric
    "3a2": 0.20,  # Professional Progression
    "3a3": 0.22,  # Urbanite Mix
    "3a4": 0.18,  # Affluent Graduate Living
    "3b1": 0.30,  # Private Rental Ethnic Minority Families
    "3b2": 0.28,  # Young Ethnic Minority Families
    "3c1": 0.18,  # Centrally Located Professionals
    "3c2": 0.16,  # Career Progression
}


def normalize_rural_risk_weights(weight_values: dict[str, float]) -> dict[str, float]:
    values = np.array([float(weight_values[k]) for k in RURAL_RISK_WEIGHT_KEYS], dtype=float)
    values = np.clip(values, a_min=0.0, a_max=None)
    total = float(values.sum())
    if np.isclose(total, 0.0):
        equal = 1.0 / len(RURAL_RISK_WEIGHT_KEYS)
        return {k: equal for k in RURAL_RISK_WEIGHT_KEYS}
    values = values / total
    return {k: float(v) for k, v in zip(RURAL_RISK_WEIGHT_KEYS, values)}


def apply_rural_risk_index(
    lsoa_df: pd.DataFrame,
    rural_weight: float = 14.3,
    gp_pt_weight: float = 14.3,
    gp_car_weight: float = 14.3,
    no_car_weight: float = 14.3,
    imd_weight: float = 14.3,
    oac_weight: float = 14.3,
    household_weight: float = 14.2,
) -> pd.DataFrame:
    """Computes the multi-dimensional Rural Risk Index across 7 indicators (0-1, higher = higher risk)."""
    normalized_weights = normalize_rural_risk_weights(
        {
            "rural_weight": rural_weight,
            "gp_pt_weight": gp_pt_weight,
            "gp_car_weight": gp_car_weight,
            "no_car_weight": no_car_weight,
            "imd_weight": imd_weight,
            "oac_weight": oac_weight,
            "household_weight": household_weight,
        }
    )

    out = lsoa_df.copy()

    # 1. Rural / Urban Isolation Risk (Invert Rural_Access: Smaller/remote rural = 1.0, Urban = 0.0)
    rural_access = pd.to_numeric(out.get("Rural_Access", 0.5), errors="coerce").fillna(0.5)
    out["Rural_Isolation_Normalized"] = 1.0 - rural_access

    # 2. GP Travel Time (PT / Walk) (Higher travel time = higher barrier/risk)
    gp_pt_time = pd.to_numeric(out.get("GP_PT_Time", np.nan), errors="coerce")
    out["GP_PT_Travel_Time_Normalized"] = minmax_scale(gp_pt_time).fillna(0.5)

    # 3. GP Travel Time (Car) (Higher travel time = higher barrier/risk)
    gp_car_time = pd.to_numeric(out.get("GP_Car_Time", np.nan), errors="coerce")
    out["GP_Car_Travel_Time_Normalized"] = minmax_scale(gp_car_time).fillna(0.5)

    # 4. Car Non-Ownership / Transport Vulnerability (Higher % no-car = higher risk)
    no_cars_pct = pd.to_numeric(out.get("No_Cars_Pct", np.nan), errors="coerce")
    out["No_Car_Normalized"] = minmax_scale(no_cars_pct).fillna(0.5)

    # 5. Deprivation (IMD 2025 Rank: Rank 1 is most deprived in England out of 33,755)
    imd_rank = pd.to_numeric(out.get("IMD_2025_Rank", np.nan), errors="coerce")
    max_nat_rank = 33755.0
    out["IMD_Deprivation_Normalized"] = (1.0 - (imd_rank / max_nat_rank)).clip(lower=0.0, upper=1.0).fillna(0.5)

    # 6. Geodemographic Vulnerability (LSOAC 2021/2 Subgroup mapping with Supergroup fallback)
    subgroup_series = out.get("Subgroup_Code", pd.Series("", index=out.index)).astype(str).str.strip()
    supergroup_series = out.get("Supergroup_Code", pd.Series("", index=out.index)).astype(str).str.strip()

    mapped_subgroup = subgroup_series.map(LSOAC_SUBGROUP_RISK_MAP)
    mapped_supergroup = supergroup_series.map(LSOAC_SUPERGROUP_RISK_MAP)
    raw_oac_scores = mapped_subgroup.combine_first(mapped_supergroup).fillna(0.5)

    out["LSOAC_Risk_Normalized"] = raw_oac_scores

    # 7. Household Composition Vulnerability (Solitary Elderly 66+, Lone Parents, and Elderly Couples)
    hh_vuln = pd.to_numeric(out.get("Household_Vulnerability_Score", np.nan), errors="coerce")
    out["Household_Vulnerability_Normalized"] = minmax_scale(hh_vuln).fillna(0.5)

    # Composite Rural Risk Index
    out["Rural_Risk_Index"] = (
        out["Rural_Isolation_Normalized"] * normalized_weights["rural_weight"]
        + out["GP_PT_Travel_Time_Normalized"] * normalized_weights["gp_pt_weight"]
        + out["GP_Car_Travel_Time_Normalized"] * normalized_weights["gp_car_weight"]
        + out["No_Car_Normalized"] * normalized_weights["no_car_weight"]
        + out["IMD_Deprivation_Normalized"] * normalized_weights["imd_weight"]
        + out["LSOAC_Risk_Normalized"] * normalized_weights["oac_weight"]
        + out["Household_Vulnerability_Normalized"] * normalized_weights["household_weight"]
    )

    return out
