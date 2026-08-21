from __future__ import annotations

import pandas as pd

from .common import find_column, normalize_code, parse_numeric


def build_2011_to_2021_lookup(lookup_df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each 2011 LSOA code to the 2021 LSOA code(s) it corresponds to (handles splits/merges)."""
    code_2011_col = find_column(lookup_df, ["LSOA11CD"])
    code_2021_col = find_column(lookup_df, ["LSOA21CD"])

    mapping: dict[str, list[str]] = {}
    for code_2011, code_2021 in zip(
        lookup_df[code_2011_col].map(normalize_code), lookup_df[code_2021_col].map(normalize_code)
    ):
        if not code_2011 or not code_2021:
            continue
        mapping.setdefault(code_2011, []).append(code_2021)
    return mapping


def _remap_2011_df_to_2021(df: pd.DataFrame, lookup_map: dict[str, list[str]], value_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        code_2011 = row["LSOA_CODE"]
        targets = lookup_map.get(code_2011, [code_2011])
        for target in targets:
            new_row = {"LSOA_CODE": target}
            for col in value_cols:
                new_row[col] = row.get(col)
            rows.append(new_row)

    remapped = pd.DataFrame(rows)
    # Splits duplicate the 2011 value across new 2021 areas; merges are averaged across their 2011 sources.
    return remapped.groupby("LSOA_CODE", as_index=False)[value_cols].mean()


def prepare_travel_times(
    gp_raw: pd.DataFrame,
    hospital_raw: pd.DataFrame,
    lookup_map: dict[str, list[str]],
) -> pd.DataFrame:
    gp_code_col = find_column(gp_raw, ["LSOA_code", "LSOA11CD"])
    gp_pt_col = find_column(gp_raw, ["GPPTt"])
    gp_car_col = find_column(gp_raw, ["GPCart"])

    gp_df = pd.DataFrame(
        {
            "LSOA_CODE": gp_raw[gp_code_col].map(normalize_code),
            "GP_PT_Time": parse_numeric(gp_raw[gp_pt_col]),
            "GP_Car_Time": parse_numeric(gp_raw[gp_car_col]),
        }
    )
    gp_df = gp_df[gp_df["LSOA_CODE"].ne("")].copy()
    gp_2021 = _remap_2011_df_to_2021(gp_df, lookup_map, ["GP_PT_Time", "GP_Car_Time"])

    hosp_code_col = find_column(hospital_raw, ["LSOA_code", "LSOA11CD"])
    hosp_pt_col = find_column(hospital_raw, ["HospPTt"])
    hosp_car_col = find_column(hospital_raw, ["HospCart"])

    hosp_df = pd.DataFrame(
        {
            "LSOA_CODE": hospital_raw[hosp_code_col].map(normalize_code),
            "Hosp_PT_Time": parse_numeric(hospital_raw[hosp_pt_col]),
            "Hosp_Car_Time": parse_numeric(hospital_raw[hosp_car_col]),
        }
    )
    hosp_df = hosp_df[hosp_df["LSOA_CODE"].ne("")].copy()
    hosp_2021 = _remap_2011_df_to_2021(hosp_df, lookup_map, ["Hosp_PT_Time", "Hosp_Car_Time"])

    return gp_2021.merge(hosp_2021, on="LSOA_CODE", how="outer")


# Ordered from most-connected (0) to most-isolated (5); higher = more isolated.
ISOLATION_SCALE = {
    "Urban: Nearer to a major town or city": 0,
    "Urban: Further from a major town or city": 1,
    "Larger rural: Nearer to a major town or city": 2,
    "Larger rural: Further from a major town or city": 3,
    "Smaller rural: Nearer to a major town or city": 4,
    "Smaller rural: Further from a major town or city": 5,
}


def prepare_rural_urban(df: pd.DataFrame) -> pd.DataFrame:
    code_col = find_column(df, ["LSOA21CD", "LSOA_CODE"])
    name_col = find_column(df, ["RUC21NM"])

    out = pd.DataFrame(
        {
            "LSOA_CODE": df[code_col].map(normalize_code),
            "RUC21NM": df[name_col].astype(str).str.strip(),
        }
    )
    out = out[out["LSOA_CODE"].ne("")].copy()
    out["Isolation_Scale"] = out["RUC21NM"].map(ISOLATION_SCALE)

    min_scale = min(ISOLATION_SCALE.values())
    max_scale = max(ISOLATION_SCALE.values())
    out["Isolation_Normalized"] = (out["Isolation_Scale"] - min_scale) / (max_scale - min_scale)
    out["Rural_Access"] = 1 - out["Isolation_Normalized"]
    return out
