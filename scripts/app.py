from pathlib import Path
import html
import json

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import linear
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static


def is_running_via_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def normalize_key(text: str) -> str:
    if text is None:
        return ""
    return "".join(ch for ch in str(text).strip().lower() if ch.isalnum())


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def optional_numeric(df: pd.DataFrame, column_name: str | None) -> pd.Series:
    if column_name:
        return parse_numeric(df[column_name])
    return pd.Series(np.nan, index=df.index)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def minmax_scale(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    min_v = valid.min()
    max_v = valid.max()
    if np.isclose(max_v, min_v):
        return pd.Series(0.5, index=series.index)
    return (series - min_v) / (max_v - min_v)


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    columns = list(df.columns)
    normalized_map = {normalize_key(c): c for c in columns}

    for cand in candidates:
        if cand in columns:
            return cand
        nk = normalize_key(cand)
        if nk in normalized_map:
            return normalized_map[nk]

    for cand in candidates:
        nk = normalize_key(cand)
        partial_matches = [c for c in columns if nk in normalize_key(c)]
        if partial_matches:
            return partial_matches[0]

    if required:
        raise ValueError(f"Missing required column. Expected one of: {candidates}")
    return None


def find_prefix_column(df: pd.DataFrame, prefix: str, required: bool = True) -> str | None:
    prefix_norm = normalize_key(prefix)
    for col in df.columns:
        if normalize_key(col).startswith(prefix_norm):
            return col
    if required:
        raise ValueError(f"Missing required column with prefix: {prefix}")
    return None


def get_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "depression": base_dir / "datasets" / "quality_outcomes_framework" / "qof_depression_2425_lincolnshire.csv",
        "smi": base_dir / "datasets" / "quality_outcomes_framework" / "qof_mental_health_2425_lincolnshire.csv",
        "prescribing": base_dir
        / "datasets"
        / "gp_prescribing_data"
        / "items_for_antidepressant_drugs_per_gp_lincolnshire_may_2026.csv",
        "mapping_male": base_dir
        / "datasets"
        / "patients_registered_gp_practice"
        / "july_2026"
        / "gp-reg-pat-prac-lsoa-male.csv",
        "mapping_female": base_dir
        / "datasets"
        / "patients_registered_gp_practice"
        / "july_2026"
        / "gp-reg-pat-prac-lsoa-female.csv",
        "gp_locations": base_dir / "datasets" / "gp_locations" / "Lincolnshire_ICB_GPs.csv",
        "lsoa_geo": base_dir / "datasets" / "lincolnshire_lsoa" / "lower-super-output-areas-2021-5RrVTw.geojson",
    }


def resolve_base_dir(script_file: Path) -> Path:
    script_dir = script_file.resolve().parent
    candidates = [script_dir, script_dir.parent]
    for candidate in candidates:
        if (candidate / "datasets").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the datasets directory. Expected it under either the script directory "
        "or its parent directory."
    )


def load_raw_data(
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    paths = get_paths(base_dir)

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found for {name}: {path}")

    dep_df = pd.read_csv(paths["depression"])
    smi_df = pd.read_csv(paths["smi"])
    prescribing_df = pd.read_csv(paths["prescribing"])
    map_male_df = pd.read_csv(paths["mapping_male"])
    map_female_df = pd.read_csv(paths["mapping_female"])
    gp_loc_df = pd.read_csv(paths["gp_locations"])
    # Use a JSON loader path first to avoid GDAL/OGR runtime instability on some systems.
    with open(paths["lsoa_geo"], "r", encoding="utf-8") as f:
        gj = json.load(f)
    lsoa_gdf = gpd.GeoDataFrame.from_features(gj["features"]) 
    if lsoa_gdf.crs is None:
        lsoa_gdf = lsoa_gdf.set_crs("EPSG:4326")

    return dep_df, smi_df, prescribing_df, map_male_df, map_female_df, gp_loc_df, lsoa_gdf


def prepare_depression(df: pd.DataFrame) -> pd.DataFrame:
    practice_col = find_column(df, ["Practice Code", "PRACTICE_CODE", "Practice_Code"])
    name_col = find_column(df, ["Practice Name", "PRACTICE_NAME", "Practice_Name"], required=False)
    reg_col = find_column(df, ["Register 2425", "Register"])
    list_col = find_column(df, ["List size aged 18+ 2425", "List Size", "List size"])
    prev_col = find_column(df, ["Prevalence (%) 2425", "Prevalence"])
    pca_col = find_column(df, ["Overall PCA Rate (%) 2425", "Overall PCA Rate (%)"], required=False)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "Practice_Name_Dep": df[name_col].map(clean_text) if name_col else "",
            "Dep_Register": parse_numeric(df[reg_col]),
            "Dep_List_Size": parse_numeric(df[list_col]),
            "Dep_Prevalence_Pct": parse_numeric(df[prev_col]),
            "Dep_Exception_Rate_Pct": optional_numeric(df, pca_col),
        }
    )

    out = out[out["PRACTICE_CODE"].ne("")].copy()

    calc_prev = safe_divide(out["Dep_Register"], out["Dep_List_Size"]) * 100
    out["Dep_Prevalence_Pct"] = out["Dep_Prevalence_Pct"].fillna(calc_prev)
    return out


def prepare_smi(df: pd.DataFrame) -> pd.DataFrame:
    practice_col = find_column(df, ["Practice Code", "PRACTICE_CODE", "Practice_Code"])
    name_col = find_column(df, ["Practice Name", "PRACTICE_NAME", "Practice_Name"], required=False)
    reg_col = find_column(df, ["Register 2425", "Register"])
    list_col = find_column(df, ["List size aged 18+ 2425", "List Size", "List size"])
    prev_col = find_column(df, ["Prevalence (%) 2425", "Prevalence"])

    mh002_col = find_prefix_column(df, "MH002", required=False)
    mh003_col = find_prefix_column(df, "MH003", required=False)
    mh006_col = find_prefix_column(df, "MH006", required=False)
    mh007_col = find_prefix_column(df, "MH007", required=False)
    mh011_col = find_prefix_column(df, "MH011", required=False)
    mh012_col = find_prefix_column(df, "MH012", required=False)
    pca_col = find_column(df, ["Overall PCA Rate (%) 2425", "Overall PCA Rate (%)"], required=False)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "Practice_Name_SMI": df[name_col].map(clean_text) if name_col else "",
            "SMI_Register": parse_numeric(df[reg_col]),
            "SMI_List_Size": parse_numeric(df[list_col]),
            "SMI_Prevalence_Pct": parse_numeric(df[prev_col]),
            "MH002_Pct": optional_numeric(df, mh002_col),
            "MH003_Pct": optional_numeric(df, mh003_col),
            "MH006_Pct": optional_numeric(df, mh006_col),
            "MH007_Pct": optional_numeric(df, mh007_col),
            "MH011_Pct": optional_numeric(df, mh011_col),
            "MH012_Pct": optional_numeric(df, mh012_col),
            "SMI_Exception_Rate_Pct": optional_numeric(df, pca_col),
        }
    )

    out = out[out["PRACTICE_CODE"].ne("")].copy()

    calc_prev = safe_divide(out["SMI_Register"], out["SMI_List_Size"]) * 100
    out["SMI_Prevalence_Pct"] = out["SMI_Prevalence_Pct"].fillna(calc_prev)
    out["Physical_Health_Review_Avg_Pct"] = out[
        ["MH003_Pct", "MH006_Pct", "MH007_Pct", "MH011_Pct", "MH012_Pct"]
    ].mean(axis=1, skipna=True)
    return out


def prepare_prescribing(df: pd.DataFrame) -> pd.DataFrame:
    practice_col = find_column(df, ["id", "PRACTICE_CODE", "Practice Code"])
    items_col = find_column(df, ["items", "ITEMS", "Number of Items"])
    name_col = find_column(df, ["name", "Practice Name", "PRACTICE_NAME"], required=False)
    cost_col = find_column(df, ["actual_cost", "ACTUAL_COST", "cost"], required=False)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "Practice_Name_Rx": df[name_col].map(clean_text) if name_col else "",
            "Antidepressant_Items": parse_numeric(df[items_col]),
            "Antidepressant_Actual_Cost": optional_numeric(df, cost_col),
        }
    )
    out = out[out["PRACTICE_CODE"].ne("")].copy()
    out = out.groupby("PRACTICE_CODE", as_index=False).agg(
        {
            "Practice_Name_Rx": "first",
            "Antidepressant_Items": "sum",
            "Antidepressant_Actual_Cost": "sum",
        }
    )
    return out


def prepare_mapping(df: pd.DataFrame, sex_label: str | None = None) -> pd.DataFrame:
    practice_col = find_column(df, ["PRACTICE_CODE", "Practice Code"])
    lsoa_col = find_column(df, ["LSOA_CODE", "LSOA"])
    patients_col = find_column(df, ["NUMBER_OF_PATIENTS", "Number of Patients"])
    sex_col = find_column(df, ["SEX"], required=False)

    if sex_col:
        sex_series = df[sex_col].astype(str).str.strip().str.upper().replace({"M": "MALE", "F": "FEMALE"})
    elif sex_label:
        sex_series = pd.Series(str(sex_label).upper(), index=df.index)
    else:
        sex_series = pd.Series("ALL", index=df.index)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "LSOA_CODE": df[lsoa_col].map(normalize_code),
            "NUMBER_OF_PATIENTS": parse_numeric(df[patients_col]),
            "SEX": sex_series,
        }
    )

    out = out[out["NUMBER_OF_PATIENTS"].notna()]
    if sex_col and sex_label is None:
        out = out[out["SEX"] == "ALL"]

    out = out[(out["PRACTICE_CODE"].ne("")) & (out["LSOA_CODE"].ne(""))]
    out = out[out["NUMBER_OF_PATIENTS"] > 0]
    return out


def prepare_mapping_from_sex_split(male_df: pd.DataFrame, female_df: pd.DataFrame) -> pd.DataFrame:
    male = prepare_mapping(male_df, sex_label="MALE")
    female = prepare_mapping(female_df, sex_label="FEMALE")

    combined = pd.concat([male, female], ignore_index=True)
    combined = combined[combined["SEX"].isin(["MALE", "FEMALE"])].copy()

    split = (
        combined.pivot_table(
            index=["PRACTICE_CODE", "LSOA_CODE"],
            columns="SEX",
            values="NUMBER_OF_PATIENTS",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .copy()
    )

    if "MALE" not in split.columns:
        split["MALE"] = 0
    if "FEMALE" not in split.columns:
        split["FEMALE"] = 0

    split = split.rename(
        columns={
            "MALE": "NUMBER_OF_PATIENTS_MALE",
            "FEMALE": "NUMBER_OF_PATIENTS_FEMALE",
        }
    )
    split["NUMBER_OF_PATIENTS"] = split["NUMBER_OF_PATIENTS_MALE"] + split["NUMBER_OF_PATIENTS_FEMALE"]
    split = split[split["NUMBER_OF_PATIENTS"] > 0].copy()

    gp_totals = split.groupby("PRACTICE_CODE", as_index=False)["NUMBER_OF_PATIENTS"].sum()
    gp_totals = gp_totals.rename(columns={"NUMBER_OF_PATIENTS": "GP_Total_Mapped_Patients"})
    split = split.merge(gp_totals, on="PRACTICE_CODE", how="left")
    split["Weight"] = safe_divide(split["NUMBER_OF_PATIENTS"], split["GP_Total_Mapped_Patients"])
    return split


def prepare_lsoa_geography(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    code_col = find_column(gdf, ["LSOA_CODE", "CODE", "lsoa_code", "code"])
    out = gdf.copy()
    out["LSOA_CODE"] = out[code_col].map(normalize_code)
    out = out[["LSOA_CODE", "geometry"]].drop_duplicates(subset=["LSOA_CODE"])  # Guard against accidental duplicates.

    if out.crs is None:
        out = out.set_crs("EPSG:4326")

    return out


def get_lsoa_centroids(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    # Use a projected CRS for stable centroid computation.
    projected = gdf.to_crs(epsg=27700)
    cent = projected.copy()
    cent["geometry"] = cent.geometry.centroid
    cent = cent.to_crs(epsg=4326)
    cent["lon"] = cent.geometry.x
    cent["lat"] = cent.geometry.y
    return cent[["LSOA_CODE", "lat", "lon"]]


def prepare_gp_locations(df: pd.DataFrame) -> pd.DataFrame:
    code_col = find_column(df, ["Code", "PRACTICE_CODE", "Practice Code"])
    name_col = find_column(df, ["Name", "Practice Name", "PRACTICE_NAME"], required=False)
    lsoa_col = find_column(df, ["LSOA", "LSOA_CODE"], required=False)

    lat_col = find_column(df, ["Latitude", "LAT", "lat"], required=False)
    lon_col = find_column(df, ["Longitude", "LON", "lng", "Long"], required=False)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[code_col].map(normalize_code),
            "GP_Name_File": df[name_col].map(clean_text) if name_col else "",
            "GP_LSOA": df[lsoa_col].map(normalize_code) if lsoa_col else "",
            "GP_Lat": parse_numeric(df[lat_col]) if lat_col else np.nan,
            "GP_Lon": parse_numeric(df[lon_col]) if lon_col else np.nan,
        }
    )
    out = out[out["PRACTICE_CODE"].ne("")].copy()

    valid_lat = out["GP_Lat"].between(49.0, 61.0)
    valid_lon = out["GP_Lon"].between(-8.5, 2.5)
    out.loc[~(valid_lat & valid_lon), ["GP_Lat", "GP_Lon"]] = np.nan
    return out


def build_gp_master(dep_df: pd.DataFrame, smi_df: pd.DataFrame, prescribing_df: pd.DataFrame) -> pd.DataFrame:
    gp = dep_df.merge(smi_df, on="PRACTICE_CODE", how="outer")
    gp = gp.merge(prescribing_df, on="PRACTICE_CODE", how="outer")

    gp["Practice_Name"] = gp["Practice_Name_Dep"].where(gp["Practice_Name_Dep"].astype(bool), gp["Practice_Name_SMI"])
    gp["Practice_Name"] = gp["Practice_Name"].where(gp["Practice_Name"].astype(bool), gp["Practice_Name_Rx"])
    gp["Practice_Name"] = gp["Practice_Name"].fillna("")

    gp["Exception_Rate_Pct"] = gp["SMI_Exception_Rate_Pct"].where(
        gp["SMI_Exception_Rate_Pct"].notna(), gp["Dep_Exception_Rate_Pct"]
    )
    return gp


def allocate_registers_to_lsoa(mapping_df: pd.DataFrame, gp_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    alloc = mapping_df.merge(
        gp_df[["PRACTICE_CODE", "Dep_Register", "SMI_Register", "Antidepressant_Items"]],
        on="PRACTICE_CODE",
        how="left",
    )

    missing_dep_by_practice = alloc["Dep_Register"].isna().groupby(alloc["PRACTICE_CODE"]).first()
    missing_smi_by_practice = alloc["SMI_Register"].isna().groupby(alloc["PRACTICE_CODE"]).first()
    missing_rx_by_practice = alloc["Antidepressant_Items"].isna().groupby(alloc["PRACTICE_CODE"]).first()
    missing_any_by_practice = (missing_dep_by_practice | missing_smi_by_practice)

    mismatch_summary = {
        "mapped_practices_without_any_qof": int(missing_any_by_practice.sum()),
        "mapped_practices_without_dep_qof": int(missing_dep_by_practice.sum()),
        "mapped_practices_without_smi_qof": int(missing_smi_by_practice.sum()),
        "mapped_practices_without_prescribing": int(missing_rx_by_practice.sum()),
        "allocation_rows_missing_any_qof": int((alloc["Dep_Register"].isna() | alloc["SMI_Register"].isna()).sum()),
        "allocation_rows_missing_prescribing": int(alloc["Antidepressant_Items"].isna().sum()),
    }

    alloc["Allocated_Depression"] = alloc["Dep_Register"].fillna(0) * alloc["Weight"].fillna(0)
    alloc["Allocated_SMI"] = alloc["SMI_Register"].fillna(0) * alloc["Weight"].fillna(0)
    alloc["Allocated_Antidepressant_Items"] = alloc["Antidepressant_Items"].fillna(0) * alloc["Weight"].fillna(0)

    lsoa_dep = alloc.groupby("LSOA_CODE", as_index=False)["Allocated_Depression"].sum()
    lsoa_smi = alloc.groupby("LSOA_CODE", as_index=False)["Allocated_SMI"].sum()
    lsoa_rx = alloc.groupby("LSOA_CODE", as_index=False)["Allocated_Antidepressant_Items"].sum()
    lsoa_list = alloc.groupby("LSOA_CODE", as_index=False)["NUMBER_OF_PATIENTS"].sum()

    lsoa = (
        lsoa_list.merge(lsoa_dep, on="LSOA_CODE", how="left")
        .merge(lsoa_smi, on="LSOA_CODE", how="left")
        .merge(lsoa_rx, on="LSOA_CODE", how="left")
    )
    lsoa = lsoa.rename(columns={"NUMBER_OF_PATIENTS": "LSOA_Total_List"})

    lsoa["Depression_Prevalence"] = safe_divide(lsoa["Allocated_Depression"], lsoa["LSOA_Total_List"])
    lsoa["SMI_Prevalence"] = safe_divide(lsoa["Allocated_SMI"], lsoa["LSOA_Total_List"])
    lsoa["Antidepressant_Items_Per_Patient"] = safe_divide(
        lsoa["Allocated_Antidepressant_Items"], lsoa["LSOA_Total_List"]
    )

    return lsoa, mismatch_summary


def normalize_weights(depression_weight: float, smi_weight: float, prescribing_weight: float) -> tuple[float, float, float]:
    weights = np.array([depression_weight, smi_weight, prescribing_weight], dtype=float)
    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = float(weights.sum())
    if np.isclose(total, 0.0):
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    weights = weights / total
    return float(weights[0]), float(weights[1]), float(weights[2])


def rebalance_weight_points(changed_key: str) -> None:
    keys = ["dep_weight", "smi_weight", "prescribing_weight"]
    if changed_key not in keys:
        return

    if st.session_state.get("_weight_rebalancing", False):
        return

    st.session_state["_weight_rebalancing"] = True
    try:
        changed_val = float(st.session_state.get(changed_key, 0.0))
        changed_val = min(max(changed_val, 0.0), 100.0)
        st.session_state[changed_key] = changed_val

        other_keys = [k for k in keys if k != changed_key]
        other_vals = [float(st.session_state.get(k, 0.0)) for k in other_keys]
        other_sum = sum(other_vals)
        remaining = max(0.0, 100.0 - changed_val)

        if np.isclose(other_sum, 0.0):
            for k in other_keys:
                st.session_state[k] = remaining / 2.0
        else:
            scale = remaining / other_sum
            for k in other_keys:
                st.session_state[k] = float(st.session_state.get(k, 0.0)) * scale

        # Keep display tidy while preserving exact total via final correction.
        for k in keys:
            st.session_state[k] = round(float(st.session_state[k]), 1)
        total = sum(float(st.session_state[k]) for k in keys)
        diff = round(100.0 - total, 1)
        st.session_state[other_keys[-1]] = round(float(st.session_state[other_keys[-1]]) + diff, 1)
    finally:
        st.session_state["_weight_rebalancing"] = False


def apply_need_score(
    lsoa_df: pd.DataFrame,
    depression_weight: float,
    smi_weight: float,
    prescribing_weight: float,
) -> pd.DataFrame:
    dep_w, smi_w, rx_w = normalize_weights(depression_weight, smi_weight, prescribing_weight)

    out = lsoa_df.copy()
    out["Depression_Normalized"] = minmax_scale(out["Depression_Prevalence"])
    out["SMI_Normalized"] = minmax_scale(out["SMI_Prevalence"])
    out["Prescribing_Normalized"] = minmax_scale(out["Antidepressant_Items_Per_Patient"])
    out["Need_Score"] = (
        out["Depression_Normalized"].fillna(0) * dep_w
        + out["SMI_Normalized"].fillna(0) * smi_w
        + out["Prescribing_Normalized"].fillna(0) * rx_w
    )
    return out


def build_gp_marker_df(
    gp_locations: pd.DataFrame,
    gp_master: pd.DataFrame,
    mapping_df: pd.DataFrame,
    lsoa_centroids: pd.DataFrame,
    in_area_lsoa_codes: set[str],
) -> pd.DataFrame:
    dominant_lsoa = (
        mapping_df.sort_values("NUMBER_OF_PATIENTS", ascending=False)
        .drop_duplicates(subset=["PRACTICE_CODE"])[["PRACTICE_CODE", "LSOA_CODE"]]
        .rename(columns={"LSOA_CODE": "Dominant_LSOA"})
    )

    dominant_lsoa_in_area = (
        mapping_df[mapping_df["LSOA_CODE"].isin(in_area_lsoa_codes)]
        .sort_values("NUMBER_OF_PATIENTS", ascending=False)
        .drop_duplicates(subset=["PRACTICE_CODE"])[["PRACTICE_CODE", "LSOA_CODE"]]
        .rename(columns={"LSOA_CODE": "Dominant_LSOA_InArea"})
    )

    gp_registered_totals = (
        mapping_df.groupby("PRACTICE_CODE", as_index=False)["NUMBER_OF_PATIENTS"]
        .sum()
        .rename(columns={"NUMBER_OF_PATIENTS": "NUMBER_OF_PATIENTS"})
    )

    gp = gp_locations.merge(gp_master, on="PRACTICE_CODE", how="outer")
    gp = gp.merge(dominant_lsoa, on="PRACTICE_CODE", how="left")
    gp = gp.merge(dominant_lsoa_in_area, on="PRACTICE_CODE", how="left")
    gp = gp.merge(gp_registered_totals, on="PRACTICE_CODE", how="left")

    gp["Effective_LSOA"] = gp["GP_LSOA"].where(gp["GP_LSOA"].astype(str).str.len() > 0, gp["Dominant_LSOA"])
    gp["Effective_LSOA"] = gp["Effective_LSOA"].where(
        gp["Effective_LSOA"].isin(in_area_lsoa_codes), gp["Dominant_LSOA_InArea"]
    )

    gp = gp.merge(
        lsoa_centroids.rename(columns={"LSOA_CODE": "Effective_LSOA", "lat": "LSOA_Lat", "lon": "LSOA_Lon"}),
        on="Effective_LSOA",
        how="left",
    )

    gp["Lat"] = gp["GP_Lat"].where(gp["GP_Lat"].notna(), gp["LSOA_Lat"])
    gp["Lon"] = gp["GP_Lon"].where(gp["GP_Lon"].notna(), gp["LSOA_Lon"])

    gp["Practice_Name"] = gp["Practice_Name"].where(gp["Practice_Name"].astype(str).str.len() > 0, gp["GP_Name_File"])

    return gp


def make_popup_html(row: pd.Series) -> str:
    def fmt_num(value: object, ndp: int = 1) -> str:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{ndp}f}"

    def fmt_int(value: object) -> str:
        if pd.isna(value):
            return "N/A"
        return f"{int(round(float(value))):,}"

    name = html.escape(clean_text(row.get("Practice_Name", "Unknown GP")) or "Unknown GP")
    code = html.escape(clean_text(row.get("PRACTICE_CODE", "")))

    popup = f"""
    <div style=\"font-family:Arial,sans-serif; min-width:280px;\">
      <h4 style=\"margin:0 0 8px 0;\">{name}</h4>
      <div><b>Code:</b> {code}</div>
      <div><b>Total Registered Patients:</b> {fmt_int(row.get('NUMBER_OF_PATIENTS'))}</div>
      <hr style=\"margin:8px 0;\"/>
      <div><b>Depression Register (2024-25):</b> {fmt_int(row.get('Dep_Register'))}</div>
      <div><b>Depression Prevalence (2024-25):</b> {fmt_num(row.get('Dep_Prevalence_Pct'), 2)}%</div>
      <div><b>SMI Register (2024-25):</b> {fmt_int(row.get('SMI_Register'))}</div>
      <div><b>SMI Prevalence (2024-25):</b> {fmt_num(row.get('SMI_Prevalence_Pct'), 2)}%</div>
      <hr style=\"margin:8px 0;\"/>
            <div><b>Antidepressant Items (May 2026):</b> {fmt_int(row.get('Antidepressant_Items'))}</div>
            <div><b>Antidepressant Actual Cost (May 2026):</b> £{fmt_num(row.get('Antidepressant_Actual_Cost'), 2)}</div>
            <hr style=\"margin:8px 0;\"/>
      <div><b>Care Plan Achievement Rate (MH002):</b> {fmt_num(row.get('MH002_Pct'), 2)}%</div>
      <div><b>Avg Physical Health Review Rate:</b> {fmt_num(row.get('Physical_Health_Review_Avg_Pct'), 2)}%</div>
      <div><b>Exception Rate (Overall PCA):</b> {fmt_num(row.get('Exception_Rate_Pct'), 2)}%</div>
    </div>
    """
    return popup


def build_map(
    lsoa_gdf: gpd.GeoDataFrame,
    show_gps: bool,
    gp_marker_df: pd.DataFrame,
    dep_weight: float,
    smi_weight: float,
    prescribing_weight: float,
    max_gp_markers: int = 700,
) -> folium.Map:
    dep_w, smi_w, rx_w = normalize_weights(dep_weight, smi_weight, prescribing_weight)

    metric_config = {
        "Need_Score": {"label": "Need Index", "value_col": "Need_Score"},
        "Depression_Prevalence": {"label": "Depression Prevalence (%)", "value_col": "Depression_Prevalence"},
        "SMI_Prevalence": {"label": "SMI Prevalence (%)", "value_col": "SMI_Prevalence"},
        "Antidepressant_Items_Per_Patient": {
            "label": "Antidepressant Items Per Patient",
            "value_col": "Antidepressant_Items_Per_Patient",
        },
    }

    default_metric_key = "Need_Score"
    valid_geo = lsoa_gdf.dropna(subset=[metric_config[default_metric_key]["value_col"]]).copy()
    if valid_geo.empty:
        center_lat, center_lon = 53.23, -0.54
    else:
        center_pts = valid_geo.to_crs(epsg=27700).geometry.centroid
        center_pts = gpd.GeoSeries(center_pts, crs="EPSG:27700").to_crs(epsg=4326)
        center_lat, center_lon = center_pts.y.mean(), center_pts.x.mean()

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None)
    folium.TileLayer("CartoDB positron", name="CartoDB Positron", control=False).add_to(fmap)
    folium.map.CustomPane("lsoa_pane", z_index=400).add_to(fmap)
    folium.map.CustomPane("gp_pane", z_index=650).add_to(fmap)

    base_lsoa_draw = lsoa_gdf.copy().to_crs(epsg=27700)
    base_lsoa_draw["geometry"] = base_lsoa_draw.geometry.simplify(tolerance=15, preserve_topology=True)
    base_lsoa_draw = base_lsoa_draw.to_crs(epsg=4326)

    for col in ["Need_Score", "Depression_Prevalence", "SMI_Prevalence", "Antidepressant_Items_Per_Patient"]:
        base_lsoa_draw[col] = base_lsoa_draw[col].round(6)
    base_lsoa_draw["Depression_Prevalence_Pct"] = (base_lsoa_draw["Depression_Prevalence"] * 100).round(2)
    base_lsoa_draw["SMI_Prevalence_Pct"] = (base_lsoa_draw["SMI_Prevalence"] * 100).round(2)

    for metric_key, metric in metric_config.items():
        metric_col = metric["value_col"]
        valid_metric = base_lsoa_draw.dropna(subset=[metric_col])
        min_val = float(valid_metric[metric_col].min()) if not valid_metric.empty else 0.0
        max_val = float(valid_metric[metric_col].max()) if not valid_metric.empty else 1.0
        if np.isclose(min_val, max_val):
            max_val = min_val + 1e-9

        colormap = linear.YlOrRd_09.scale(min_val, max_val)
        lsoa_draw = base_lsoa_draw.copy()
        lsoa_draw["fill_color"] = lsoa_draw[metric_col].apply(
            lambda x: colormap(x) if pd.notna(x) else "#d9d9d9"
        )

        group = folium.FeatureGroup(name=metric["label"], overlay=False, control=True, show=(metric_key == default_metric_key))
        folium.GeoJson(
            lsoa_draw,
            pane="lsoa_pane",
            style_function=lambda feature: {
                "fillColor": feature["properties"].get("fill_color", "#d9d9d9"),
                "color": "#666666",
                "weight": 0.4,
                "fillOpacity": 0.8,
            },
            highlight_function=lambda _: {"weight": 1.2, "color": "#111111", "fillOpacity": 0.95},
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "LSOA_CODE",
                    "Need_Score",
                    "Depression_Prevalence_Pct",
                    "SMI_Prevalence_Pct",
                    "Antidepressant_Items_Per_Patient",
                ],
                aliases=[
                    "LSOA:",
                    "Need Index:",
                    "Depression Prevalence (%):",
                    "SMI Prevalence (%):",
                    "Antidepressant Items Per Patient:",
                ],
                localize=True,
                labels=True,
                sticky=True,
            ),
        ).add_to(group)
        group.add_to(fmap)

    weight_legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #bbb; border-radius: 4px; font-size: 12px;">
      <div style="font-weight: bold; margin-bottom: 4px;">Need Index Weights</div>
      <div>Depression: {dep_w:.2f}</div>
      <div>SMI: {smi_w:.2f}</div>
      <div>Prescribing: {rx_w:.2f}</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(weight_legend_html))

    if show_gps:
        markers = MarkerCluster(name="GP Locations", overlay=True, control=True)
        gp_points = gp_marker_df.dropna(subset=["Lat", "Lon"]).copy().head(max_gp_markers)
        for _, row in gp_points.iterrows():
            popup_html = make_popup_html(row)
            folium.Marker(
                location=[float(row["Lat"]), float(row["Lon"])],
                icon=folium.DivIcon(
                    html="""
                    <div style='width:10px;height:10px;border-radius:50%;
                    background:#1f78b4;border:1px solid #ffffff;opacity:0.9;'></div>
                    """
                ),
                popup=folium.Popup(popup_html, max_width=360),
                tooltip=f"{clean_text(row.get('Practice_Name', 'GP'))} ({clean_text(row.get('PRACTICE_CODE', ''))})",
            ).add_to(markers)
        markers.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def main() -> None:
    st.set_page_config(page_title="Lincolnshire Mental Health Index", layout="wide")

    st.markdown(
        """
        <style>
        div[data-testid="stMarkdownContainer"] table {
            background-color: #ffffff !important;
            opacity: 1 !important;
        }
        iframe {
            background-color: #ffffff !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Lincolnshire Mental Health Index")

    with st.expander("Method Summary", expanded=False):
        st.markdown(
            """
            This Index proportionally allocates GP-level QOF registers to LSOAs using the patient registration matrix.

            Allocation equations:
            - GP total mapped patients: sum of NUMBER_OF_PATIENTS across all mapped LSOAs for each GP.
            - Weight(i,j): NUMBER_OF_PATIENTS(i,j) / GP_Total_Mapped_Patients(j)
            - Allocated register(i,j): GP_Register(j) * Weight(i,j)
            - LSOA register(i): sum of allocated registers across all GPs.
            - LSOA prevalence(i): LSOA_Register(i) / LSOA_Total_List(i)

            Need Index:
            - Depression prevalence, SMI prevalence, and antidepressant items per patient are min-max normalized to [0,1].
            - Need Index = (Depression_Normalized * w_dep) + (SMI_Normalized * w_smi) + (Prescribing_Normalized * w_rx)
            """
        )

    base_dir = resolve_base_dir(Path(__file__))

    try:
        dep_raw, smi_raw, prescribing_raw, mapping_male_raw, mapping_female_raw, gp_loc_raw, lsoa_raw = load_raw_data(base_dir)
        dep_df = prepare_depression(dep_raw)
        smi_df = prepare_smi(smi_raw)
        prescribing_df = prepare_prescribing(prescribing_raw)
        mapping_df = prepare_mapping_from_sex_split(mapping_male_raw, mapping_female_raw)
        gp_loc_df = prepare_gp_locations(gp_loc_raw)
        lsoa_gdf = prepare_lsoa_geography(lsoa_raw)
        in_area_lsoa_codes = set(lsoa_gdf["LSOA_CODE"].dropna().unique())

        gp_master = build_gp_master(dep_df, smi_df, prescribing_df)

        mapped_lsoa, mismatch_summary = allocate_registers_to_lsoa(mapping_df, gp_master)
        out_of_area_lsoa_codes = set(mapped_lsoa["LSOA_CODE"].dropna().unique()) - in_area_lsoa_codes
        out_of_area_rows = mapping_df[mapping_df["LSOA_CODE"].isin(out_of_area_lsoa_codes)]
        out_of_area_patients = float(out_of_area_rows["NUMBER_OF_PATIENTS"].sum()) if not out_of_area_rows.empty else 0.0

        lsoa_metrics = lsoa_gdf.merge(mapped_lsoa, on="LSOA_CODE", how="left")

        lsoa_centroids = get_lsoa_centroids(lsoa_gdf)
        gp_marker_df = build_gp_marker_df(gp_loc_df, gp_master, mapping_df, lsoa_centroids, in_area_lsoa_codes)
    except Exception as exc:
        st.error("Failed to load or prepare data. Check file paths, column names, and data quality.")
        st.exception(exc)
        st.stop()

    if "dep_weight" not in st.session_state:
        st.session_state.dep_weight = 34.0
    if "smi_weight" not in st.session_state:
        st.session_state.smi_weight = 33.0
    if "prescribing_weight" not in st.session_state:
        st.session_state.prescribing_weight = 33.0
    if "_weight_rebalancing" not in st.session_state:
        st.session_state._weight_rebalancing = False
    if "show_gps" not in st.session_state:
        st.session_state.show_gps = True
    if "filter_enabled" not in st.session_state:
        st.session_state.filter_enabled = False
    if "filter_metric" not in st.session_state:
        st.session_state.filter_metric = "Need_Score"
    if "filter_percentile" not in st.session_state:
        st.session_state.filter_percentile = 80

    with st.sidebar:
        st.header("Need Index Controls")
        st.caption("Weight points always sum to 100.")
        st.slider(
            "Depression Weight Points",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key="dep_weight",
            on_change=rebalance_weight_points,
            args=("dep_weight",),
            help="Weight points (0-100). Final contribution is normalized with the other two controls.",
        )
        st.slider(
            "SMI Weight Points",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key="smi_weight",
            on_change=rebalance_weight_points,
            args=("smi_weight",),
            help="Weight points (0-100). Final contribution is normalized with the other two controls.",
        )
        st.slider(
            "Prescribing Weight Points",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key="prescribing_weight",
            on_change=rebalance_weight_points,
            args=("prescribing_weight",),
            help="Weight points (0-100). Final contribution is normalized with the other two controls.",
        )

    with st.sidebar.form("controls_form"):

        st.header("Map Layers")
        st.toggle("Show GP Locations Overlay", key="show_gps")
        st.caption("Choropleth metric is selected in the map key using radio buttons.")

        st.header("Map Filter")
        st.toggle("Filter to Areas Above Percentile", key="filter_enabled")
        st.selectbox(
            "Filter Metric",
            options=["Need_Score", "Depression_Prevalence", "SMI_Prevalence", "Antidepressant_Items_Per_Patient"],
            key="filter_metric",
            format_func=lambda x: {
                "Need_Score": "Need Index",
                "Depression_Prevalence": "Depression Prevalence",
                "SMI_Prevalence": "SMI Prevalence",
                "Antidepressant_Items_Per_Patient": "Antidepressant Items Per Patient",
            }[x],
        )
        st.slider(
            "Percentile Threshold",
            min_value=50,
            max_value=99,
            step=1,
            key="filter_percentile",
        )

        st.form_submit_button("Apply")

    dep_weight = float(st.session_state.dep_weight)
    smi_weight = float(st.session_state.smi_weight)
    prescribing_weight = float(st.session_state.prescribing_weight)
    show_gps = bool(st.session_state.show_gps)
    filter_enabled = bool(st.session_state.filter_enabled)
    filter_metric = str(st.session_state.filter_metric)
    filter_percentile = int(st.session_state.filter_percentile)

    dep_w, smi_w, rx_w = normalize_weights(dep_weight, smi_weight, prescribing_weight)
    raw_total = dep_weight + smi_weight + prescribing_weight
    st.sidebar.caption(f"Weight points total: {raw_total:.1f}")
    st.sidebar.caption(
        "Effective Need Index weights (always normalized to 100%): "
        f"Depression {dep_w * 100:.1f}% | SMI {smi_w * 100:.1f}% | Prescribing {rx_w * 100:.1f}%"
    )

    scored_lsoa = apply_need_score(lsoa_metrics, dep_weight, smi_weight, prescribing_weight)

    display_lsoa = scored_lsoa.copy()
    threshold_value = None
    if filter_enabled:
        metric_series = display_lsoa[filter_metric].dropna()
        if not metric_series.empty:
            threshold_value = float(np.nanpercentile(metric_series, filter_percentile))
            display_lsoa = display_lsoa[display_lsoa[filter_metric] >= threshold_value].copy()

    active_view = st.radio(
        "View",
        options=["Interactive Map", "Data Explorer"],
        horizontal=True,
        index=0,
    )

    if active_view == "Interactive Map":
        if display_lsoa.empty:
            st.warning("No LSOAs match the selected filter. Relax the percentile threshold or disable filtering.")
        else:
            fmap = build_map(
                display_lsoa,
                show_gps,
                gp_marker_df,
                dep_weight,
                smi_weight,
                prescribing_weight,
            )
            # Static rendering avoids component-driven rerun loops that can cause visible flicker.
            folium_static(fmap, width=None, height=700)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LSOAs Displayed", f"{display_lsoa['LSOA_CODE'].nunique():,}")
        c2.metric("Mapped GP Practices", f"{mapping_df['PRACTICE_CODE'].nunique():,}")
        c3.metric("GPs with map coordinates", f"{gp_marker_df.dropna(subset=['Lat', 'Lon'])['PRACTICE_CODE'].nunique():,}")
        if threshold_value is not None:
            c4.metric("Applied Threshold", f"{threshold_value:.4f}")
        else:
            c4.metric("Applied Threshold", "None")

        practices_in_mapping = set(mapping_df["PRACTICE_CODE"].unique())
        practices_in_qof = set(gp_master["PRACTICE_CODE"].dropna().unique())
        practices_in_gp_locations = set(gp_loc_df["PRACTICE_CODE"].dropna().unique())
        qof_without_mapping = len(practices_in_qof - practices_in_mapping)
        gp_locations_without_qof = len(practices_in_gp_locations - practices_in_qof)

        if mismatch_summary["mapped_practices_without_any_qof"] > 0:
            st.warning(
                "There are mapped GP practices with no matching QOF records. "
                f"Count: {mismatch_summary['mapped_practices_without_any_qof']}. "
                "Those practices contribute to LSOA list size but not allocated register counts."
            )

        if mismatch_summary["mapped_practices_without_dep_qof"] > 0 or mismatch_summary["mapped_practices_without_smi_qof"] > 0:
            st.info(
                "QOF linkage details: "
                f"missing depression register for {mismatch_summary['mapped_practices_without_dep_qof']} mapped practices, "
                f"missing SMI register for {mismatch_summary['mapped_practices_without_smi_qof']} mapped practices."
            )

        if mismatch_summary["mapped_practices_without_prescribing"] > 0:
            st.info(
                "Prescribing linkage details: "
                f"missing antidepressant prescribing records for {mismatch_summary['mapped_practices_without_prescribing']} mapped practices."
            )

        if qof_without_mapping > 0:
            st.info(
                f"There are {qof_without_mapping} QOF practices with no mapping rows in the registration matrix."
            )

        if gp_locations_without_qof > 0:
            st.info(
                f"There are {gp_locations_without_qof} GP location records without matching QOF records."
            )

        if len(out_of_area_lsoa_codes) > 0:
            st.info(
                "The registration matrix includes out-of-Lincolnshire LSOAs that are not in the boundary file. "
                f"Distinct out-of-area LSOAs: {len(out_of_area_lsoa_codes):,}. "
                f"Patient records in those LSOAs: {out_of_area_patients:,.0f}. "
                "They are retained in GP allocation weights but excluded from Lincolnshire map polygons."
            )

    if active_view == "Data Explorer":
        explorer_cols = [
            "LSOA_CODE",
            "LSOA_Total_List",
            "Allocated_Depression",
            "Allocated_SMI",
            "Allocated_Antidepressant_Items",
            "Depression_Prevalence",
            "SMI_Prevalence",
            "Antidepressant_Items_Per_Patient",
            "Need_Score",
        ]

        explorer = scored_lsoa[explorer_cols].copy()
        explorer = explorer.rename(
            columns={
                "LSOA_CODE": "LSOA Code",
                "LSOA_Total_List": "List Size",
                "Allocated_Depression": "Mapped Depression Count",
                "Allocated_SMI": "Mapped SMI Count",
                "Allocated_Antidepressant_Items": "Mapped Antidepressant Items",
                "Depression_Prevalence": "Depression Prevalence",
                "SMI_Prevalence": "SMI Prevalence",
                "Antidepressant_Items_Per_Patient": "Antidepressant Items Per Patient",
                "Need_Score": "Need Index",
            }
        )

        search_text = st.text_input("Search rows", placeholder="Type LSOA code or value...")
        if search_text:
            mask = explorer.astype(str).apply(lambda col: col.str.contains(search_text, case=False, na=False))
            explorer_view = explorer[mask.any(axis=1)].copy()
        else:
            explorer_view = explorer.copy()

        explorer_view = explorer_view.sort_values("Need Index", ascending=False)

        page_size = st.select_slider(
            "Rows per page",
            options=[25, 50, 100, 200, 500],
            value=100,
            help="Smaller pages reduce websocket load and prevent flicker.",
        )
        total_rows = len(explorer_view)
        total_pages = max(1, int(np.ceil(total_rows / page_size)))
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (int(page_num) - 1) * page_size
        end_idx = start_idx + page_size
        explorer_preview = explorer_view.iloc[start_idx:end_idx].copy()

        # Use HTML rendering instead of st.dataframe to avoid pyarrow-based dataframe
        # serialization crashes observed in some Linux geospatial environments.
        st.markdown(
            explorer_preview.to_html(index=False, escape=True),
            unsafe_allow_html=True,
        )

        if total_rows > page_size:
            st.info(
                f"Showing rows {start_idx + 1:,}-{min(end_idx, total_rows):,} of {total_rows:,}. "
                "Use the CSV download for the complete filtered dataset."
            )

        csv_data = explorer_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current table as CSV",
            data=csv_data,
            file_name="lincolnshire_lsoa_mental_health_need.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    if is_running_via_streamlit():
        main()
    else:
        print("This is a Streamlit app.")
        print("Run it with one of these commands:")
        print("  streamlit run app.py")
        print("  streamlit run scripts/app.py")
