from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from core.need_index import apply_need_index
from core.access_index import apply_access_index
from core.rural_risk_index import apply_rural_risk_index
from core.samhi import SAMHI_YEARS, get_samhi_columns
from data_loader import get_prepared_bundle_cached, resolve_base_dir

app = Flask(__name__, template_folder="templates")

BASE_DIR = resolve_base_dir(Path(__file__))
DATASETS_DIR = BASE_DIR / "datasets"
GEOJSON_REL_PATH = "lincolnshire_lsoa/lower-super-output-areas-2021-5RrVTw.geojson"

# Load once at startup so API requests only do lightweight transforms.
BUNDLE = get_prepared_bundle_cached(str(BASE_DIR))
LSOA_METRICS = BUNDLE["lsoa_metrics"].copy()
GP_MARKER_DF = BUNDLE["gp_marker_df"].copy()


def _parse_float_arg(name: str, default: float) -> float:
    raw = request.args.get(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for '{name}': {raw}") from exc


def _parse_int_arg(name: str, default: int) -> int:
    raw = request.args.get(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for '{name}': {raw}") from exc


def _scores_to_dict(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    valid = df[["LSOA_CODE", score_col]].dropna(subset=["LSOA_CODE", score_col]).copy()
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    valid = valid.dropna(subset=[score_col])
    return {str(code): float(value) for code, value in zip(valid["LSOA_CODE"], valid[score_col])}


def _num_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _get_lsoa_name_column(df: pd.DataFrame) -> str | None:
    candidates = ["LSOA21NM", "LSOA21NM_x", "LSOA21NM_y", "LSOA_NAME", "NAME"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


@app.get("/")
def home_page() -> str:
    return render_template("home.html")


@app.get("/need_index")
def need_index_page() -> str:
    out_lsoa_count = len(BUNDLE["out_of_area_lsoa_codes"])
    out_patients = int(round(float(BUNDLE["out_of_area_patients"])))
    return render_template(
        "need_index.html",
        geojson_rel_path=GEOJSON_REL_PATH,
        out_of_area_lsoa_count=out_lsoa_count,
        out_of_area_lsoa_count_fmt=f"{out_lsoa_count:,}",
        out_of_area_patients=out_patients,
        out_of_area_patients_fmt=f"{out_patients:,}",
    )


@app.get("/samhi")
def samhi_page() -> str:
    return render_template(
        "samhi.html",
        geojson_rel_path=GEOJSON_REL_PATH,
        min_year=min(SAMHI_YEARS),
        max_year=max(SAMHI_YEARS),
        default_year=max(SAMHI_YEARS),
    )


@app.get("/access_index")
def access_index_page() -> str:
    out_lsoa_count = len(BUNDLE["out_of_area_lsoa_codes"])
    out_patients = int(round(float(BUNDLE["out_of_area_patients"])))
    return render_template(
        "access_index.html",
        geojson_rel_path=GEOJSON_REL_PATH,
        out_of_area_lsoa_count_fmt=f"{out_lsoa_count:,}",
        out_of_area_patients_fmt=f"{out_patients:,}",
    )


@app.get("/access_gap_index")
def access_gap_index_page() -> str:
    out_lsoa_count = len(BUNDLE["out_of_area_lsoa_codes"])
    out_patients = int(round(float(BUNDLE["out_of_area_patients"])))
    return render_template(
        "access_gap_index.html",
        geojson_rel_path=GEOJSON_REL_PATH,
        out_of_area_lsoa_count_fmt=f"{out_lsoa_count:,}",
        out_of_area_patients_fmt=f"{out_patients:,}",
        min_year=min(SAMHI_YEARS),
        max_year=max(SAMHI_YEARS),
    )


@app.get("/rural_risk_index")
def rural_risk_index_page() -> str:
    return render_template(
        "rural_risk_index.html",
        geojson_rel_path=GEOJSON_REL_PATH,
    )


@app.get("/datasets/<path:filename>")
def dataset_files(filename: str):
    return send_from_directory(DATASETS_DIR, filename)


@app.get("/api/gp_locations")
def gp_locations_api():
    cols = [
        "PRACTICE_CODE",
        "Practice_Name",
        "Lat",
        "Lon",
        "NUMBER_OF_PATIENTS",
        "Dep_Register",
        "SMI_Register",
        "Dep_Prevalence_Pct",
        "SMI_Prevalence_Pct",
        "Antidepressant_Items",
        "Antidepressant_Actual_Cost",
        "MH002_Pct",
        "Physical_Health_Review_Avg_Pct",
        "Exception_Rate_Pct",
        "MH021_Pct",
        "Dep_Exception_Rate_Pct",
        "SMI_Exception_Rate_Pct",
        "DEP004_Pct",
        "Effective_LSOA",
    ]
    available_cols = [c for c in cols if c in GP_MARKER_DF.columns]
    out = GP_MARKER_DF[available_cols].copy()
    out["Lat"] = pd.to_numeric(out.get("Lat"), errors="coerce")
    out["Lon"] = pd.to_numeric(out.get("Lon"), errors="coerce")
    out = out.dropna(subset=["Lat", "Lon"])

    markers: list[dict[str, object]] = []
    for row in out.to_dict(orient="records"):
        marker = {
            "practice_code": str(row.get("PRACTICE_CODE", "")),
            "practice_name": str(row.get("Practice_Name", "") or ""),
            "lat": float(row["Lat"]),
            "lon": float(row["Lon"]),
            "patients": _num_or_none(row.get("NUMBER_OF_PATIENTS")),
            "dep_register": _num_or_none(row.get("Dep_Register")),
            "dep_prev_pct": _num_or_none(row.get("Dep_Prevalence_Pct")),
            "smi_register": _num_or_none(row.get("SMI_Register")),
            "smi_prev_pct": _num_or_none(row.get("SMI_Prevalence_Pct")),
            "antidepressant_items": _num_or_none(row.get("Antidepressant_Items")),
            "antidepressant_actual_cost": _num_or_none(row.get("Antidepressant_Actual_Cost")),
            "mh002_pct": _num_or_none(row.get("MH002_Pct")),
            "physical_health_review_avg_pct": _num_or_none(row.get("Physical_Health_Review_Avg_Pct")),
            "exception_rate_pct": _num_or_none(row.get("Exception_Rate_Pct")),
            "mh021_pct": _num_or_none(row.get("MH021_Pct")),
            "mh_pca_pct": _num_or_none(row.get("SMI_Exception_Rate_Pct")),
            "dep_pca_pct": _num_or_none(row.get("Dep_Exception_Rate_Pct")),
            "dep004_pct": _num_or_none(row.get("DEP004_Pct")),
            "effective_lsoa": str(row.get("Effective_LSOA", "") or ""),
        }
        markers.append(marker)

    return jsonify(markers)


@app.get("/api/need_scores")
def need_scores_api():
    try:
        dep = _parse_float_arg("dep", 25.0)
        smi = _parse_float_arg("smi", 25.0)
        prescribing = _parse_float_arg("prescribing", 25.0)
        samhi_weight = _parse_float_arg("samhi", 25.0)
        samhi_year = _parse_int_arg("samhi_year", max(SAMHI_YEARS))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if samhi_year not in SAMHI_YEARS:
        return jsonify({"error": f"samhi_year must be one of {SAMHI_YEARS}"}), 400

    samhi_index_col, _ = get_samhi_columns(samhi_year)
    if samhi_index_col not in LSOA_METRICS.columns:
        return jsonify({"error": f"Missing SAMHI column: {samhi_index_col}"}), 400

    scored = apply_need_index(
        LSOA_METRICS,
        dep,
        smi,
        prescribing,
        samhi_weight,
        samhi_index_col,
    )
    scored["Depression_Prevalence_Pct"] = pd.to_numeric(scored["Depression_Prevalence"], errors="coerce") * 100.0
    scored["SMI_Prevalence_Pct"] = pd.to_numeric(scored["SMI_Prevalence"], errors="coerce") * 100.0

    layers = {
        "Need_Index": _scores_to_dict(scored, "Need_Index"),
        "Depression_Prevalence": _scores_to_dict(scored, "Depression_Prevalence"),
        "SMI_Prevalence": _scores_to_dict(scored, "SMI_Prevalence"),
        "Antidepressant_Items_Per_Patient": _scores_to_dict(scored, "Antidepressant_Items_Per_Patient"),
        "SAMHI_Selected": _scores_to_dict(scored, "SAMHI_Selected"),
        "Pct_65plus": _scores_to_dict(scored, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(scored, "GP_Registration_Rate_Pct"),
    }

    lsoa_name_col = _get_lsoa_name_column(scored)

    detail_cols = [
        "LSOA_CODE",
        "Need_Index",
        "Depression_Prevalence_Pct",
        "SMI_Prevalence_Pct",
        "Antidepressant_Items_Per_Patient",
        "SAMHI_Selected",
        "ONS_Pop_Total_2024",
        "ONS_Pop_18plus",
        "ONS_Pop_65plus",
        "ONS_Pop_0to17",
        "Pct_18plus",
        "Pct_65plus",
        "Pct_0to17",
        "GP_Registered_Patients",
        "GP_Registration_Rate_Pct",
        "Registration_Gap_Est",
        "List_Inflation_Est",
    ]
    if lsoa_name_col:
        detail_cols.insert(1, lsoa_name_col)
    details_df = scored[[c for c in detail_cols if c in scored.columns]].copy()
    lsoa_details: dict[str, dict[str, object]] = {}
    for row in details_df.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        lsoa_details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "need_index": _num_or_none(row.get("Need_Index")),
            "depression_prevalence_pct": _num_or_none(row.get("Depression_Prevalence_Pct")),
            "smi_prevalence_pct": _num_or_none(row.get("SMI_Prevalence_Pct")),
            "antidepressant_items_per_patient": _num_or_none(row.get("Antidepressant_Items_Per_Patient")),
            "samhi_selected": _num_or_none(row.get("SAMHI_Selected")),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
            "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
            "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_0to17": _num_or_none(row.get("Pct_0to17")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
            "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
        }

    return jsonify(
        {
            "layers": layers,
            "lsoa_details": lsoa_details,
            "meta": {
                "samhi_year": samhi_year,
                "samhi_label": f"SAMHI Index ({samhi_year})",
            },
        }
    )


@app.get("/api/access_scores")
def access_scores_api():
    try:
        mh002 = _parse_float_arg("mh002", 8.33)
        mh021 = _parse_float_arg("mh021", 8.33)
        mh_pca = _parse_float_arg("mh_pca", 8.33)
        dep_pca = _parse_float_arg("dep_pca", 8.33)
        dep004 = _parse_float_arg("dep004", 8.33)
        gp_pt = _parse_float_arg("gp_pt", 8.33)
        gp_car = _parse_float_arg("gp_car", 8.33)
        hosp_pt = _parse_float_arg("hosp_pt", 8.33)
        hosp_car = _parse_float_arg("hosp_car", 8.33)
        rural = _parse_float_arg("rural", 8.33)
        car = _parse_float_arg("car", 8.33)
        digital = _parse_float_arg("digital", 8.33)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    required_cols = [
        "MH002_Access_Pct",
        "MH021_Access_Pct",
        "MH_PCA_Access_Pct",
        "Dep_PCA_Access_Pct",
        "DEP004_Access_Pct",
        "GP_PT_Time",
        "GP_Car_Time",
        "Hosp_PT_Time",
        "Hosp_Car_Time",
        "Rural_Access",
        "Car_Access",
        "Digital_Access",
    ]
    missing = [c for c in required_cols if c not in LSOA_METRICS.columns]
    if missing:
        return jsonify({"error": f"Missing access columns: {', '.join(missing)}"}), 400

    scored = apply_access_index(
        LSOA_METRICS, mh002, mh021, mh_pca, dep_pca, dep004, gp_pt, gp_car, hosp_pt, hosp_car, rural, car, digital
    )

    layers = {
        "Access_Index": _scores_to_dict(scored, "Access_Index"),
        "MH002_Access_Pct": _scores_to_dict(scored, "MH002_Access_Pct"),
        "MH021_Access_Pct": _scores_to_dict(scored, "MH021_Access_Pct"),
        "MH_PCA_Access_Pct": _scores_to_dict(scored, "MH_PCA_Access_Pct"),
        "Dep_PCA_Access_Pct": _scores_to_dict(scored, "Dep_PCA_Access_Pct"),
        "DEP004_Access_Pct": _scores_to_dict(scored, "DEP004_Access_Pct"),
        "GP_PT_Time": _scores_to_dict(scored, "GP_PT_Time"),
        "GP_Car_Time": _scores_to_dict(scored, "GP_Car_Time"),
        "Hosp_PT_Time": _scores_to_dict(scored, "Hosp_PT_Time"),
        "Hosp_Car_Time": _scores_to_dict(scored, "Hosp_Car_Time"),
        "Rural_Access": _scores_to_dict(scored, "Rural_Access"),
        "Car_Access": _scores_to_dict(scored, "Car_Access"),
        "Digital_Access": _scores_to_dict(scored, "Digital_Access"),
        "Pct_65plus": _scores_to_dict(scored, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(scored, "GP_Registration_Rate_Pct"),
    }

    lsoa_name_col = _get_lsoa_name_column(scored)

    extra_detail_cols = [
        "Car_Access_Pct",
        "No_Cars_Pct",
        "One_Car_Pct",
        "Two_Cars_Pct",
        "Three_Plus_Cars_Pct",
        "Total_Households",
        "DERI_Score",
        "Demography_Score",
        "Deprivation_Score",
        "Broadband_Score",
        "Avg_Download_Speed_Mbps",
        "No_Superfast_Broadband_Pct",
        "Slow_Connections_Pct",
        "ONS_Pop_Total_2024",
        "ONS_Pop_18plus",
        "ONS_Pop_65plus",
        "ONS_Pop_0to17",
        "Pct_18plus",
        "Pct_65plus",
        "Pct_0to17",
        "GP_Registered_Patients",
        "GP_Registration_Rate_Pct",
        "Registration_Gap_Est",
        "List_Inflation_Est",
    ]
    detail_cols = [
        "LSOA_CODE",
        "Access_Index",
        "RUC21NM",
        *required_cols,
        *extra_detail_cols,
    ]
    if lsoa_name_col:
        detail_cols.insert(1, lsoa_name_col)
    details_df = scored[[c for c in detail_cols if c in scored.columns]].copy()
    lsoa_details: dict[str, dict[str, object]] = {}
    for row in details_df.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        lsoa_details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "access_index": _num_or_none(row.get("Access_Index")),
            "mh002_pct": _num_or_none(row.get("MH002_Access_Pct")),
            "mh021_pct": _num_or_none(row.get("MH021_Access_Pct")),
            "mh_pca_pct": _num_or_none(row.get("MH_PCA_Access_Pct")),
            "dep_pca_pct": _num_or_none(row.get("Dep_PCA_Access_Pct")),
            "dep004_pct": _num_or_none(row.get("DEP004_Access_Pct")),
            "gp_pt_time": _num_or_none(row.get("GP_PT_Time")),
            "gp_car_time": _num_or_none(row.get("GP_Car_Time")),
            "hosp_pt_time": _num_or_none(row.get("Hosp_PT_Time")),
            "hosp_car_time": _num_or_none(row.get("Hosp_Car_Time")),
            "rural_access": _num_or_none(row.get("Rural_Access")),
            "ruc21nm": str(row.get("RUC21NM", "") or ""),
            "car_access": _num_or_none(row.get("Car_Access")),
            "car_access_pct": _num_or_none(row.get("Car_Access_Pct")),
            "no_cars_pct": _num_or_none(row.get("No_Cars_Pct")),
            "one_car_pct": _num_or_none(row.get("One_Car_Pct")),
            "two_cars_pct": _num_or_none(row.get("Two_Cars_Pct")),
            "three_plus_cars_pct": _num_or_none(row.get("Three_Plus_Cars_Pct")),
            "total_households": _num_or_none(row.get("Total_Households")),
            "digital_access": _num_or_none(row.get("Digital_Access")),
            "deri_score": _num_or_none(row.get("DERI_Score")),
            "demography_score": _num_or_none(row.get("Demography_Score")),
            "deprivation_score": _num_or_none(row.get("Deprivation_Score")),
            "broadband_score": _num_or_none(row.get("Broadband_Score")),
            "avg_download_speed_mbps": _num_or_none(row.get("Avg_Download_Speed_Mbps")),
            "no_superfast_broadband_pct": _num_or_none(row.get("No_Superfast_Broadband_Pct")),
            "slow_connections_pct": _num_or_none(row.get("Slow_Connections_Pct")),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
            "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
            "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_0to17": _num_or_none(row.get("Pct_0to17")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
            "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
        }

    return jsonify(
        {
            "layers": layers,
            "lsoa_details": lsoa_details,
        }
    )


@app.get("/api/access_gap_scores")
def access_gap_scores_api():
    try:
        dep = _parse_float_arg("dep", 25.0)
        smi = _parse_float_arg("smi", 25.0)
        prescribing = _parse_float_arg("prescribing", 25.0)
        samhi_weight = _parse_float_arg("samhi", 25.0)
        samhi_year = _parse_int_arg("samhi_year", max(SAMHI_YEARS))
        mh002 = _parse_float_arg("mh002", 8.33)
        mh021 = _parse_float_arg("mh021", 8.33)
        mh_pca = _parse_float_arg("mh_pca", 8.33)
        dep_pca = _parse_float_arg("dep_pca", 8.33)
        dep004 = _parse_float_arg("dep004", 8.33)
        gp_pt = _parse_float_arg("gp_pt", 8.33)
        gp_car = _parse_float_arg("gp_car", 8.33)
        hosp_pt = _parse_float_arg("hosp_pt", 8.33)
        hosp_car = _parse_float_arg("hosp_car", 8.33)
        rural = _parse_float_arg("rural", 8.33)
        car = _parse_float_arg("car", 8.33)
        digital = _parse_float_arg("digital", 8.33)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if samhi_year not in SAMHI_YEARS:
        return jsonify({"error": f"samhi_year must be one of {SAMHI_YEARS}"}), 400

    samhi_index_col, _ = get_samhi_columns(samhi_year)
    if samhi_index_col not in LSOA_METRICS.columns:
        return jsonify({"error": f"Missing SAMHI column: {samhi_index_col}"}), 400

    access_required_cols = [
        "MH002_Access_Pct",
        "MH021_Access_Pct",
        "MH_PCA_Access_Pct",
        "Dep_PCA_Access_Pct",
        "DEP004_Access_Pct",
        "GP_PT_Time",
        "GP_Car_Time",
        "Hosp_PT_Time",
        "Hosp_Car_Time",
        "Rural_Access",
        "Car_Access",
        "Digital_Access",
    ]
    missing = [c for c in access_required_cols if c not in LSOA_METRICS.columns]
    if missing:
        return jsonify({"error": f"Missing access columns: {', '.join(missing)}"}), 400

    need_scored = apply_need_index(LSOA_METRICS, dep, smi, prescribing, samhi_weight, samhi_index_col)
    access_scored = apply_access_index(
        LSOA_METRICS, mh002, mh021, mh_pca, dep_pca, dep004, gp_pt, gp_car, hosp_pt, hosp_car, rural, car, digital
    )

    combined = need_scored[["LSOA_CODE", "Need_Index"]].merge(
        access_scored[["LSOA_CODE", "Access_Index"]], on="LSOA_CODE", how="outer"
    )
    combined["Access_Gap_Index"] = combined["Need_Index"] - combined["Access_Index"]

    pop_cols = [
        "ONS_Pop_Total_2024",
        "ONS_Pop_18plus",
        "ONS_Pop_65plus",
        "ONS_Pop_0to17",
        "Pct_18plus",
        "Pct_65plus",
        "Pct_0to17",
        "GP_Registered_Patients",
        "GP_Registration_Rate_Pct",
        "Registration_Gap_Est",
        "List_Inflation_Est",
    ]
    extra_join_cols = [c for c in pop_cols if c in LSOA_METRICS.columns]
    if extra_join_cols:
        combined = combined.merge(LSOA_METRICS[["LSOA_CODE", *extra_join_cols]], on="LSOA_CODE", how="left")

    lsoa_name_col = _get_lsoa_name_column(LSOA_METRICS)
    if lsoa_name_col:
        combined = combined.merge(LSOA_METRICS[["LSOA_CODE", lsoa_name_col]], on="LSOA_CODE", how="left")

    layers = {
        "Access_Gap_Index": _scores_to_dict(combined, "Access_Gap_Index"),
        "Need_Index": _scores_to_dict(combined, "Need_Index"),
        "Access_Index": _scores_to_dict(combined, "Access_Index"),
        "Pct_65plus": _scores_to_dict(combined, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(combined, "GP_Registration_Rate_Pct"),
    }

    lsoa_details: dict[str, dict[str, object]] = {}
    for row in combined.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        lsoa_details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "access_gap_index": _num_or_none(row.get("Access_Gap_Index")),
            "need_index": _num_or_none(row.get("Need_Index")),
            "access_index": _num_or_none(row.get("Access_Index")),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
            "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
            "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_0to17": _num_or_none(row.get("Pct_0to17")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
            "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
        }

    return jsonify(
        {
            "layers": layers,
            "lsoa_details": lsoa_details,
            "meta": {
                "samhi_year": samhi_year,
            },
        }
    )


@app.get("/api/samhi_scores")
def samhi_scores_api():
    mode = str(request.args.get("mode", "Index")).strip()
    analysis_mode = str(request.args.get("analysis_mode", "Single Year")).strip()

    mode_norm = mode.lower()
    if mode_norm not in {"index", "decile"}:
        return jsonify({"error": "mode must be 'Index' or 'Decile'"}), 400

    try:
        year = _parse_int_arg("year", max(SAMHI_YEARS))
        from_year = _parse_int_arg("from_year", min(SAMHI_YEARS))
        to_year = _parse_int_arg("to_year", max(SAMHI_YEARS))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    for val, name in [(year, "year"), (from_year, "from_year"), (to_year, "to_year")]:
        if val not in SAMHI_YEARS:
            return jsonify({"error": f"{name} must be one of {SAMHI_YEARS}"}), 400

    is_change = analysis_mode.lower() in {"change", "change between years"}

    if not is_change:
        index_col, dec_col = get_samhi_columns(year)
        selected_col = index_col if mode_norm == "index" else dec_col
        if selected_col not in LSOA_METRICS.columns:
            return jsonify({"error": f"Missing SAMHI column: {selected_col}"}), 400

        lsoa_name_col = _get_lsoa_name_column(LSOA_METRICS)
        cols = ["LSOA_CODE", selected_col]
        if lsoa_name_col:
            cols.insert(1, lsoa_name_col)
        pop_cols = ["ONS_Pop_Total_2024", "Pct_65plus", "Pct_18plus", "GP_Registered_Patients", "GP_Registration_Rate_Pct"]
        cols.extend([c for c in pop_cols if c in LSOA_METRICS.columns])
        out_df = LSOA_METRICS[cols].copy()
        out_df = out_df.rename(columns={selected_col: "score"})

        details: dict[str, dict[str, object]] = {}
        for row in out_df.to_dict(orient="records"):
            code = str(row.get("LSOA_CODE", "") or "")
            if not code:
                continue
            details[code] = {
                "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
                "value": _num_or_none(row.get("score")),
                "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
                "pct_65plus": _num_or_none(row.get("Pct_65plus")),
                "pct_18plus": _num_or_none(row.get("Pct_18plus")),
                "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
                "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            }

        return jsonify(
            {
                "scores": _scores_to_dict(out_df, "score"),
                "details": details,
                "meta": {
                    "analysis_mode": "Single Year",
                    "mode": mode,
                    "year": year,
                },
            }
        )

    from_index_col, from_dec_col = get_samhi_columns(from_year)
    to_index_col, to_dec_col = get_samhi_columns(to_year)
    from_col = from_index_col if mode_norm == "index" else from_dec_col
    to_col = to_index_col if mode_norm == "index" else to_dec_col

    missing = [col for col in [from_col, to_col] if col not in LSOA_METRICS.columns]
    if missing:
        return jsonify({"error": f"Missing SAMHI columns: {', '.join(missing)}"}), 400

    lsoa_name_col = _get_lsoa_name_column(LSOA_METRICS)
    cols = ["LSOA_CODE", from_col, to_col]
    if lsoa_name_col:
        cols.insert(1, lsoa_name_col)
    pop_cols = ["ONS_Pop_Total_2024", "Pct_65plus", "Pct_18plus", "GP_Registered_Patients", "GP_Registration_Rate_Pct"]
    cols.extend([c for c in pop_cols if c in LSOA_METRICS.columns])
    out_df = LSOA_METRICS[cols].copy()
    out_df["from_value"] = pd.to_numeric(out_df[from_col], errors="coerce")
    out_df["to_value"] = pd.to_numeric(out_df[to_col], errors="coerce")
    out_df["score"] = out_df["to_value"] - out_df["from_value"]

    details: dict[str, dict[str, object]] = {}
    for row in out_df.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "from_value": _num_or_none(row.get("from_value")),
            "to_value": _num_or_none(row.get("to_value")),
            "change": _num_or_none(row.get("score")),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
        }

    return jsonify(
        {
            "scores": _scores_to_dict(out_df, "score"),
            "details": details,
            "meta": {
                "analysis_mode": "Change Between Years",
                "mode": mode,
                "from_year": from_year,
                "to_year": to_year,
            },
        }
    )


@app.get("/api/rural_risk_scores")
def rural_risk_scores_api():
    try:
        rural_weight = _parse_float_arg("w_rural", 15.0)
        gp_pt_weight = _parse_float_arg("w_gp_pt", 15.0)
        gp_car_weight = _parse_float_arg("w_gp_car", 15.0)
        no_car_weight = _parse_float_arg("w_no_car", 15.0)
        imd_weight = _parse_float_arg("w_imd", 15.0)
        oac_weight = _parse_float_arg("w_oac", 10.0)
        household_weight = _parse_float_arg("w_household", 15.0)
        rural_weight = _parse_float_arg("w_rural", 14.3)
        gp_pt_weight = _parse_float_arg("w_gp_pt", 14.3)
        gp_car_weight = _parse_float_arg("w_gp_car", 14.3)
        no_car_weight = _parse_float_arg("w_no_car", 14.3)
        imd_weight = _parse_float_arg("w_imd", 14.3)
        oac_weight = _parse_float_arg("w_oac", 14.3)
        household_weight = _parse_float_arg("w_household", 14.2)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    scored = apply_rural_risk_index(
        LSOA_METRICS,
        rural_weight=rural_weight,
        gp_pt_weight=gp_pt_weight,
        gp_car_weight=gp_car_weight,
        no_car_weight=no_car_weight,
        imd_weight=imd_weight,
        oac_weight=oac_weight,
        household_weight=household_weight,
    )

    layers = {
        "Rural_Risk_Index": _scores_to_dict(scored, "Rural_Risk_Index"),
        "Rural_Isolation": _scores_to_dict(scored, "Rural_Isolation_Normalized"),
        "GP_PT_Travel_Time": _scores_to_dict(scored, "GP_PT_Time"),
        "GP_Car_Travel_Time": _scores_to_dict(scored, "GP_Car_Time"),
        "No_Cars_Pct": _scores_to_dict(scored, "No_Cars_Pct"),
        "IMD_2025_Decile": _scores_to_dict(scored, "IMD_2025_Decile"),
        "IMD_2025_Rank": _scores_to_dict(scored, "IMD_2025_Rank"),
        "LSOAC_Risk": _scores_to_dict(scored, "LSOAC_Risk_Normalized"),
        "Household_Vulnerability": _scores_to_dict(scored, "Household_Vulnerability_Normalized"),
        "Single_Pensioner_HH_Pct": _scores_to_dict(scored, "Single_Pensioner_HH_Pct"),
        "Non_Couple_HH_Pct": _scores_to_dict(scored, "Non_Couple_HH_Pct"),
        "Pensioner_Couple_HH_Pct": _scores_to_dict(scored, "Pensioner_Couple_HH_Pct"),
        "Lone_Parent_HH_Pct": _scores_to_dict(scored, "Lone_Parent_Dep_Children_HH_Pct"),
        "Pct_65plus": _scores_to_dict(scored, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(scored, "GP_Registration_Rate_Pct"),
    }

    lsoa_name_col = _get_lsoa_name_column(scored)
    lsoa_details: dict[str, dict[str, object]] = {}
    for row in scored.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        lsoa_details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "rural_risk_index": _num_or_none(row.get("Rural_Risk_Index")),
            "rural_isolation_score": _num_or_none(row.get("Rural_Isolation_Normalized")),
            "ruc21nm": str(row.get("RUC21NM", "") or ""),
            "gp_pt_time": _num_or_none(row.get("GP_PT_Time")),
            "gp_car_time": _num_or_none(row.get("GP_Car_Time")),
            "no_cars_pct": _num_or_none(row.get("No_Cars_Pct")),
            "imd_2025_rank": _num_or_none(row.get("IMD_2025_Rank")),
            "imd_2025_decile": _num_or_none(row.get("IMD_2025_Decile")),
            "supergroup_code": str(row.get("Supergroup_Code", "") or ""),
            "supergroup_name": str(row.get("Supergroup_Name", "") or ""),
            "group_code": str(row.get("Group_Code", "") or ""),
            "group_name": str(row.get("Group_Name", "") or ""),
            "subgroup_code": str(row.get("Subgroup_Code", "") or ""),
            "subgroup_name": str(row.get("Subgroup_Name", "") or ""),
            "total_households_2021": _num_or_none(row.get("Total_Households_2021")),
            "single_pensioner_hh_count": _num_or_none(row.get("Single_Pensioner_HH_Count")),
            "single_pensioner_hh_pct": _num_or_none(row.get("Single_Pensioner_HH_Pct")),
            "pensioner_couple_hh_count": _num_or_none(row.get("Pensioner_Couple_HH_Count")),
            "pensioner_couple_hh_pct": _num_or_none(row.get("Pensioner_Couple_HH_Pct")),
            "lone_parent_dep_hh_count": _num_or_none(row.get("Lone_Parent_Dep_Children_HH_Count")),
            "lone_parent_dep_hh_pct": _num_or_none(row.get("Lone_Parent_Dep_Children_HH_Pct")),
            "non_couple_hh_count": _num_or_none(row.get("Non_Couple_HH_Count")),
            "non_couple_hh_pct": _num_or_none(row.get("Non_Couple_HH_Pct")),
            "household_vulnerability_score": _num_or_none(row.get("Household_Vulnerability_Normalized")),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
            "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
            "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_0to17": _num_or_none(row.get("Pct_0to17")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
            "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
        }

    return jsonify(
        {
            "layers": layers,
            "lsoa_details": lsoa_details,
            "weights": {
                "rural_weight": rural_weight,
                "gp_pt_weight": gp_pt_weight,
                "gp_car_weight": gp_car_weight,
                "no_car_weight": no_car_weight,
                "imd_weight": imd_weight,
                "oac_weight": oac_weight,
                "household_weight": household_weight,
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
