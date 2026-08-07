from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from core.need_index import apply_need_score
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


@app.get("/datasets/<path:filename>")
def dataset_files(filename: str):
    return send_from_directory(DATASETS_DIR, filename)


@app.get("/api/gp-locations")
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
            "effective_lsoa": str(row.get("Effective_LSOA", "") or ""),
        }
        markers.append(marker)

    return jsonify(markers)


@app.get("/api/need-scores")
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

    scored = apply_need_score(
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
        "Need_Score": _scores_to_dict(scored, "Need_Score"),
        "Depression_Prevalence": _scores_to_dict(scored, "Depression_Prevalence"),
        "SMI_Prevalence": _scores_to_dict(scored, "SMI_Prevalence"),
        "Antidepressant_Items_Per_Patient": _scores_to_dict(scored, "Antidepressant_Items_Per_Patient"),
        "SAMHI_Selected": _scores_to_dict(scored, "SAMHI_Selected"),
    }

    lsoa_name_col = _get_lsoa_name_column(scored)

    detail_cols = [
        "LSOA_CODE",
        "Need_Score",
        "Depression_Prevalence_Pct",
        "SMI_Prevalence_Pct",
        "Antidepressant_Items_Per_Patient",
        "SAMHI_Selected",
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
            "need_score": _num_or_none(row.get("Need_Score")),
            "depression_prevalence_pct": _num_or_none(row.get("Depression_Prevalence_Pct")),
            "smi_prevalence_pct": _num_or_none(row.get("SMI_Prevalence_Pct")),
            "antidepressant_items_per_patient": _num_or_none(row.get("Antidepressant_Items_Per_Patient")),
            "samhi_selected": _num_or_none(row.get("SAMHI_Selected")),
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


@app.get("/api/samhi-scores")
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
