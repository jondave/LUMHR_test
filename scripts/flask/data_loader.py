from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.allocation import (
    allocate_registers_to_lsoa,
    build_gp_master,
    build_gp_marker_df,
    prepare_depression,
    prepare_gp_locations,
    prepare_mapping_from_sex_split,
    prepare_prescribing,
    prepare_smi,
)
from core.common import normalize_code
from core.samhi import join_samhi, prepare_samhi


def resolve_base_dir(script_file: Path) -> Path:
    script_dir = script_file.resolve().parent
    candidates = [script_dir, *script_dir.parents]
    for candidate in candidates:
        if (candidate / "datasets").exists():
            return candidate
    searched = "\n".join(str(candidate / "datasets") for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate the datasets directory. Searched:\n"
        f"{searched}"
    )


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
        "samhi": base_dir / "datasets" / "samhi" / "samhi_lincolnshire_2021_lsoa.csv",
    }


def _polygon_centroid_area(ring: list[list[float]]) -> tuple[float, float, float]:
    if not ring or len(ring) < 3:
        return 0.0, np.nan, np.nan

    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        cross = (x0 * y1) - (x1 * y0)
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    if np.isclose(area2, 0.0):
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        return 0.0, float(np.mean(xs)), float(np.mean(ys))

    area = area2 / 2.0
    cx = cx / (3.0 * area2)
    cy = cy / (3.0 * area2)
    return float(area), float(cx), float(cy)


def _feature_centroid(geometry: dict[str, object]) -> tuple[float, float]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    best_abs_area = -1.0
    best_lon = np.nan
    best_lat = np.nan

    polygons: list[list[list[float]]] = []
    if gtype == "Polygon":
        polygons = coords
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                polygons.append(poly[0])

    for ring in polygons:
        area, lon, lat = _polygon_centroid_area(ring)
        abs_area = abs(area)
        if abs_area > best_abs_area:
            best_abs_area = abs_area
            best_lon = lon
            best_lat = lat

    return float(best_lat), float(best_lon)


def _load_lsoa_codes_and_centroids(geojson_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(geojson_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    code_rows: list[dict[str, str]] = []
    centroid_rows: list[dict[str, object]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {}) or {}
        code = normalize_code(props.get("CODE") or props.get("LSOA_CODE") or props.get("LSOA21CD"))
        if not code:
            continue
        code_rows.append(
            {
                "LSOA_CODE": code,
                "LSOA21NM": str(props.get("NAME") or props.get("LSOA21NM") or "").strip(),
            }
        )
        geometry = feature.get("geometry", {}) or {}
        lat, lon = _feature_centroid(geometry)
        centroid_rows.append({"LSOA_CODE": code, "lat": lat, "lon": lon})

    lsoa_codes_df = pd.DataFrame(code_rows).drop_duplicates(subset=["LSOA_CODE"])
    if lsoa_codes_df.empty:
        raise ValueError(f"No LSOA codes found in GeoJSON: {geojson_path}")

    lsoa_centroids_df = pd.DataFrame(centroid_rows).drop_duplicates(subset=["LSOA_CODE"])
    return lsoa_codes_df, lsoa_centroids_df


def load_raw_data(base_dir_str: str) -> dict[str, object]:
    base_dir = Path(base_dir_str)
    paths = get_paths(base_dir)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found for {name}: {path}")

    lsoa_codes_df, lsoa_centroids_df = _load_lsoa_codes_and_centroids(paths["lsoa_geo"])

    return {
        "dep_raw": pd.read_csv(paths["depression"]),
        "smi_raw": pd.read_csv(paths["smi"]),
        "prescribing_raw": pd.read_csv(paths["prescribing"]),
        "mapping_male_raw": pd.read_csv(paths["mapping_male"]),
        "mapping_female_raw": pd.read_csv(paths["mapping_female"]),
        "gp_loc_raw": pd.read_csv(paths["gp_locations"]),
        "samhi_raw": pd.read_csv(paths["samhi"]),
        "lsoa_codes": lsoa_codes_df,
        "lsoa_centroids": lsoa_centroids_df,
    }


@lru_cache(maxsize=2)
def get_prepared_bundle_cached(base_dir_str: str) -> dict[str, object]:
    raw = load_raw_data(base_dir_str)

    dep_df = prepare_depression(raw["dep_raw"])
    smi_df = prepare_smi(raw["smi_raw"])
    prescribing_df = prepare_prescribing(raw["prescribing_raw"])
    mapping_df = prepare_mapping_from_sex_split(raw["mapping_male_raw"], raw["mapping_female_raw"])
    gp_loc_df = prepare_gp_locations(raw["gp_loc_raw"])
    samhi_df = prepare_samhi(raw["samhi_raw"])

    lsoa_codes_df: pd.DataFrame = raw["lsoa_codes"]
    lsoa_centroids_df: pd.DataFrame = raw["lsoa_centroids"]
    in_area_lsoa_codes = set(lsoa_codes_df["LSOA_CODE"].dropna().unique())

    gp_master = build_gp_master(dep_df, smi_df, prescribing_df)
    mapped_lsoa, mismatch_summary = allocate_registers_to_lsoa(mapping_df, gp_master)

    out_of_area_lsoa_codes = set(mapped_lsoa["LSOA_CODE"].dropna().unique()) - in_area_lsoa_codes
    out_of_area_rows = mapping_df[mapping_df["LSOA_CODE"].isin(out_of_area_lsoa_codes)]
    out_of_area_patients = float(out_of_area_rows["NUMBER_OF_PATIENTS"].sum()) if not out_of_area_rows.empty else 0.0

    lsoa_metrics = lsoa_codes_df.merge(mapped_lsoa, on="LSOA_CODE", how="left")
    lsoa_metrics = join_samhi(lsoa_metrics, samhi_df)

    gp_marker_df = build_gp_marker_df(gp_loc_df, gp_master, mapping_df, lsoa_centroids_df, in_area_lsoa_codes)

    return {
        "dep_df": dep_df,
        "smi_df": smi_df,
        "prescribing_df": prescribing_df,
        "mapping_df": mapping_df,
        "gp_loc_df": gp_loc_df,
        "gp_master": gp_master,
        "lsoa_metrics": lsoa_metrics,
        "gp_marker_df": gp_marker_df,
        "mismatch_summary": mismatch_summary,
        "out_of_area_lsoa_codes": out_of_area_lsoa_codes,
        "out_of_area_patients": out_of_area_patients,
    }
