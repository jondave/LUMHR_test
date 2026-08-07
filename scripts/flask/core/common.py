from __future__ import annotations

import numpy as np
import pandas as pd


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
