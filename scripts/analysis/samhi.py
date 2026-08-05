import pandas as pd

from analysis.common import normalize_code


SAMHI_YEARS = list(range(2011, 2023))


def prepare_samhi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["LSOA21CD"] = out["LSOA21CD"].map(normalize_code)

    keep_cols = ["LSOA21CD", "LSOA21NM"]
    for year in SAMHI_YEARS:
        keep_cols.append(f"samhi_index.{year}")
        keep_cols.append(f"samhi_dec.{year}")

    existing_cols = [c for c in keep_cols if c in out.columns]
    out = out[existing_cols].copy()

    for year in SAMHI_YEARS:
        index_col = f"samhi_index.{year}"
        dec_col = f"samhi_dec.{year}"
        if index_col in out.columns:
            out[index_col] = pd.to_numeric(out[index_col], errors="coerce")
        if dec_col in out.columns:
            out[dec_col] = pd.to_numeric(out[dec_col], errors="coerce")

    return out


def join_samhi(lsoa_gdf: pd.DataFrame, samhi_df: pd.DataFrame) -> pd.DataFrame:
    merged = lsoa_gdf.merge(samhi_df, left_on="LSOA_CODE", right_on="LSOA21CD", how="left")
    if "LSOA21CD" in merged.columns:
        merged = merged.drop(columns=["LSOA21CD"])
    return merged


def get_samhi_columns(year: int) -> tuple[str, str]:
    return f"samhi_index.{year}", f"samhi_dec.{year}"
