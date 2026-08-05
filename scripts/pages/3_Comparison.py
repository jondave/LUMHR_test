from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from analysis.need_index import apply_need_score
from analysis.samhi import SAMHI_YEARS, get_samhi_columns
from data.loader import get_prepared_bundle_cached, resolve_base_dir
from maps.folium_map import build_choropleth_map


def render() -> None:
    st.title("Comparison")

    base_dir = resolve_base_dir(Path(__file__))
    try:
        bundle = get_prepared_bundle_cached(str(base_dir))
    except Exception as exc:
        st.error("Failed to load comparison data.")
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.header("Comparison Controls")
        year = st.slider("SAMHI Year", min_value=min(SAMHI_YEARS), max_value=max(SAMHI_YEARS), value=2022, step=1)
        mode = st.radio("Mode", options=["Need vs SAMHI", "Difference"], index=0)

    dep_weight = float(st.session_state.get("dep_weight", 25.0))
    smi_weight = float(st.session_state.get("smi_weight", 25.0))
    prescribing_weight = float(st.session_state.get("prescribing_weight", 25.0))
    samhi_weight = float(st.session_state.get("samhi_weight", 25.0))

    samhi_index_col, _ = get_samhi_columns(year)
    scored = apply_need_score(
        bundle["lsoa_metrics"],
        dep_weight,
        smi_weight,
        prescribing_weight,
        samhi_weight,
        samhi_index_col,
    )
    scored = scored.dropna(subset=["Need_Score", "SAMHI_Selected"]).copy()
    if scored.empty:
        st.warning("No overlapping Need Index and SAMHI values for the selected year.")
        st.stop()

    corr = float(scored["Need_Score"].corr(scored["SAMHI_Selected"], method="pearson"))
    st.metric("Pearson Correlation (Need vs SAMHI)", f"{corr:.4f}")

    if mode == "Need vs SAMHI":
        left, right = st.columns(2)

        with left:
            st.subheader("Need Index")
            need_map = build_choropleth_map(
                scored,
                metric_layers=[
                    {
                        "key": "Need_Score",
                        "label": "Need Index",
                        "value_col": "Need_Score",
                        "scale": "YlOrRd_09",
                        "default": True,
                    }
                ],
                tooltip_fields=["LSOA_CODE", "Need_Score"],
                tooltip_aliases=["LSOA:", "Need Index:"],
                show_gps=False,
            )
            folium_static(need_map, width=None, height=520)

        with right:
            st.subheader(f"SAMHI Index ({year})")
            samhi_map = build_choropleth_map(
                scored,
                metric_layers=[
                    {
                        "key": "SAMHI_Selected",
                        "label": f"SAMHI Index ({year})",
                        "value_col": "SAMHI_Selected",
                        "scale": "YlGnBu_09",
                        "default": True,
                    }
                ],
                tooltip_fields=["LSOA_CODE", "SAMHI_Selected"],
                tooltip_aliases=["LSOA:", f"SAMHI Index ({year}):"],
                show_gps=False,
            )
            folium_static(samhi_map, width=None, height=520)
    else:
        scored["Need_minus_SAMHI"] = scored["Need_Score"] - scored["SAMHI_Normalized"]
        st.subheader("Difference Map (Need Index minus SAMHI Normalized)")
        diff_map = build_choropleth_map(
            scored,
            metric_layers=[
                {
                    "key": "Need_minus_SAMHI",
                    "label": "Need Index minus SAMHI",
                    "value_col": "Need_minus_SAMHI",
                    "scale": "RdYlBu_11",
                    "default": True,
                }
            ],
            tooltip_fields=["LSOA_CODE", "Need_Score", "SAMHI_Normalized", "Need_minus_SAMHI"],
            tooltip_aliases=["LSOA:", "Need Index:", "SAMHI Normalized:", "Difference:"],
            show_gps=False,
        )
        folium_static(diff_map, width=None, height=650)

    st.subheader("Need vs SAMHI Scatter")
    scatter_df = pd.DataFrame({"Need Index": scored["Need_Score"], f"SAMHI Index ({year})": scored["SAMHI_Selected"]})
    st.scatter_chart(scatter_df, x="Need Index", y=f"SAMHI Index ({year})")


render()
