import html

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import linear
from folium.plugins import MarkerCluster

from analysis.common import clean_text


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


def _resolve_scale(scale_name: str):
    scale = getattr(linear, scale_name, None)
    if scale is None:
        return linear.YlOrRd_09
    return scale


def build_choropleth_map(
    lsoa_gdf: gpd.GeoDataFrame,
    metric_layers: list[dict[str, object]],
    tooltip_fields: list[str],
    tooltip_aliases: list[str],
    show_gps: bool = False,
    gp_marker_df: pd.DataFrame | None = None,
    weight_legend_lines: list[str] | None = None,
    max_gp_markers: int = 700,
    zoom_start: int = 9,
) -> folium.Map:
    default_col = str(metric_layers[0]["value_col"])
    valid_geo = lsoa_gdf.dropna(subset=[default_col])

    if valid_geo.empty:
        center_lat, center_lon = 53.23, -0.54
    else:
        center_pts = valid_geo.to_crs(epsg=27700).geometry.centroid
        center_pts = gpd.GeoSeries(center_pts, crs="EPSG:27700").to_crs(epsg=4326)
        center_lat, center_lon = center_pts.y.mean(), center_pts.x.mean()

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None)
    folium.TileLayer("CartoDB positron", name="CartoDB Positron", control=False).add_to(fmap)
    folium.map.CustomPane("lsoa_pane", z_index=400).add_to(fmap)
    folium.map.CustomPane("gp_pane", z_index=650).add_to(fmap)

    # `lsoa_gdf` is expected to be the cached, map-ready geometry base.
    base_draw = lsoa_gdf

    for layer in metric_layers:
        metric_key = str(layer["key"])
        label = str(layer["label"])
        metric_col = str(layer["value_col"])
        scale_name = str(layer.get("scale", "YlOrRd_09"))

        valid_metric = base_draw.dropna(subset=[metric_col])
        min_val = float(valid_metric[metric_col].min()) if not valid_metric.empty else 0.0
        max_val = float(valid_metric[metric_col].max()) if not valid_metric.empty else 1.0
        if np.isclose(min_val, max_val):
            max_val = min_val + 1e-9

        color_scale = _resolve_scale(scale_name)
        colormap = color_scale.scale(min_val, max_val)

        group = folium.FeatureGroup(
            name=label,
            overlay=False,
            control=True,
            show=bool(layer.get("default", metric_key == metric_layers[0]["key"])),
        )
        folium.GeoJson(
            base_draw,
            pane="lsoa_pane",
            # Compute the layer color directly from the feature properties to avoid
            # building a throwaway DataFrame copy for every metric layer.
            style_function=lambda feature, metric_col=metric_col, colormap=colormap: {
                "fillColor": colormap(feature["properties"].get(metric_col))
                if pd.notna(feature["properties"].get(metric_col))
                else "#d9d9d9",
                "color": "#666666",
                "weight": 0.4,
                "fillOpacity": 0.8,
            },
            highlight_function=lambda _: {"weight": 1.2, "color": "#111111", "fillOpacity": 0.95},
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                labels=True,
                sticky=True,
            ),
        ).add_to(group)
        group.add_to(fmap)

    if weight_legend_lines:
        line_html = "".join([f"<div>{html.escape(line)}</div>" for line in weight_legend_lines])
        weight_legend_html = f"""
        <div style=\"position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #bbb; border-radius: 4px; font-size: 12px;\">
          <div style=\"font-weight: bold; margin-bottom: 4px;\">Need Index Weights</div>
          {line_html}
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(weight_legend_html))

    if show_gps and gp_marker_df is not None:
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
