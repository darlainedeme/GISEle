import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
import os
from folium.plugins import Draw, Fullscreen, MeasureControl
import pandas as pd

# Define paths
DATA_PATHS = {
    "major_cities": 'data/2_downloaded_input_data/cities/osm_cities.geojson',
    "main_roads": 'data/2_downloaded_input_data/roads/osm_roads.geojson',
    "airports": 'data/2_downloaded_input_data/airports/osm_airports.geojson',
    "ports": 'data/2_downloaded_input_data/ports/osm_ports.geojson',
    "national_grid": 'data/2_downloaded_input_data/grids/osm_grids.geojson',
    "substations": 'data/2_downloaded_input_data/substations/osm_substations.geojson',
    "night_lights": 'data/2_downloaded_input_data/access_status/night_lights.geojson',
    "buildings": 'data/2_downloaded_input_data/buildings/combined_buildings.geojson',
    "points_of_interest": 'data/2_downloaded_input_data/poi/osm_pois.geojson',
    "roads": 'data/2_downloaded_input_data/roads/osm_roads.geojson',
    "roads_buffer": 'data/2_downloaded_input_data/roads/osm_roads_buffer.geojson',
    "water_bodies": 'data/2_downloaded_input_data/water_bodies/osm_water_bodies.geojson',
    "elevation": 'data/2_downloaded_input_data/elevation/elevation_data.tif',
    "solar": 'data/2_downloaded_input_data/solar/solar_potential.geojson',
    "wind": 'data/2_downloaded_input_data/wind/wind_potential.geojson',
    # Add paths for other datasets as needed
}

# Load data function
def load_data(file_path):
    if os.path.exists(file_path):
        return gpd.read_file(file_path)
    return gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')

# Save enhanced data function
def save_enhanced_data(gdf, file_path):
    gdf.to_file(file_path, driver='GeoJSON')

# Create map function
def create_map(data_gdf=None, data_key=None, draw_enabled=False):
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    folium.TileLayer('cartodbpositron', name="Positron").add_to(m)
    folium.TileLayer('cartodbdark_matter', name="Dark Matter").add_to(m)
    folium.TileLayer(
        tiles='http://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Maps',
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles='http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Hybrid',
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m)


    if data_gdf is not None and not data_gdf.empty:
        bounds = data_gdf.geometry.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        if data_key == "buildings":
            # Add building layers with different colors based on the "source"
            unique_sources = data_gdf["source"].unique()
            colors = folium.colors.linear.Set1_09.scale(0, len(unique_sources))
            source_color_map = dict(zip(unique_sources, colors))

            for _, row in data_gdf.iterrows():
                folium.GeoJson(
                    row['geometry'],
                    tooltip=f"Source: {row['source']}",
                    style_function=lambda feature, color=source_color_map[row['source']]: {
                        'fillColor': color,
                        'color': color,
                        'weight': 1,
                        'fillOpacity': 0.6
                    }
                ).add_to(m)

            # Add a legend for building sources
            legend_html = """
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 200px; height: auto; 
                        background-color: white; z-index:9999; font-size:14px;
                        border:2px solid grey; padding: 10px;">
            <b>Buildings by Source</b><br>
            """
            for source, color in source_color_map.items():
                legend_html += f"<i style='background:{color}'></i> {source}<br>"
            legend_html += "</div>"

            m.get_root().html.add_child(folium.Element(legend_html))
        else:
            # For other layers, just add with a tooltip
            folium.GeoJson(
                data_gdf,
                tooltip=folium.GeoJsonTooltip(fields=data_gdf.columns.to_list(), labels=True)
            ).add_to(m)
    else:
        st.warning("This layer is not available for the selected area.")

    if draw_enabled:
        draw = Draw(
            export=True,
            filename='drawn_data.geojson',
            draw_options={'polygon': True, 'polyline': False, 'circle': False, 'rectangle': False, 'marker': False, 'circlemarker': False}
        )
        draw.add_to(m)

    Fullscreen().add_to(m)
    MeasureControl().add_to(m)

    return m

def show():
    sections = {
        "Out of the Study Area": ["Access Status", "Major Cities", "Roads Buffer", "Airports", "Ports", "National Grid", "Substations", "Night Time Lights"],
        "Within the Study Area": ["Buildings", "Points of Interest", "Relative Wealth Index", "Roads", "Elevation", "Crops and Biomass Potential", "Water Bodies and Hydro Potential", "Solar Potential", "Wind Potential", "Landcover", "Available Land for Infrastructure"]
    }

    selected_section = st.sidebar.radio("Section", list(sections.keys()))
    selected_subpage = st.sidebar.radio("Subpage", sections[selected_section])

    data_key = selected_subpage.lower().replace(' ', '_')
    if data_key not in DATA_PATHS:
        st.write("Work in Progress")
        return

    data_gdf = load_data(DATA_PATHS[data_key])

    # Create map centered on the data if available
    m = create_map(data_gdf, data_key=data_key, draw_enabled=True)
    map_output = st_folium(m, width=1400, height=800)

    # Save enhanced data if any
    if map_output and 'last_active_drawing' in map_output and map_output['last_active_drawing']:
        drawn_data = map_output['last_active_drawing']['geometry']
        drawn_gdf = gpd.GeoDataFrame([drawn_data], columns=['geometry'], crs='EPSG:4326')
        enhanced_gdf = pd.concat([data_gdf, drawn_gdf], ignore_index=True)
        save_enhanced_data(enhanced_gdf, DATA_PATHS[data_key])
        st.success(f"Enhanced data saved for {selected_subpage}")

if __name__ == "__main__":
    show()
