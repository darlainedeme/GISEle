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
    "main_roads_buffer": 'data/2_downloaded_input_data/roads/osm_roads_buffer.geojson',
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
def create_map(data_gdf=None, draw_enabled=False):
    m = folium.Map(location=[0, 0], zoom_start=2)

    if data_gdf is not None and not data_gdf.empty:
        bounds = data_gdf.geometry.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        folium.GeoJson(data_gdf).add_to(m)
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
        "Out of the Study Area": ["Major Cities", "Main Roads Buffer", "Airports", "Ports", "National Grid", "Substations", "Night Time Lights"],
        "Within the Study Area": ["Buildings", "Points of Interest", "Access Status", "Relative Wealth Index", "Roads", "Elevation", "Crops and Biomass Potential", "Water Bodies and Hydro Potential", "Solar Potential", "Wind Potential", "Landcover", "Available Land for Infrastructure"]
    }

    selected_section = st.sidebar.radio("Section", list(sections.keys()))
    selected_subpage = st.sidebar.radio("Subpage", sections[selected_section])

    data_key = selected_subpage.lower().replace(' ', '_')
    if data_key not in DATA_PATHS:
        st.write("Work in Progress")
        return

    # Load data only once and save it in session state
    if "data_gdf" not in st.session_state or st.session_state.selected_subpage != selected_subpage:
        st.session_state.selected_subpage = selected_subpage
        st.session_state.data_gdf = load_data(DATA_PATHS[data_key])
        st.session_state.map = create_map(st.session_state.data_gdf, draw_enabled=True)

    # Display the map
    st_folium(st.session_state.map, width=1400, height=800)

    # Save enhanced data if any
    map_output = st.session_state.get('map_output', None)
    if map_output and 'last_active_drawing' in map_output and map_output['last_active_drawing']:
        drawn_data = map_output['last_active_drawing']['geometry']
        drawn_gdf = gpd.GeoDataFrame([drawn_data], columns=['geometry'], crs='EPSG:4326')
        enhanced_gdf = pd.concat([st.session_state.data_gdf, drawn_gdf], ignore_index=True)
        save_enhanced_data(enhanced_gdf, DATA_PATHS[data_key])
        st.success(f"Enhanced data saved for {selected_subpage}")

    st.session_state['map_output'] = st_folium(st.session_state.map, width=1400, height=800)

if __name__ == "__main__":
    show()
