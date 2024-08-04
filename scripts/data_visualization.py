import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import json
import os
from folium.plugins import Draw, Fullscreen, MeasureControl

# Define paths
DATA_PATHS = {
    "major_cities": 'data/output/cities/osm_cities.geojson',
    "main_roads": 'data/output/roads/osm_roads.geojson',
    "airports": 'data/output/airports/osm_airports.geojson',
    "ports": 'data/output/ports/osm_ports.geojson',
    "national_grid": 'data/output/power_lines/osm_power_lines.geojson',
    "substations": 'data/output/substations/osm_substations.geojson',
    "night_lights": 'data/output/nighttime_lights/nighttime_lights.geojson',
    "buildings": 'data/output/buildings/combined_buildings.geojson',
    "points_of_interest": 'data/output/poi/osm_pois.geojson',
    "roads": 'data/output/roads/osm_roads.geojson',
    "water_bodies": 'data/output/water_bodies/osm_water_bodies.geojson'
    # Add paths for other datasets
}

# Load data function
def load_data(file_path):
    if os.path.exists(file_path):
        return gpd.read_file(file_path)
    return gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')

# Save enhanced data function
def save_enhanced_data(gdf, file_path):
    # Convert Timestamps to strings to avoid JSON serialization issues
    for col in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(str)
    gdf.to_file(file_path, driver='GeoJSON')

# Create map function
def create_map(data_gdf=None, draw_enabled=False):
    m = folium.Map(location=[0, 0], zoom_start=2)

    if data_gdf is not None and not data_gdf.empty:
        bounds = data_gdf.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        m.location = center
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        folium.GeoJson(data_gdf).add_to(m)

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
        "Out of the Study Area": ["Major Cities", "Main Roads", "Airports", "Ports", "National Grid", "Substations", "Night Time Lights"],
        "Within the Study Area": ["Buildings", "Points of Interest", "Access Status", "Relative Wealth Index", "Roads", "Elevation", "Crops and Biomass Potential", "Water Bodies and Hydro Potential", "Solar Potential", "Wind Potential", "Landcover", "Available Land for Infrastructure"]
    }

    selected_section = st.sidebar.radio("Section", list(sections.keys()))
    selected_subpage = st.sidebar.radio("Subpage", sections[selected_section])

    # Load and display the selected dataset
    data_gdf = load_data(DATA_PATHS[selected_subpage.lower().replace(' ', '_')])
    
    # Create map
    m = create_map(data_gdf, draw_enabled=True)
    map_output = st_folium(m, width=1400, height=800)

    # Save enhanced data if any
    if map_output and 'last_active_drawing' in map_output and map_output['last_active_drawing']:
        drawn_data = map_output['last_active_drawing']['geometry']
        drawn_gdf = gpd.GeoDataFrame([drawn_data], columns=['geometry'], crs='EPSG:4326')
        enhanced_gdf = pd.concat([data_gdf, drawn_gdf], ignore_index=True)
        save_enhanced_data(enhanced_gdf, DATA_PATHS[selected_subpage.lower().replace(' ', '_')])
        st.success(f"Enhanced data saved for {selected_subpage}")

if __name__ == "__main__":
    show()


def show():
    st.title("Data Retrieve")

    datasets = [
        "Buildings",
        "Roads",
        "Points of Interest",
        "Water Bodies",
        "Major Cities",
        "Airports",
        "Ports",
        "Power Lines",
        "Transformers and Substations",
        "Elevation",
        "Solar Potential",
        "Wind Potential",
        "Satellite",
        "Nighttime Lights",
        "Population"
    ]

    selected_datasets = st.multiselect("Select datasets to download", datasets, default=datasets[-1])

    if st.button("Retrieve Data"):
        st.write("Downloading data...")

        clear_output_directories()
        
        with open('data/input/selected_area.geojson') as f:
            selected_area = json.load(f)

        gdf = gpd.GeoDataFrame.from_features(selected_area["features"])
        polygon = gdf.geometry.unary_union

        polygon_gdf = gpd.GeoDataFrame.from_features(selected_area["features"])
        polygon_gdf = polygon_gdf.set_crs(epsg=4326)
        projected_polygon = polygon_gdf.to_crs(epsg=3857)
        buffer_polygon = projected_polygon.geometry.buffer(200000)
        buffer_gdf = gpd.GeoDataFrame(geometry=buffer_polygon, crs=projected_polygon.crs)
        buffer_gdf = buffer_gdf.to_crs(epsg=4326)
        
        buffer_polygon = buffer_gdf.geometry.unary_union
            
        initialize_earth_engine()

        osm_buildings_file = 'data/output/buildings/osm_buildings.geojson'
        google_buildings_file = 'data/output/buildings/google_buildings.geojson'
        combined_buildings_file = 'data/output/buildings/combined_buildings.geojson'
        roads_file = 'data/output/roads/osm_roads.geojson'
        pois_file = 'data/output/poi/osm_pois.geojson'
        water_bodies_file = 'data/output/water_bodies/osm_water_bodies.geojson'
        cities_file = 'data/output/cities/osm_cities.geojson'
        airports_file = 'data/output/airports/osm_airports.geojson'
        ports_file = 'data/output/ports/osm_ports.geojson'
        power_lines_file = 'data/output/power_lines/osm_power_lines.geojson'
        substations_file = 'data/output/substations/osm_substations.geojson'

        os.makedirs('data/output/buildings', exist_ok=True)
        os.makedirs('data/output/roads', exist_ok=True)
        os.makedirs('data/output/poi', exist_ok=True)
        os.makedirs('data/output/water_bodies', exist_ok=True)
        os.makedirs('data/output/cities', exist_ok=True)
        os.makedirs('data/output/airports', exist_ok=True)
        os.makedirs('data/output/ports', exist_ok=True)
        os.makedirs('data/output/power_lines', exist_ok=True)
        os.makedirs('data/output/substations', exist_ok=True)

        progress = st.progress(0)
        status_text = st.empty()

        try:
            if "Buildings" in selected_datasets:
                status_text.text("Downloading OSM buildings data...")
                osm_buildings_path = download_osm_data(polygon, {'building': True}, osm_buildings_file)
                progress.progress(0.1)

                status_text.text("Downloading Google buildings data...")
                google_buildings_path = download_google_buildings(polygon, google_buildings_file)
                progress.progress(0.2)

                if osm_buildings_path and google_buildings_path:
                    status_text.text("Combining buildings data...")
                    with open(google_buildings_path) as f:
                        google_buildings_geojson = json.load(f)
                    osm_buildings = gpd.read_file(osm_buildings_path)
                    combined_buildings = create_combined_buildings_layer(osm_buildings, google_buildings_geojson)
                    combined_buildings.to_file(combined_buildings_file, driver='GeoJSON')
                    st.write(f"{len(combined_buildings)} buildings in the combined dataset")
                else:
                    st.write("Skipping buildings combination due to missing data.")
                progress.progress(0.3)

            if "Roads" in selected_datasets:
                status_text.text("Downloading OSM roads data...")
                roads_path = download_osm_data(buffer_polygon, {'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']}, roads_file)
                progress.progress(0.4)

            if "Points of Interest" in selected_datasets:
                status_text.text("Downloading OSM points of interest data...")
                pois_path = download_osm_data(polygon, {'amenity': True}, pois_file)
                progress.progress(0.5)

            if "Water Bodies" in selected_datasets:
                status_text.text("Downloading OSM water bodies data...")
                water_bodies_path = download_osm_data(polygon, {'natural': 'water'}, water_bodies_file)
                progress.progress(0.6)

            if "Major Cities" in selected_datasets:
                status_text.text("Downloading OSM major cities data...")
                cities_path = download_osm_data(buffer_polygon, {'place': 'city'}, cities_file)
                progress.progress(0.7)

            if "Airports" in selected_datasets:
                status_text.text("Downloading OSM airports data...")
                airports_path = download_osm_data(buffer_polygon, {'aeroway': 'aerodrome'}, airports_file)
                progress.progress(0.75)

            if "Ports" in selected_datasets:
                status_text.text("Downloading OSM ports data...")
                ports_path = download_osm_data(buffer_polygon, {'amenity': 'port'}, ports_file)
                progress.progress(0.8)

            if "Power Lines" in selected_datasets:
                status_text.text("Downloading OSM power lines data...")
                power_lines_path = download_osm_data(buffer_polygon, {'power': 'line'}, power_lines_file)
                progress.progress(0.85)

            if "Transformers and Substations" in selected_datasets:
                status_text.text("Downloading OSM transformers and substations data...")
                substations_path = download_osm_data(buffer_polygon, {'power': ['transformer', 'substation']}, substations_file)
                progress.progress(0.9)
                             
            if "Elevation" in selected_datasets:
                status_text.text("Downloading elevation data...")
                elevation_file = 'data/output/elevation/image_original.tif'
                os.makedirs('data/output/elevation', exist_ok=True)
                elevation_path = download_elevation_data(polygon, elevation_file)

                if elevation_path:
                    st.write("Elevation data downloaded to the selected area.")
                progress.progress(0.95)

            if "Solar Potential" in selected_datasets:
                status_text.text("Downloading solar data...")
                solar_file = 'data/output/solar/solar_data.tif'
                os.makedirs('data/output/solar', exist_ok=True)
                solar_path = download_solar_data(polygon, solar_file)

                if solar_path:
                    st.write("Solar data downloaded for the selected area.")
                progress.progress(0.95)

            if "Wind Potential" in selected_datasets:
                status_text.text("Downloading wind data...")
                wind_file = 'data/output/wind/wind_data.tif'
                os.makedirs('data/output/wind', exist_ok=True)
                wind_path = download_wind_data(polygon, wind_file)

                if wind_path:
                    st.write("Wind data downloaded for the selected area.")
                progress.progress(0.95)

            if "Satellite" in selected_datasets:
                status_text.text("Downloading satellite data...")
                satellite_file = 'data/output/satellite/satellite_image.tif'
                os.makedirs('data/output/satellite', exist_ok=True)
                download_ee_image('COPERNICUS/S2_SR_HARMONIZED', ['B4', 'B3', 'B2'], polygon, satellite_file, scale=30, dateMin='2020-04-01', dateMax='2020-04-30')
                st.write("Satellite data downloaded for the selected area.")
                progress.progress(0.9)

            if "Nighttime Lights" in selected_datasets:
                status_text.text("Downloading nighttime lights data...")
                nighttime_lights_file = 'data/output/nighttime_lights/nighttime_lights.tif'
                clipped_nighttime_lights_file = 'data/output/nighttime_lights/clipped_nighttime_lights.tif'
                os.makedirs('data/output/nighttime_lights', exist_ok=True)
                nighttime_lights_path = download_nighttime_lights_mpc(polygon, nighttime_lights_file, clipped_nighttime_lights_file)

                if nighttime_lights_path:
                    st.write("Nighttime lights data downloaded and clipped to the selected area.")
                progress.progress(0.95)
               
            if "Population" in selected_datasets:
                status_text.text("Downloading population data...")
                geojson_path = 'data/input/selected_area.geojson'
                population_file = 'data/output/population/age_structure_output.csv'
                os.makedirs('data/output/population', exist_ok=True)
                worldpop.download_worldpop_age_structure(geojson_path, 2020, population_file)
                st.write("Population data downloaded for the selected area.")
                progress.progress(1.0)

    
            zip_files = [
                file_path for file_path in [
                    osm_buildings_file, google_buildings_file, combined_buildings_file, 
                    roads_file, pois_file, water_bodies_file, cities_file,
                    airports_file, ports_file, power_lines_file, substations_file,
                    'data/output/elevation/image_original.tif',  # Include elevation file
                    'data/output/solar/solar_data.tif',          # Include solar data file
                    'data/output/wind/wind_data.tif',            # Include wind data file
                    'data/output/satellite/satellite_image.tif', # Include satellite data file
                    'data/output/nighttime_lights/clipped_nighttime_lights.tif', # Include nighttime lights file
                    'data/output/population/age_structure_output.csv' # Include population data file
                ] if file_path and os.path.exists(file_path)
            ]

            status_text.text("Zipping results...")
            zip_results(zip_files, 'data/output/results.zip')
            progress.progress(1.0)

            st.success("Data download complete. You can now proceed to the next section.")
            with open('data/output/results.zip', 'rb') as f:
                st.download_button('Download All Results', f, file_name='results.zip')

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    show()
