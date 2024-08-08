import streamlit as st
import geopandas as gpd
import json
import os

from scripts.data_retrieve_scripts._data_utils import initialize_earth_engine, zip_results, clear_output_directories
from scripts.data_retrieve_scripts.buildings import *
from scripts.data_retrieve_scripts.airports import *
from scripts.data_retrieve_scripts.cities import *
from scripts.data_retrieve_scripts.roads import *
from scripts.data_retrieve_scripts.poi import *
from scripts.data_retrieve_scripts.water_bodies import *
from scripts.data_retrieve_scripts.ports import *
from scripts.data_retrieve_scripts.power_lines import *
from scripts.data_retrieve_scripts.substations import *
from scripts.data_retrieve_scripts.population import *
from scripts.data_retrieve_scripts.solar import *
from scripts.data_retrieve_scripts.wind import *
from scripts.data_retrieve_scripts.elevation import *
'''
from scripts.data_retrieve_scripts.satellite import *
from scripts.data_retrieve_scripts.night_time_lights import *
add worldpop raster
'''

def show():
    datasets = sorted([
        "Airports",
        "Buildings",
        "Elevation",
        "Major Cities",
        "Nighttime Lights",
        "Points of Interest",
        "Ports",
        "Population",
        "Power Lines",
        "Roads",
        "Satellite",
        "Solar Potential",
        "Transformers and Substations",
        "Water Bodies",
        "Wind Potential"
    ])

    selected_datasets = st.multiselect("Select datasets to download", datasets, default=[datasets[0]])

    if st.button("Retrieve Data"):
        st.write("Downloading data...")

        clear_output_directories()

        selected_area_path = os.path.join('data', 'input', 'selected_area.geojson')
        gdf = gpd.read_file(selected_area_path)
        polygon = gdf.geometry.unary_union

        polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=gdf.crs)
        projected_polygon = polygon_gdf.to_crs(epsg=3857)
        buffer_polygon = projected_polygon.geometry.buffer(20000)
        buffer_gdf = gpd.GeoDataFrame(geometry=buffer_polygon, crs=projected_polygon.crs)
        buffer_gdf = buffer_gdf.to_crs(epsg=4326)

        buffer_polygon = buffer_gdf.geometry.unary_union

        initialize_earth_engine()

        progress = st.progress(0)
        status_text = st.empty()

        try:
            if "Buildings" in selected_datasets:
                status_text.text("Downloading buildings data...")
                download_buildings_data(polygon)
                progress.progress(0.1)
                st.write("Buildings data downloaded.")

            if "Roads" in selected_datasets:
                status_text.text("Downloading roads data...")
                download_roads_data(buffer_polygon)
                progress.progress(0.2)
                st.write("Roads data downloaded.")

            if "Points of Interest" in selected_datasets:
                status_text.text("Downloading points of interest data...")
                download_poi_data(polygon)
                progress.progress(0.3)
                st.write("Points of interest data downloaded.")

            if "Water Bodies" in selected_datasets:
                status_text.text("Downloading water bodies data...")
                download_water_bodies_data(polygon)
                progress.progress(0.4)
                st.write("Water bodies data downloaded.")

            if "Major Cities" in selected_datasets:
                status_text.text("Downloading major cities data...")
                download_cities_data(buffer_polygon)
                progress.progress(0.5)
                st.write("Major cities data downloaded.")

            if "Airports" in selected_datasets:
                status_text.text("Downloading airports data...")
                download_airports_data(buffer_polygon)
                progress.progress(0.6)
                st.write("Airports data downloaded.")

            if "Ports" in selected_datasets:
                status_text.text("Downloading ports data...")
                download_ports_data(buffer_polygon)
                progress.progress(0.7)
                st.write("Ports data downloaded.")

            if "Power Lines" in selected_datasets:
                status_text.text("Downloading power lines data...")
                download_power_lines_data(buffer_polygon)
                progress.progress(0.75)
                st.write("Power lines data downloaded.")

            if "Transformers and Substations" in selected_datasets:
                status_text.text("Downloading transformers and substations data...")
                download_substations_data(buffer_polygon)
                progress.progress(0.8)
                st.write("Transformers and substations data downloaded.")

            if "Elevation" in selected_datasets:
                status_text.text("Downloading elevation data...")
                download_elevation_data(polygon)
                progress.progress(0.85)
                st.write("Elevation data downloaded.")

            if "Solar Potential" in selected_datasets:
                status_text.text("Downloading solar data...")
                download_solar_data(polygon)
                progress.progress(0.9)
                st.write("Solar data downloaded.")

            if "Wind Potential" in selected_datasets:
                status_text.text("Downloading wind data...")
                download_wind_data(polygon)
                progress.progress(0.9)
                st.write("Wind data downloaded.")

            if "Satellite" in selected_datasets:
                status_text.text("Downloading satellite data...")
                download_satellite_data(polygon)
                progress.progress(0.95)
                st.write("Satellite data downloaded.")

            if "Nighttime Lights" in selected_datasets:
                status_text.text("Downloading nighttime lights data...")
                download_nighttime_lights_data(polygon)
                progress.progress(1)
                st.write("Nighttime lights data downloaded.")

            if "Population" in selected_datasets:
                status_text.text("Downloading population data...")
                population_file = os.path.join('data', '2_downloaded_input_data', 'population', 'age_structure_output.csv')
                output_raster = os.path.join('data', '2_downloaded_input_data', 'population', 'population.tiff')
                download_population_data(selected_area_path, 2020, population_file)
                progress.progress(1)
                st.write("Population data downloaded.")

            status_text.text("Zipping results...")
            zip_results(os.path.join('data', '2_downloaded_input_data'), os.path.join('data', 'output', 'results.zip'))
            progress.progress(1.0)

            st.success("Data download complete. You can now proceed to the next section.")
            with open(os.path.join('data', 'output', 'results.zip'), 'rb') as f:
                st.download_button('Download All Results', f, file_name='results.zip')

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    show()
