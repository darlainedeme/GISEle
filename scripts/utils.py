import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
import tempfile
import ee
from folium.plugins import MarkerCluster
import streamlit as st
import pandas as pd
import os
import shutil
import zipfile


# Initialize Earth Engine
@st.cache_resource
def initialize_earth_engine():
    # Retrieve json_data from Streamlit secrets
    json_data = st.secrets["json_data"]
    
    # Check if json_data is None
    if json_data is None:
        raise ValueError("json_data is None")
    
    # Convert the AttrDict to a JSON string
    json_data_str = json.dumps(json_data)
    
    # Check if json_data_str is of valid type
    if not isinstance(json_data_str, str):
        raise TypeError(f"json_data must be str, but got {type(json_data_str)}")
    
    # Optionally print the data type and part of the content for debugging
    print(f"json_data_str type: {type(json_data_str)}")
    print(f"json_data_str content: {json_data_str[:100]}...")  # Print the first 100 characters
    
    # Parse the JSON string
    json_object = json.loads(json_data_str, strict=False)
    
    # Extract the service account email from the JSON object
    service_account = json_object['client_email']
    
    # Initialize the Earth Engine service account credentials
    credentials = ee.ServiceAccountCredentials(service_account, key_data=json_data_str)
    ee.Initialize(credentials)

def clear_output_directories():
    output_dirs = [
        'data/output/buildings', 'data/output/roads', 'data/output/poi',
        'data/output/water_bodies', 'data/output/cities', 'data/output/airports',
        'data/output/ports', 'data/output/power_lines', 'data/output/substations',
        'data/output/elevation', 'data/output/solar', 'data/output/wind', 'data/output/satellite', 'data/output/nighttime_lights', 'data/output/population'
    ]
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

def zip_results(files, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file_path in files:
            zipf.write(file_path, os.path.basename(file_path))
            
def create_map(latitude, longitude, geojson_data=None, combined_buildings=None, osm_roads=None, osm_pois=None):
    m = folium.Map(location=[latitude, longitude], zoom_start=15)

    # Add map tiles
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

    # Add the original area of interest
    if geojson_data:
        folium.GeoJson(
            geojson_data,
            name="Original Area",
            style_function=lambda x: {
                'fillColor': 'blue',
                'color': 'blue',
                'weight': 2,
                'fillOpacity': 0.2,
            }
        ).add_to(m)

    # Add combined buildings data to the map
    if combined_buildings is not None:
        combined_buildings_layer = folium.FeatureGroup(name="Combined Buildings").add_to(m)
        style_function = lambda x: {
            'fillColor': 'green' if x['properties']['source'] == 'osm' else 'blue',
            'color': 'green' if x['properties']['source'] == 'osm' else 'blue',
            'weight': 1,
        }
        folium.GeoJson(combined_buildings, name="Combined Buildings", style_function=style_function).add_to(combined_buildings_layer)

        # Add MarkerCluster for combined buildings
        marker_cluster = MarkerCluster(name='Combined Buildings Clusters').add_to(m)
        for _, row in combined_buildings.iterrows():
            folium.Marker(location=[row.geometry.centroid.y, row.geometry.centroid.x]).add_to(marker_cluster)

    # Add OSM Roads data to the map
    if osm_roads is not None:
        folium.GeoJson(osm_roads.to_json(), name='OSM Roads', style_function=lambda x: {
            'fillColor': 'orange',
            'color': 'orange',
            'weight': 1,
        }).add_to(m)

    # Add OSM Points of Interest data to the map
    if osm_pois is not None:
        folium.GeoJson(osm_pois.to_json(), name='OSM Points of Interest', style_function=lambda x: {
            'fillColor': 'red',
            'color': 'red',
            'weight': 1,
        }).add_to(m)

    # Add drawing and fullscreen plugins
    folium.plugins.Draw(export=True, filename='data.geojson', position='topleft').add_to(m)
    folium.plugins.Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen',
                              force_separate_button=False).add_to(m)
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='meters', secondary_length_unit='miles',
                                  primary_area_unit='sqmeters', secondary_area_unit='acres').add_to(m)
    folium.LayerControl().add_to(m)
    
    # Display the map
    st_data = st_folium(m, width=1400, height=800)
    return st_data

def uploaded_file_to_gdf(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as temp_file:
        temp_file.write(data.getvalue())
        temp_filepath = temp_file.name

    gdf = gpd.read_file(temp_filepath)
    return gdf

def create_combined_buildings_layer(osm_buildings, google_buildings_geojson):
    # Ensure both GeoDataFrames are in the same CRS
    osm_buildings = osm_buildings.to_crs(epsg=4326)
    google_buildings = gpd.GeoDataFrame.from_features(google_buildings_geojson["features"]).set_crs(epsg=4326)

    # Label sources
    osm_buildings['source'] = 'osm'
    google_buildings['source'] = 'google'

    # Remove Google buildings that touch OSM buildings
    osm_dissolved = osm_buildings.geometry.union_all()

    # Filter Google buildings that do not intersect with OSM buildings
    filtered_google = google_buildings[~google_buildings.intersects(osm_dissolved)]

    # Combine OSM buildings and filtered Google buildings
    combined_buildings = gpd.GeoDataFrame(pd.concat([osm_buildings, filtered_google], ignore_index=True))

    return combined_buildings
