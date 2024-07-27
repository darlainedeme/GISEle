import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
import requests
import tempfile
import ee
from geopy.geocoders import Nominatim
from folium.plugins import Draw, Fullscreen, MeasureControl, MarkerCluster
import osmnx as ox
from shapely.geometry import mapping
import pandas as pd

# Initialize Earth Engine
def initialize_earth_engine():
    json_data = st.secrets["json_data"]
    json_object = json.loads(json_data, strict=False)
    service_account = json_object['client_email']
    credentials = ee.ServiceAccountCredentials(service_account, key_data=json_data)
    ee.Initialize(credentials)

initialize_earth_engine()

# Initialize the app
st.set_page_config(layout="wide")
st.title("Local GISEle")

# Define navigation
page = st.sidebar.radio("Navigation", ["Home", "Area Selection", "Analysis"], key="main_nav")

# Functions to create and manipulate map
def create_map(latitude, longitude, geojson_data=None, combined_buildings=None, osm_roads=None, osm_pois=None, missing_layers=None):
    m = folium.Map(location=[latitude, longitude], zoom_start=15)
    add_map_tiles(m)
    add_geojson_data(m, geojson_data)
    add_combined_buildings(m, combined_buildings)
    add_osm_data(m, osm_roads, osm_pois)
    add_plugins(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=1450, height=800)
    if missing_layers:
        st.write("The following layers weren't possible to obtain for the selected area:")
        for layer in missing_layers:
            st.write(f"- {layer}")

def add_map_tiles(m):
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

def add_geojson_data(m, geojson_data):
    if geojson_data:
        folium.GeoJson(geojson_data, name="Uploaded GeoJSON").add_to(m)

def add_combined_buildings(m, combined_buildings):
    if combined_buildings is not None:
        style_function = lambda x: {
            'fillColor': 'green' if x['properties']['source'] == 'osm' else 'blue',
            'color': 'green' if x['properties']['source'] == 'osm' else 'blue',
            'weight': 1,
        }
        folium.GeoJson(combined_buildings, name="Combined Buildings", style_function=style_function).add_to(m)
        marker_cluster = MarkerCluster(name='Combined Buildings Clusters').add_to(m)
        for _, row in combined_buildings.iterrows():
            folium.Marker(location=[row.geometry.centroid.y, row.geometry.centroid.x]).add_to(marker_cluster)

def add_osm_data(m, osm_roads, osm_pois):
    if osm_roads is not None:
        folium.GeoJson(osm_roads.to_json(), name='OSM Roads', style_function=lambda x: {
            'fillColor': 'orange',
            'color': 'orange',
            'weight': 1,
        }).add_to(m)
    if osm_pois is not None:
        folium.GeoJson(osm_pois.to_json(), name='OSM Points of Interest', style_function=lambda x: {
            'fillColor': 'red',
            'color': 'red',
            'weight': 1,
        }).add_to(m)

def add_plugins(m):
    Draw(export=True, filename='data.geojson', position='topleft').add_to(m)
    Fullscreen(position='topleft').add_to(m)
    MeasureControl(position='bottomleft').add_to(m)

def uploaded_file_to_gdf(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as temp_file:
        temp_file.write(data.getvalue())
        temp_filepath = temp_file.name
    gdf = gpd.read_file(temp_filepath)
    return gdf

def handle_address_input():
    geolocator = Nominatim(user_agent="example app")
    address = st.sidebar.text_input('Enter your address:', value='B12 Bovisa', key="address_input")
    if address:
        with st.spinner('Fetching location...'):
            location = geolocator.geocode(address)
            if location:
                create_map(location.latitude, location.longitude)

def handle_coordinates_input():
    latitude = st.sidebar.text_input('Latitude:', value=45.5065, key="latitude_input")
    longitude = st.sidebar.text_input('Longitude:', value=9.1598, key="longitude_input")
    if latitude and longitude:
        create_map(float(latitude), float(longitude))

def handle_file_upload():
    data = st.sidebar.file_uploader("Upload a GeoJSON file", type=["geojson"], key="file_uploader")
    if data:
        gdf = uploaded_file_to_gdf(data)
        if not gdf.empty and not gdf.is_empty.any():
            geojson_data = gdf.to_json()
            fetch_and_create_layers(gdf, geojson_data)

def fetch_and_create_layers(gdf, geojson_data):
    coords = gdf.geometry.total_bounds
    geom = ee.Geometry.Rectangle([coords[0], coords[1], coords[2], coords[3]])

    buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons').filter(ee.Filter.intersects('.geo', geom))
    download_url = buildings.getDownloadURL('geojson')
    response = requests.get(download_url)
    google_buildings = response.json()

    polygon = gdf.unary_union
    osm_buildings, osm_roads, osm_pois, missing_layers = fetch_osm_data(polygon)

    combined_buildings = create_combined_buildings_layer(osm_buildings, google_buildings)
    gdf = gdf.to_crs(epsg=4326)
    centroid = gdf.geometry.centroid.iloc[0]

    create_map(centroid.y, centroid.x, geojson_data, combined_buildings, osm_roads, osm_pois, missing_layers)

def fetch_osm_data(polygon):
    missing_layers = []
    try:
        osm_buildings = ox.features_from_polygon(polygon, tags={'building': True})
    except:
        osm_buildings = None
        missing_layers.append('buildings')

    try:
        osm_roads = ox.features_from_polygon(polygon, tags={'highway': True})
    except:
        osm_roads = None
        missing_layers.append('roads')

    try:
        osm_pois = ox.features_from_polygon(polygon, tags={'amenity': True})
    except:
        osm_pois = None
        missing_layers.append('points of interest')

    return osm_buildings, osm_roads, osm_pois, missing_layers

if page == "Home":
    st.write("Welcome to Local GISEle")
    st.write("Use the sidebar to navigate to different sections of the app.")
elif page == "Area Selection":
    which_modes = ['By address', 'By coordinates', 'Upload file']
    which_mode = st.sidebar.selectbox('Select mode', which_modes, index=2, key="mode_select")

    if which_mode == 'By address':
        handle_address_input()
    elif which_mode == 'By coordinates':
        handle_coordinates_input()
    elif which_mode == 'Upload file':
        handle_file_upload()
elif page == "Analysis":
    st.write("Analysis page under construction")

st.sidebar.title("About")
st.sidebar.info(
    """
    Web App URL: [https://gisele.streamlit.app/](https://gisele.streamlit.app/)
    GitHub repository: [https://github.com/darlainedeme/GISEle](https://github.com/darlainedeme/GISEle)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Darlain Edeme: [http://www.e4g.polimi.it/](http://www.e4g.polimi.it/)
    [GitHub](https://github.com/darlainedeme) | [Twitter](https://twitter.com/darlainedeme) | [LinkedIn](https://www.linkedin.com/in/darlain-edeme)
    """
)
