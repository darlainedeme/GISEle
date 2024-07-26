import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import json
import requests
import tempfile
import ee

# Initialize Earth Engine
@st.cache_resource
def initialize_earth_engine():
    json_data = st.secrets["json_data"]
    json_object = json.loads(json_data, strict=False)
    service_account = json_object['client_email']
    credentials = ee.ServiceAccountCredentials(service_account, key_data=json_data)
    ee.Initialize(credentials)

initialize_earth_engine()

st.set_page_config(layout="wide")
st.title("Simple GIS App")

# File uploader for GeoJSON
uploaded_file = st.file_uploader("Upload a GeoJSON file", type=["geojson"])

if uploaded_file:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_filepath = temp_file.name

    # Load the GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(temp_filepath)

    # Convert GeoDataFrame to JSON
    geojson_data = gdf.to_json()

    # Create a Folium map centered on the centroid of the GeoDataFrame
    centroid = gdf.geometry.centroid.iloc[0]
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12)

    # Add the GeoDataFrame to the map
    folium.GeoJson(geojson_data, name="Uploaded GeoJSON").add_to(m)

    # Fetch and add Google Buildings data to the map
    coords = gdf.geometry.iloc[0].bounds
    geom = ee.Geometry.Rectangle([coords[0], coords[1], coords[2], coords[3]])
    
    buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
        .filter(ee.Filter.intersects('.geo', geom))
    
    download_url = buildings.getDownloadURL('geojson')
    response = requests.get(download_url)
    buildings_data = response.json()

    folium.GeoJson(buildings_data, name="Google Buildings", style_function=lambda x: {
        'fillColor': 'green',
        'color': 'green',
        'weight': 1,
    }).add_to(m)

    folium.LayerControl().add_to(m)

    # Display the map
    folium_static(m, width=800, height=600)
else:
    st.write("Please upload a GeoJSON file to display the map.")

st.sidebar.title("About")
st.sidebar.info(
    """
    This app allows you to upload a GeoJSON file and visualize it on a map,
    along with buildings data from Google Earth Engine.
    """
)
