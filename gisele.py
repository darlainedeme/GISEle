import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import json
import requests
import tempfile
import ee
from geopy.geocoders import Nominatim

# Initialize Earth Engine
def initialize_earth_engine():
    json_data = st.secrets["json_data"]
    json_object = json.loads(json_data, strict=False)
    service_account = json_object['client_email']
    credentials = ee.ServiceAccountCredentials(service_account, key_data=json_data)
    ee.Initialize(credentials)

initialize_earth_engine()

<<<<<<< HEAD
# Load the world administrative boundaries GeoJSON
world = gpd.read_file("data/world-administrative-boundaries.geojson")

def create_map(latitude, longitude, geojson_data, buildings_data):
    m = folium.Map(location=[latitude, longitude], zoom_start=12)
    
    # Add GeoDataFrame to the map
    if geojson_data:
        folium.GeoJson(geojson_data, name="Uploaded GeoJSON").add_to(m)

    # Add new dataset buildings data to the map
    if buildings_data:
        folium.GeoJson(buildings_data, name="New Buildings Dataset", style_function=lambda x: {
            'fillColor': 'green',
            'color': 'green',
            'weight': 1,
        }).add_to(m)
=======
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
<<<<<<< HEAD

def uploaded_file_to_gdf(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as temp_file:
        temp_file.write(data.getvalue())
        temp_filepath = temp_file.name

    gdf = gpd.read_file(temp_filepath)
    return gdf

def get_country_iso(gdf, world_data):
    gdf = gdf.to_crs(epsg=3857)  # Reproject to a projected CRS
    centroid = gdf.geometry.centroid.iloc[0]
    point = gpd.GeoDataFrame(geometry=[centroid], crs=gdf.crs)
    world_data = world_data.to_crs(epsg=3857)  # Ensure world data is in the same CRS
    country = gpd.sjoin(point, world_data, how="left", predicate="intersects")
    return country.iloc[0]['iso3'] if not country.empty else None

if page == "Home":
    st.write("Welcome to Local GISEle")
    st.write("Use the sidebar to navigate to different sections of the app.")
elif page == "Area Selection":
    which_modes = ['By address', 'By coordinates', 'Upload file']
    which_mode = st.sidebar.selectbox('Select mode', which_modes, index=2, key="mode_select")

    if which_mode == 'By address':  
        geolocator = Nominatim(user_agent="example app")
        address = st.sidebar.text_input('Enter your address:', value='B12 Bovisa', key="address_input") 

        try:
            location = geolocator.geocode(address)
            if location:
                create_map(location.latitude, location.longitude, None, None)
            else:
                st.error("Could not geocode the address.")
        except Exception as e:
            st.error(f"Error fetching location: {e}")
    elif which_mode == 'By coordinates':  
        latitude = st.sidebar.text_input('Latitude:', value=45.5065, key="latitude_input") 
        longitude = st.sidebar.text_input('Longitude:', value=9.1598, key="longitude_input") 
        
        try:
            if latitude and longitude:
                create_map(float(latitude), float(longitude), None, None)
            else:
                st.error("Please provide both latitude and longitude.")
        except Exception as e:
            st.error(f"Error creating map: {e}")
    elif which_mode == 'Upload file':
        data = st.sidebar.file_uploader("Upload a GeoJSON file", type=["geojson"], key="file_uploader")

        if data:
            try:
                gdf = uploaded_file_to_gdf(data)
                if gdf.empty or gdf.isnull().values.any():
                    st.error("Uploaded GeoJSON file is empty or contains null values.")
                else:
                    geojson_data = gdf.to_json()
                    iso_code = get_country_iso(gdf, world)
                    
                    if iso_code:
                        # Fetch and add new dataset buildings data
                        buildings = ee.FeatureCollection(f"projects/sat-io/open-datasets/VIDA_COMBINED/{iso_code}")
                        
                        download_url = buildings.getDownloadURL('geojson')
                        response = requests.get(download_url)
                        buildings_data = response.json()
                        
                        # Reproject GeoDataFrame to geographic CRS for mapping and display
                        gdf = gdf.to_crs(epsg=4326)
                        st.dataframe(gdf.drop(columns='geometry'))  # Display the GeoDataFrame as a table without the geometry column
                        
                        centroid = gdf.geometry.centroid.iloc[0]
                        create_map(centroid.y, centroid.x, geojson_data, buildings_data)
                    else:
                        st.error("Unable to determine the country for the provided location.")
            except KeyError as e:
                st.error(f"Error processing file: {e}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
elif page == "Analysis":
    st.write("Analysis page under construction")
=======
else:
    st.write("Please upload a GeoJSON file to display the map.")
>>>>>>> parent of 651e4c9 (google only)

st.sidebar.title("About")
st.sidebar.info(
    """
    This app allows you to upload a GeoJSON file and visualize it on a map,
    along with buildings data from Google Earth Engine.
    """
)
