import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import json
import requests
import tempfile
import ee
from shapely.geometry import Point

# Initialize Earth Engine
def initialize_earth_engine():
    json_data = st.secrets["json_data"]
    json_object = json.loads(json_data, strict=False)
    service_account = json_object['client_email']
    credentials = ee.ServiceAccountCredentials(service_account, key_data=json_data)
    ee.Initialize(credentials)

# Initialize the app
st.set_page_config(layout="wide")
st.title("Local GISEle")

# Define navigation
page = st.sidebar.radio("Navigation", ["Home", "Area Selection", "Analysis"], key="main_nav")

# Call to initialize Earth Engine
initialize_earth_engine()

# Download and load the naturalearth_lowres dataset
@st.cache_data
def load_world_data():
    url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip"
    world = gpd.read_file(url)
    return world

world = load_world_data()

def create_map(latitude, longitude, geojson_data, buildings_data):
    m = folium.Map(location=[latitude, longitude], zoom_start=12)
    
    # Add GeoDataFrame to the map
    folium.GeoJson(geojson_data, name="Uploaded GeoJSON").add_to(m)

    # Add new dataset buildings data to the map
    folium.GeoJson(buildings_data, name="New Buildings Dataset", style_function=lambda x: {
        'fillColor': 'green',
        'color': 'green',
        'weight': 1,
    }).add_to(m)

    folium.LayerControl().add_to(m)

    # Display the map
    folium_static(m, width=800, height=600)

def uploaded_file_to_gdf(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as temp_file:
        temp_file.write(data.getvalue())
        temp_filepath = temp_file.name

    gdf = gpd.read_file(temp_filepath)
    return gdf

def get_country_iso(lat, lon, world_data):
    point = Point(lon, lat)
    country = world_data[world_data.contains(point)]
    return country.iloc[0]['ADM0_A3'] if not country.empty else None

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
            if address:
                # Create map with dummy data as placeholder
                create_map(location.latitude, location.longitude, None, None)
        except Exception as e:
            st.error(f"Error fetching location: {e}")
    elif which_mode == 'By coordinates':  
        latitude = st.sidebar.text_input('Latitude:', value=45.5065, key="latitude_input") 
        longitude = st.sidebar.text_input('Longitude:', value=9.1598, key="longitude_input") 
        
        try:
            if latitude and longitude:
                # Create map with dummy data as placeholder
                create_map(float(latitude), float(longitude), None, None)
        except Exception as e:
            st.error(f"Error creating map: {e}")
    elif which_mode == 'Upload file':
        data = st.sidebar.file_uploader("Upload a GeoJSON file", type=["geojson"], key="file_uploader")

        if data:
            try:
                gdf = uploaded_file_to_gdf(data)
                geojson_data = gdf.to_json()
                
                # Get the centroid of the GeoDataFrame to determine the country ISO code
                centroid = gdf.geometry.centroid.iloc[0]
                iso_code = get_country_iso(centroid.y, centroid.x, world)
                
                if iso_code:
                    # Fetch and add new dataset buildings data
                    buildings = ee.FeatureCollection(f"projects/sat-io/open-datasets/VIDA_COMBINED/{iso_code}")
                    
                    download_url = buildings.getDownloadURL('geojson')
                    response = requests.get(download_url)
                    buildings_data = response.json()
                    
                    # Create map with uploaded GeoJSON and new dataset buildings data
                    create_map(centroid.y, centroid.x, geojson_data, buildings_data)
                else:
                    st.error("Unable to determine the country for the provided location.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
elif page == "Analysis":
    st.write("Analysis page under construction")

st.sidebar.title("About")
st.sidebar.info(
    """
    Web App URL: https://darlainedeme-local-gisele-local-gisele-bx888v.streamlit.app/
    GitHub repository: https://github.com/darlainedeme/local_gisele
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Darlain Edeme: http://www.e4g.polimi.it/
    [GitHub](https://github.com/darlainedeme) | [Twitter](https://twitter.com/darlainedeme) | [LinkedIn](https://www.linkedin.com/in/darlain-edeme')
    """
)
