import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import json
import requests
import tempfile
import ee
from geopy.geocoders import Nominatim
from folium.plugins import Draw, Fullscreen, MeasureControl, MarkerCluster

# Initialize Earth Engine
@st.cache_resource
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

def create_map(latitude, longitude, geojson_data, buildings_data):
    m = folium.Map(location=[latitude, longitude], zoom_start=12)
    
    # Add GeoDataFrame to the map
    if geojson_data:
        folium.GeoJson(geojson_data, name="Uploaded GeoJSON").add_to(m)

    # Add Google Buildings data to the map as a selectable layer
    if buildings_data:
        folium.GeoJson(buildings_data, name="Google Buildings", style_function=lambda x: {
            'fillColor': 'green',
            'color': 'green',
            'weight': 1,
        }).add_to(m)

        # Add MarkerCluster for Google Buildings
        marker_cluster = MarkerCluster(name='Google Buildings Clusters').add_to(m)
        for feature in buildings_data['features']:
            coords = feature['geometry']['coordinates']
            folium.Marker(location=[coords[1], coords[0]]).add_to(marker_cluster)

    # Add drawing and fullscreen plugins
    Draw(export=True, filename='data.geojson', position='topleft').add_to(m)
    Fullscreen(position='topleft').add_to(m)
    MeasureControl(position='bottomleft').add_to(m)
    
    folium.LayerControl().add_to(m)

    # Display the map
    folium_static(m, width=800, height=600)

def uploaded_file_to_gdf(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as temp_file:
        temp_file.write(data.getvalue())
        temp_filepath = temp_file.name

    gdf = gpd.read_file(temp_filepath)
    return gdf

if page == "Home":
    st.write("Welcome to Local GISEle")
    st.write("Use the sidebar to navigate to different sections of the app.")
elif page == "Area Selection":
    which_modes = ['By address', 'By coordinates', 'Upload file']
    which_mode = st.sidebar.selectbox('Select mode', which_modes, index=2, key="mode_select")

    if which_mode == 'By address':  
        geolocator = Nominatim(user_agent="example app")
        address = st.sidebar.text_input('Enter your address:', value='B12 Bovisa', key="address_input") 

        if address:
            try:
                with st.spinner('Fetching location...'):
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
        
        if latitude and longitude:
            try:
                with st.spinner('Creating map...'):
                    create_map(float(latitude), float(longitude), None, None)
            except Exception as e:
                st.error(f"Error creating map: {e}")
        else:
            st.error("Please provide both latitude and longitude.")
    elif which_mode == 'Upload file':
        data = st.sidebar.file_uploader("Upload a GeoJSON file", type=["geojson"], key="file_uploader")

        if data:
            try:
                st.info("Uploading file...")
                gdf = uploaded_file_to_gdf(data)
                geojson_data = gdf.to_json()
                
                st.info("Fetching building data...")
                coords = gdf.geometry.total_bounds
                geom = ee.Geometry.Rectangle([coords[0], coords[1], coords[2], coords[3]])
                
                buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
                    .filter(ee.Filter.intersects('.geo', geom))
                
                download_url = buildings.getDownloadURL('geojson')
                response = requests.get(download_url)
                buildings_data = response.json()

                st.info("Creating map...")
                centroid = gdf.geometry.centroid.iloc[0]
                create_map(centroid.y, centroid.x, geojson_data, buildings_data)
                st.success("Map created successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")
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
