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
from shapely.geometry import mapping, box
import pandas as pd
import os

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
main_nav = st.sidebar.radio("Navigation", ["Home", "Area Selection", "Data Collection", "Data Analysis"], key="main_nav")

# Initialize Earth Engine
initialize_earth_engine()

# Define file paths
RESULTS_DIR = "results"
BUILDINGS_GEOJSON = os.path.join(RESULTS_DIR, "combined_buildings.geojson")
OSM_BUILDINGS_GEOJSON = os.path.join(RESULTS_DIR, "osm_buildings.geojson")
GOOGLE_BUILDINGS_GEOJSON = os.path.join(RESULTS_DIR, "google_buildings.geojson")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define functions
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
    st_folium(m, width=1400, height=800)  # Wider map

def uploaded_file_to_gdf(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as temp_file:
        temp_file.write(data.getvalue())
        temp_filepath = temp_file.name

    gdf = gpd.read_file(temp_filepath)
    return gdf

def create_combined_buildings_layer(osm_buildings, google_buildings):
    # Ensure both GeoDataFrames are in the same CRS
    osm_buildings = osm_buildings.to_crs(epsg=4326)
    google_buildings = gpd.GeoDataFrame.from_features(google_buildings["features"]).set_crs(epsg=4326)

    # Label sources
    osm_buildings['source'] = 'osm'
    google_buildings['source'] = 'google'

    # Remove Google buildings that touch OSM buildings
    osm_dissolved = osm_buildings.unary_union

    # Filter Google buildings that do not intersect with OSM buildings
    filtered_google = google_buildings[~google_buildings.intersects(osm_dissolved)]

    # Combine OSM buildings and filtered Google buildings
    combined_buildings = gpd.GeoDataFrame(pd.concat([osm_buildings, filtered_google], ignore_index=True))  

    return combined_buildings

# Main app logic
if main_nav == "Home":
    st.write("Welcome to Local GISEle")
    st.write("Use the sidebar to navigate to different sections of the app.")

elif main_nav == "Area Selection":
    which_modes = ['By address', 'By coordinates', 'Upload file']
    which_mode = st.sidebar.selectbox('Select mode', which_modes, index=0, key="mode_select")

    if which_mode == 'By address':  
        geolocator = Nominatim(user_agent="example app")
        address = st.sidebar.text_input('Enter your address:', value='B12 Bovisa', key="address_input") 

        if address:
            try:
                with st.spinner('Fetching location...'):
                    location = geolocator.geocode(address)
                    if location:
                        st.session_state.latitude = location.latitude
                        st.session_state.longitude = location.longitude
                        st.session_state.geojson_data = None
                        st.session_state.combined_buildings = None
                        st.session_state.osm_roads = None
                        st.session_state.osm_pois = None
                        st.session_state.missing_layers = []
                        # Create a map showing only the polygon area
                        create_map(location.latitude, location.longitude)
                    else:
                        st.error("Could not geocode the address.")
            except Exception as e:
                st.error(f"Error fetching location: {e}")

    elif which_mode == 'By coordinates':  
        latitude = st.sidebar.text_input('Latitude:', value='45.5065', key="latitude_input") 
        longitude = st.sidebar.text_input('Longitude:', value='9.1598', key="longitude_input") 
        
        if latitude and longitude:
            try:
                with st.spinner('Creating map...'):
                    st.session_state.latitude = float(latitude)
                    st.session_state.longitude = float(longitude)
                    st.session_state.geojson_data = None
                    st.session_state.combined_buildings = None
                    st.session_state.osm_roads = None
                    st.session_state.osm_pois = None
                    st.session_state.missing_layers = []
                    # Create a map showing only the polygon area
                    create_map(float(latitude), float(longitude))
            except Exception as e:
                st.error(f"Error with coordinates: {e}")

    elif which_mode == 'Upload file':  
        uploaded_file = st.file_uploader("Upload GeoJSON file", type="geojson", key="geojson_uploader")

        if uploaded_file:
            try:
                with st.spinner('Processing file...'):
                    geojson_data = json.load(uploaded_file)
                    gdf = uploaded_file_to_gdf(uploaded_file)
                    
                    if gdf.empty:
                        st.error("Uploaded file is empty or not valid GeoJSON.")
                        
                    centroid = gdf.geometry.unary_union.centroid
                    st.session_state.latitude = centroid.y
                    st.session_state.longitude = centroid.x
                    st.session_state.geojson_data = geojson_data
                    st.session_state.combined_buildings = None
                    st.session_state.osm_roads = None
                    st.session_state.osm_pois = None
                    st.session_state.missing_layers = []

                    # Create a map showing the uploaded file's polygon area
                    create_map(centroid.y, centroid.x, geojson_data)
                    st.success("Map created successfully!")
            except KeyError as e:
                st.error(f"Error processing file: {e}")
            except IndexError as e:
                st.error(f"Error processing file: {e}")
            except Exception as e:
                st.error(f"Error processing file: {e}")

elif main_nav == "Data Collection":
    if 'latitude' not in st.session_state or 'longitude' not in st.session_state:
        st.error("Please select an area in the 'Area Selection' page first.")
    else:
        data_collection_nav = st.sidebar.radio("Data Collection", ["Buildings", "Roads", "Points of Interest"], key="data_collection_nav")

        latitude = st.session_state.latitude
        longitude = st.session_state.longitude
        polygon = gpd.GeoDataFrame(
            {'geometry': [box(longitude - 0.01, latitude - 0.01, longitude + 0.01, latitude + 0.01)]},
            crs='EPSG:4326'
        )

        if data_collection_nav == "Buildings":
            st.write("Data Collection: Buildings")
            if os.path.exists(BUILDINGS_GEOJSON):
                st.info("Loading previously saved combined buildings data...")
                combined_buildings = gpd.read_file(BUILDINGS_GEOJSON)
            else:
                st.info("Fetching building data...")
                try:
                    geom = ee.Geometry.Rectangle([longitude - 0.01, latitude - 0.01, longitude + 0.01, latitude + 0.01])
                    
                    buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
                        .filter(ee.Filter.intersects('.geo', geom))
                    
                    download_url = buildings.getDownloadURL('geojson')
                    response = requests.get(download_url)
                    google_buildings = response.json()
                    
                    st.info("Fetching OSM data...")
                    try:
                        osm_buildings = ox.features_from_polygon(polygon.unary_union, tags={'building': True})
                        osm_buildings_gdf = gpd.GeoDataFrame.from_features(osm_buildings)
                        osm_buildings_gdf['source'] = 'osm'

                        google_buildings_gdf = gpd.GeoDataFrame.from_features(google_buildings["features"])
                        google_buildings_gdf['source'] = 'google'

                        combined_buildings = create_combined_buildings_layer(osm_buildings_gdf, google_buildings_gdf)
                        combined_buildings.to_file(BUILDINGS_GEOJSON, driver='GeoJSON')
                        st.info("Combined buildings data saved.")
                    except Exception as e:
                        st.error(f"Error fetching OSM buildings data: {e}")
                except Exception as e:
                    st.error(f"Error fetching building data: {e}")

            # Display the combined buildings
            create_map(latitude, longitude, combined_buildings=combined_buildings)

        elif data_collection_nav == "Roads":
            st.write("Data Collection: Roads")
            st.info("Fetching OSM roads data...")
            try:
                osm_roads = ox.features_from_polygon(polygon.unary_union, tags={'highway': True})
                osm_roads_gdf = gpd.GeoDataFrame.from_features(osm_roads)
                osm_roads_gdf.to_file('results/osm_roads.geojson', driver='GeoJSON')
                create_map(latitude, longitude, osm_roads=osm_roads_gdf)
            except Exception as e:
                st.error(f"Error fetching OSM roads data: {e}")

        elif data_collection_nav == "Points of Interest":
            st.write("Data Collection: Points of Interest")
            st.info("Fetching OSM points of interest data...")
            try:
                osm_pois = ox.features_from_polygon(polygon.unary_union, tags={'amenity': True})
                osm_pois_gdf = gpd.GeoDataFrame.from_features(osm_pois)
                osm_pois_gdf.to_file('results/osm_pois.geojson', driver='GeoJSON')
                create_map(latitude, longitude, osm_pois=osm_pois_gdf)
            except Exception as e:
                st.error(f"Error fetching OSM points of interest data: {e}")

elif main_nav == "Data Analysis":
    st.write("Data Analysis page under construction")

# Add About and Contact sections in the sidebar
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

