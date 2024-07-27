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
from shapely.geometry import Polygon
import pandas as pd

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
page = st.sidebar.radio("Navigation", ["Home", "Area Selection", "Data Gathering", "Analysis"], key="main_nav")

# Call to initialize Earth Engine
initialize_earth_engine()

def create_combined_buildings_layer(osm_buildings, google_buildings):
    # Ensure both GeoDataFrames are in the same CRS
    osm_buildings = osm_buildings.to_crs(epsg=4326)
    google_buildings = gpd.GeoDataFrame.from_features(google_buildings["features"]).set_crs(epsg=4326)

    # Remove Google buildings that touch OSM buildings
    osm_buildings['source'] = 'osm'
    google_buildings['source'] = 'google'
    
    osm_dissolved = osm_buildings.unary_union

    # Filter Google buildings that do not intersect with OSM buildings
    filtered_google = google_buildings[~google_buildings.intersects(osm_dissolved)]

    # Combine OSM buildings and filtered Google buildings
    combined_buildings = gpd.GeoDataFrame(pd.concat([osm_buildings, filtered_google], ignore_index=True))  

    return combined_buildings

def create_map(latitude, longitude, geojson_data, combined_buildings=None, osm_roads=None, osm_pois=None, missing_layers=None):
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
    folium.TileLayer(
        tiles='https://dev.virtualearth.net/REST/V1/Imagery/Map/AerialWithLabels/{z}/{y}/{x}?mapSize=500,500&key=BingMapsAPIKey',
        attr='Bing',
        name='Bing Maps',
        overlay=False,
        control=True
    ).add_to(m)

    # Add GeoDataFrame to the map
    if geojson_data:
        folium.GeoJson(geojson_data, name="Uploaded GeoJSON").add_to(m)

    # Add combined buildings data to the map
    if combined_buildings is not None:
        style_function = lambda x: {
            'fillColor': 'green' if x['properties']['source'] == 'osm' else 'blue',
            'color': 'green' if x['properties']['source'] == 'osm' else 'blue',
            'weight': 1,
        }
        folium.GeoJson(combined_buildings, name="Combined Buildings", style_function=style_function).add_to(m)

        # Add MarkerCluster for combined buildings
        marker_cluster = MarkerCluster(name='Combined Buildings Clusters').add_to(m)
        for _, row in combined_buildings.iterrows():
            folium.Marker(location=[row.geometry.centroid.y, row.geometry.centroid.x]).add_to(marker_cluster)

    # Add OSM Roads data to the map
    if osm_roads is not None and not osm_roads.empty:
        folium.GeoJson(osm_roads.to_json(), name='OSM Roads', style_function=lambda x: {
            'fillColor': 'orange',
            'color': 'orange',
            'weight': 1,
        }).add_to(m)

    # Add OSM Points of Interest data to the map
    if osm_pois is not None and not osm_pois.empty:
        folium.GeoJson(osm_pois.to_json(), name='OSM Points of Interest', style_function=lambda x: {
            'fillColor': 'red',
            'color': 'red',
            'weight': 1,
        }).add_to(m)

    # Add drawing and fullscreen plugins
    Draw(export=True, filename='data.geojson', position='topleft').add_to(m)
    Fullscreen(position='topleft').add_to(m)
    MeasureControl(position='bottomleft').add_to(m)
    
    folium.LayerControl().add_to(m)

    # Save map to session state
    st.session_state.map = m

    # Display the map
    st_folium(m, width=1450, height=800)

    # Indicate missing layers
    if missing_layers:
        st.write("The following layers weren't possible to obtain for the selected area:")
        for layer in missing_layers:
            st.write(f"- {layer}")

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
    st.write("### Area Selection")
    st.write("Define the area of interest using one of the methods below:")

    which_modes = ['By address', 'By coordinates', 'Upload file']
    which_mode = st.sidebar.selectbox('Select mode', which_modes, index=2, key="mode_select_area")

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
                        create_map(location.latitude, location.longitude, None)
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
                    st.session_state.latitude = float(latitude)
                    st.session_state.longitude = float(longitude)
                    create_map(float(latitude), float(longitude), None)
            except Exception as e:
                st.error(f"Error creating map: {e}")
        else:
            st.error("Please provide both latitude and longitude.")

    elif which_mode == 'Upload file':
        data = st.sidebar.file_uploader("Upload a GeoJSON file", type=["geojson"], key="file_uploader")

        if data:
            try:
                gdf = uploaded_file_to_gdf(data)
                if gdf.empty or gdf.is_empty.any():
                    st.error("Uploaded GeoJSON file is empty or contains null geometries.")
                else:
                    geojson_data = gdf.to_json()
                    centroid = gdf.geometry.centroid.iloc[0]

                    # Save map data to session state
                    st.session_state.latitude = centroid.y
                    st.session_state.longitude = centroid.x
                    st.session_state.geojson_data = geojson_data

                    create_map(centroid.y, centroid.x, geojson_data)
                    st.success("Area selected successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    if 'latitude' in st.session_state and 'longitude' in st.session_state:
        st.write("### Selected Area")
        create_map(st.session_state.latitude, st.session_state.longitude, st.session_state.get('geojson_data', None))
        st.write("Proceed to the Data Gathering page to extract data for the selected area.")

elif page == "Data Gathering":
    st.write("### Data Gathering")

    if 'latitude' not in st.session_state or 'longitude' not in st.session_state:
        st.error("Please select an area in the Area Selection page first.")
    else:
        try:
            st.info("Fetching building data...")
            geom = ee.Geometry.Point([st.session_state.longitude, st.session_state.latitude]).buffer(1000)
            selected_polygon = Polygon(geom.bounds().getInfo()['coordinates'][0])

            buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
                .filter(ee.Filter.intersects('.geo', geom))
            
            download_url = buildings.getDownloadURL('geojson')
            response = requests.get(download_url)
            google_buildings = response.json()
            google_buildings_gdf = gpd.GeoDataFrame.from_features(google_buildings["features"]).set_crs(epsg=4326)
            google_buildings_gdf = google_buildings_gdf.clip(selected_polygon)
            
            st.info("Fetching OSM data...")
            missing_layers = []
            try:
                osm_buildings = ox.geometries_from_polygon(selected_polygon, tags={'building': True})
                osm_buildings = osm_buildings.clip(selected_polygon)
            except Exception as e:
                st.error(f"Error fetching OSM buildings data: {e}")
                osm_buildings = gpd.GeoDataFrame()
                missing_layers.append('buildings')

            try:
                osm_roads = ox.geometries_from_polygon(selected_polygon, tags={'highway': True})
                osm_roads = osm_roads.clip(selected_polygon)
            except Exception as e:
                st.error(f"Error fetching OSM roads data: {e}")
                osm_roads = gpd.GeoDataFrame()
                missing_layers.append('roads')

            try:
                osm_pois = ox.geometries_from_polygon(selected_polygon, tags={'amenity': True})
                osm_pois = osm_pois.clip(selected_polygon)
            except Exception as e:
                st.error(f"Error fetching OSM points of interest data: {e}")
                osm_pois = gpd.GeoDataFrame()
                missing_layers.append('points of interest')

            st.info("Creating combined buildings layer...")
            combined_buildings = create_combined_buildings_layer(osm_buildings, google_buildings_gdf)
            
            st.info("Creating map...")
            create_map(st.session_state.latitude, st.session_state.longitude, st.session_state.geojson_data, combined_buildings, osm_roads, osm_pois, missing_layers)
            st.success("Data gathered and map updated successfully!")

            # Save data to session state
            st.session_state.combined_buildings = combined_buildings
            st.session_state.osm_roads = osm_roads
            st.session_state.osm_pois = osm_pois
            st.session_state.missing_layers = missing_layers

        except Exception as e:
            st.error(f"Error gathering data: {e}")

elif page == "Analysis":
    st.write("Analysis page under construction")

if 'map' in st.session_state:
    st_folium(st.session_state.map, width=1450, height=800)
