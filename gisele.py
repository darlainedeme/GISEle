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

def create_map(latitude, longitude, geojson_data, buildings_data, osm_buildings=None, osm_roads=None, osm_pois=None):
    m = folium.Map(location=[latitude, longitude], zoom_start=15)  # Increased zoom level

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

    # Add GeoDataFrame to the map
    if geojson_data:
        folium.GeoJson(geojson_data, name="Uploaded GeoJSON").add_to(m)

    # Add Google Buildings data to the map as a selectable layer
    if buildings_data:
        folium.GeoJson(buildings_data, name="Google Buildings", style_function=lambda x: {
            'fillColor': 'green',
            'color': 'green',
            'weight': 1,
        }, show=False).add_to(m)  # Default to not show

        # Add MarkerCluster for Google Buildings
        marker_cluster = MarkerCluster(name='Google Buildings Clusters', show=True).add_to(m)
        for feature in buildings_data['features']:
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0][0]
            elif geom['type'] == 'MultiPolygon':
                coords = geom['coordinates'][0][0][0]
            else:
                coords = geom['coordinates']
            if len(coords) >= 2:
                folium.Marker(location=[coords[1], coords[0]]).add_to(marker_cluster)
                
    # Add OSM buildings data to the map
    if osm_buildings is not None:
        folium.GeoJson(osm_buildings, name="OSM Buildings", style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 1,
        }).add_to(m)
        
    # Add OSM roads data to the map
    if osm_roads is not None:
        folium.GeoJson(osm_roads, name="OSM Roads", style_function=lambda x: {
            'color': 'orange',
            'weight': 2,
        }).add_to(m)

    # Add OSM POIs data to the map
    if osm_pois is not None:
        folium.GeoJson(osm_pois, name="OSM POIs", style_function=lambda x: {
            'fillColor': 'red',
            'color': 'red',
            'weight': 1,
        }).add_to(m)

    # Add drawing and fullscreen plugins
    Draw(export=True, filename='data.geojson', position='topleft').add_to(m)
    Fullscreen(position='topleft').add_to(m)
    MeasureControl(position='bottomleft').add_to(m)
    
    folium.LayerControl().add_to(m)

    # Display the map
    folium_static(m, width=1450, height=800)  # Wider map

def uploaded_file_to_gdf(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as temp_file:
        temp_file.write(data.getvalue())
        temp_filepath = temp_file.name

    gdf = gpd.read_file(temp_filepath)
    return gdf

def fetch_osm_data(bounds):
    # Define the OSM data sources
    tags = {'building': True, 'highway': True, 'amenity': True}
    bbox = [bounds[1], bounds[0], bounds[3], bounds[2]]  # [south, west, north, east]
    
    # Fetch OSM buildings
    osm_buildings = ox.geometries_from_bbox(*bbox, tags={'building': True})
    
    # Fetch OSM roads
    osm_roads = ox.geometries_from_bbox(*bbox, tags={'highway': True})
    
    # Fetch OSM POIs
    osm_pois = ox.geometries_from_bbox(*bbox, tags={'amenity': True})
    
    return osm_buildings, osm_roads, osm_pois

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

                if gdf.empty or gdf.is_empty.any():
                    st.error("Uploaded GeoJSON file is empty or contains null geometries.")
                else:
                    geojson_data = gdf.to_json()
                    
                    st.info("Fetching building data...")
                    coords = gdf.geometry.total_bounds
                    geom = ee.Geometry.Rectangle([coords[0], coords[1], coords[2], coords[3]])
                    
                    buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
                        .filter(ee.Filter.intersects('.geo', geom))
                    
                    download_url = buildings.getDownloadURL('geojson')
                    response = requests.get(download_url)
                    buildings_data = response.json()
                    
                    st.info("Fetching OSM data...")
                    osm_buildings, osm_roads, osm_pois = fetch_osm_data(coords)

                    st.info("Creating map...")
                    centroid = gdf.geometry.centroid.iloc[0]
                    create_map(centroid.y, centroid.x, geojson_data, buildings_data, osm_buildings, osm_roads, osm_pois)
                    st.success("Map created successfully!")
            except KeyError as e:
                st.error(f"Error processing file: {e}")
            except IndexError as e:
                st.error(f"Error processing file: {e}")
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
