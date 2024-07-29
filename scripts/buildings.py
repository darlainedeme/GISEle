import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
import os
from shapely.geometry import Polygon
from folium.plugins import Draw, Fullscreen, MeasureControl

# Define paths
COMBINED_BUILDINGS_FILE = 'data/output/buildings/combined_buildings.geojson'
USER_POLYGONS_FILE = 'data/output/buildings/user_polygons.geojson'

# Ensure output directory exists
os.makedirs('data/output/buildings', exist_ok=True)

def load_combined_buildings():
    if os.path.exists(COMBINED_BUILDINGS_FILE):
        return gpd.read_file(COMBINED_BUILDINGS_FILE)
    return gpd.GeoDataFrame({'geometry': [], 'source': []}, crs='EPSG:4326')

def save_combined_buildings(gdf):
    gdf.to_file(COMBINED_BUILDINGS_FILE, driver='GeoJSON')

def create_buildings_map(combined_buildings, user_polygons=None):
    m = folium.Map(location=[combined_buildings.geometry.centroid.y.mean(), combined_buildings.geometry.centroid.x.mean()], zoom_start=15)

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

    # Add combined buildings layer
    style_function = lambda x: {
        'fillColor': 'green' if x['properties']['source'] == 'osm' else 'blue',
        'color': 'green' if x['properties']['source'] == 'osm' else 'blue',
        'weight': 1,
    }
    folium.GeoJson(combined_buildings, name="Combined Buildings", style_function=style_function).add_to(m)

    # Add user polygons layer
    if user_polygons is not None:
        style_function_user = lambda x: {
            'fillColor': 'red',
            'color': 'red',
            'weight': 1,
        }
        folium.GeoJson(user_polygons, name="User Polygons", style_function=style_function_user).add_to(m)

    # Add drawing tools
    draw = Draw(
        export=True,
        filename='user_polygons.geojson',
        draw_options={'polygon': True, 'polyline': False, 'circle': False, 'rectangle': False, 'marker': False, 'circlemarker': False}
    )
    draw.add_to(m)

    # Add fullscreen and measure controls
    Fullscreen().add_to(m)
    MeasureControl().add_to(m)

    # Add legend
    legend_html = '''
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 150px; height: 90px; 
                 background-color: white; z-index:9999; font-size:14px;
                 border:2px solid grey; padding: 10px;">
     <b>Legend</b><br>
     <i class="fa fa-map-marker fa-2x" style="color:green"></i> OSM Buildings<br>
     <i class="fa fa-map-marker fa-2x" style="color:blue"></i> Google Buildings<br>
     <i class="fa fa-map-marker fa-2x" style="color:red"></i> User Polygons
     </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def show():
    st.title("Buildings Selection and Visualization")

    # Load combined buildings
    combined_buildings = load_combined_buildings()

    # Upload geojson file
    uploaded_file = st.file_uploader("Upload GeoJSON file", type="geojson")
    if uploaded_file:
        user_polygons_gdf = gpd.read_file(uploaded_file)
        user_polygons_gdf['source'] = 'user'
        combined_buildings = gpd.GeoDataFrame(pd.concat([combined_buildings, user_polygons_gdf], ignore_index=True))
        save_combined_buildings(combined_buildings)
        st.success("Uploaded polygons added to the map and combined buildings updated.")
    else:
        user_polygons_gdf = None

    # Display map with combined buildings
    m = create_buildings_map(combined_buildings, user_polygons_gdf)
    st_folium(m, width=1400, height=800)

    # Option to download the updated combined buildings
    combined_buildings_geojson = combined_buildings.to_json()
    st.download_button("Download Combined Buildings", data=combined_buildings_geojson, file_name="combined_buildings.geojson", mime="application/json")

if __name__ == "__main__":
    show()
