import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

# Define paths
COMBINED_BUILDINGS_FILE = 'data/output/buildings/combined_buildings.geojson'
CLUSTERED_POINTS_FILE = 'data/output/clustering/clustered_points.geojson'
CONVEX_HULLS_FILE = 'data/output/clustering/convex_hulls.geojson'

# Ensure output directory exists
os.makedirs('data/output/clustering', exist_ok=True)

# Load combined buildings
def load_combined_buildings():
    if os.path.exists(COMBINED_BUILDINGS_FILE):
        return gpd.read_file(COMBINED_BUILDINGS_FILE)
    return gpd.GeoDataFrame({'geometry': [], 'source': []}, crs='EPSG:4326')

# Save clustered points
def save_clustered_points(gdf):
    gdf.to_file(CLUSTERED_POINTS_FILE, driver='GeoJSON')

# Save convex hulls
def save_convex_hulls(gdf):
    gdf.to_file(CONVEX_HULLS_FILE, driver='GeoJSON')

# Create clustering map
def create_clustering_map(clustered_gdf=None, hulls_gdf=None):
    m = folium.Map(location=[combined_buildings.to_crs(epsg=4326).geometry.centroid.y.mean(), 
                             combined_buildings.to_crs(epsg=4326).geometry.centroid.x.mean()], zoom_start=15)

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

    # Add combined buildings as points
    combined_buildings_4326 = combined_buildings.to_crs(epsg=4326)
    for idx, row in combined_buildings_4326.iterrows():
        folium.CircleMarker(location=[row.geometry.centroid.y, row.geometry.centroid.x], radius=2, color='black').add_to(m)

    # Add clustered points
    if clustered_gdf is not None:
        clustered_gdf_4326 = clustered_gdf.to_crs(epsg=4326)
        cluster_colors = plt.cm.get_cmap('tab20', clustered_gdf_4326['cluster'].max() + 1)
        for cluster_id in clustered_gdf_4326['cluster'].unique():
            cluster_points = clustered_gdf_4326[clustered_gdf_4326['cluster'] == cluster_id]
            color = cluster_colors(cluster_id)
            for idx, row in cluster_points.iterrows():
                folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=2, color=f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}').add_to(m)

    # Add convex hulls
    if hulls_gdf is not None:
        hulls_gdf_4326 = hulls_gdf.to_crs(epsg=4326)
        for idx, row in hulls_gdf_4326.iterrows():
            color = cluster_colors(row['cluster'])
            folium.GeoJson(row.geometry, style_function=lambda x, color=color: {'color': f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}', 'fillOpacity': 0.1}).add_to(m)

    return m

def show():
    # Load combined buildings
    global combined_buildings
    combined_buildings = load_combined_buildings()
    combined_buildings = combined_buildings.to_crs(epsg=3857)  # Reproject to meters

    # Get building centroids
    building_centroids = combined_buildings.copy()
    building_centroids['geometry'] = building_centroids['geometry'].centroid

    # Streamlit UI
    st.title("Building Clustering")

    # DBSCAN parameters
    eps = st.number_input("EPS (meters)", min_value=1, value=100)
    min_samples = st.number_input("Min Samples", min_value=1, value=5)

    # Cluster button
    if st.button("Cluster"):
        coords = building_centroids.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = db.labels_
        building_centroids['cluster'] = labels

        # Print table with cluster id and number of points
        cluster_summary = building_centroids['cluster'].value_counts().reset_index()
        cluster_summary.columns = ['Cluster ID', 'Number of Points']
        st.write(cluster_summary)

        # Create convex hulls
        hulls = []
        for cluster_id in cluster_summary['Cluster ID']:
            if cluster_id != -1:
                cluster_points = building_centroids[building_centroids['cluster'] == cluster_id]
                hull = MultiPoint(cluster_points.geometry.tolist()).convex_hull
                hulls.append({'cluster': cluster_id, 'geometry': hull})

        hulls_gdf = gpd.GeoDataFrame(hulls, crs=building_centroids.crs)
        save_clustered_points(building_centroids)
        save_convex_hulls(hulls_gdf)

        # Update map with clusters and convex hulls
        m = create_clustering_map(building_centroids, hulls_gdf)
        st_folium(m, width=1400, height=800)

    # Confirm button to save clusters and hulls
    if st.button("Confirm Clusters and Save"):
        save_clustered_points(building_centroids)
        save_convex_hulls(hulls_gdf)
        st.success("Clusters and convex hulls saved successfully!")

    # Display initial map
    else:
        m = create_clustering_map()
        st_folium(m, width=1400, height=800)

if __name__ == "__main__":
    show()
