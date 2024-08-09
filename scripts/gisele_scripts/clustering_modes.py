import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.interpolate import interp1d
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import zipfile
from rasterio.enums import Resampling
import branca.colormap as cm

import os
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.ops import unary_union

# Main function to perform the building clustering
def building_to_cluster_v1(crs, flag):
    # Hardcoded path to the study region
    path = os.path.join("data", "3_user_uploaded_data", "selected_area.geojson")
    studyregion_original = gpd.read_file(path)
    study_area_buffered = studyregion_original.buffer((2500 * 0.1 / 11250))

    # Update paths as per the new directory structure
    building_path = os.path.join("data", "2_downloaded_input_data", "buildings", "mit", "merged.shp")
    urbanity = os.path.join("data", "2_downloaded_input_data", "urbanity", "urbanity.tif")
    output_folder_clusters = os.path.join("data", "4_intermediate_output", "clustering")
    output_path_clusters = os.path.join(output_folder_clusters, "Communities_boundaries.shp")
    
    output_folder_pointsclipped = os.path.join("data", "2_downloaded_input_data", "buildings", "mit", "points")
    output_path_points_clipped = os.path.join(output_folder_pointsclipped, 'points_clipped.shp')

    if not flag:
        # Load and process building data
        buildings_df_original = gpd.read_file(building_path)
        buildings_df_original.rename(columns={'area_in_me': 'area'}, inplace=True)
        buildings_df = buildings_df_original.to_crs(crs)
        studyregion = studyregion_original.to_crs(crs)
        buildings_df['geometry'] = buildings_df.geometry.centroid
        buildings_df = gpd.clip(buildings_df, studyregion)
        buildings_df = buildings_df.reset_index(drop=True)
        area_lower_bound = 12
        buildings_df = buildings_df[buildings_df['area'] > area_lower_bound]
        buildings_df['ID'] = [*range(len(buildings_df))]
        buildings_df.reset_index(inplace=True, drop=True)

        with rasterio.open(urbanity) as src:
            out_image, out_transform = mask(src, study_area_buffered.to_crs(src.crs), crop=True)

        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        with rasterio.open(os.path.join(os.path.dirname(urbanity), "Urbanity_clip.tif"), "w", **out_meta) as dest:
            dest.write(out_image)

        with rasterio.open(os.path.join(os.path.dirname(urbanity), "Urbanity_clip.tif")) as src:
            warp = src.read(1, out_shape=(src.height, src.width), resampling=Resampling.bilinear)
            output_modified_raster = os.path.join(os.path.dirname(urbanity), "Urbanity_clip_rep_convolve.tif")
            raster = src.read(1)
            kernel = np.ones((3, 3))
            neighbor_sum = convolve(raster, kernel, mode='nearest')
            result = raster + 0.5 * (neighbor_sum - raster)
            result = result.astype(np.int16)
            meta = src.meta.copy()

            with rasterio.open(output_modified_raster, "w", **meta) as dest:
                dest.write(result, 1)
                
        with rasterio.open(urbanity) as src:
            out_image, out_transform = mask(src, study_area_buffered.to_crs(src.crs), crop=True)

        # Calculate min and max urbanity values from the raster
        min_urbanity = np.min(out_image[out_image > 0])  # Ignoring zeros which might represent no data
        max_urbanity = np.max(out_image)

        Urbanity_final = rasterio.open(output_modified_raster)
        coords = [(point.x, point.y) for point in buildings_df['geometry']]
        buildings_df['urbanity'] = [x[0] for x in Urbanity_final.sample(coords)]

        if not os.path.exists(output_folder_pointsclipped):
            os.makedirs(output_folder_pointsclipped)
        buildings_df.to_file(output_path_points_clipped)

    else:
        buildings_df = gpd.read_file(output_path_points_clipped)

    buildings_df = buildings_df.reset_index(drop=True)
    x_interp = [min_urbanity, max_urbanity]  # Automatically calculated from the raster
    x_interp = [55, 150]
    y_interp = [60, 25]

    interpolator = interp1d(x_interp, y_interp, kind='linear', fill_value='extrapolate')
    for index, row in buildings_df.iterrows():
        point_geometry = row.geometry
        urbanity_value = row['urbanity']
        buffer_radius = interpolator(urbanity_value)
        buildings_df.at[index, 'buffer'] = point_geometry.buffer(buffer_radius)

    geometries = buildings_df['buffer'].tolist()
    clusters_MP = unary_union(geometries)

    if isinstance(clusters_MP, MultiPolygon):
        clusters = [poly for poly in clusters_MP.geoms]
    elif isinstance(clusters_MP, Polygon):
        clusters = [clusters_MP]
    else:
        clusters = []

    clusters_gdf = gpd.GeoDataFrame(geometry=clusters, crs=crs)
    clusters_gdf = clusters_gdf.reset_index().rename(columns={'index': 'cluster_ID'})
    clusters_gdf['cluster_ID'] = clusters_gdf['cluster_ID'] + 1
    spatial_join = gpd.sjoin(buildings_df, clusters_gdf, how='left', predicate='within')

    try:
        buildings_df['cluster_ID'] = spatial_join['cluster_ID']
    except:
        buildings_df['cluster_ID'] = spatial_join['cluster_ID_right']

    cluster_counts = buildings_df['cluster_ID'].value_counts()
    clusters_to_keep = cluster_counts[cluster_counts >= 40].index
    clusters_gdf = clusters_gdf[clusters_gdf['cluster_ID'].isin(clusters_to_keep)]

    buildings_df.loc[~buildings_df['cluster_ID'].isin(clusters_to_keep), 'cluster_ID'] = -1
    average_elec_access = buildings_df.groupby('cluster_ID')['elec acces'].mean()

    if len(clusters_gdf) > 0:
        clusters_gdf = clusters_gdf.merge(average_elec_access, left_on='cluster_ID', right_index=True, how='left')
        clusters_gdf.drop(columns=['cluster_ID'], inplace=True)
        clusters_gdf['elec acces'] = clusters_gdf['elec acces'] / 100

    clusters_gdf = clusters_gdf.reset_index().rename(columns={'index': 'cluster_ID'})
    clusters_gdf['cluster_ID'] = clusters_gdf.index + 1

    if not os.path.exists(output_folder_clusters):
        os.makedirs(output_folder_clusters)

    clusters_gdf.to_file(output_path_clusters)

    return clusters_gdf, buildings_df, output_path_clusters, output_path_points_clipped

def create_map(clusters_gdf):
    # Ensure the GeoDataFrame is in EPSG 4326 for Folium
    clusters_gdf = clusters_gdf.to_crs(epsg=4326)

    # Initialize the map centered around the mean of the cluster centroids
    m = folium.Map(
        location=[clusters_gdf.geometry.centroid.y.mean(), clusters_gdf.geometry.centroid.x.mean()],
        zoom_start=12
    )

    # Add base map tiles
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

    # Define a colormap based on 'elec acces'
    colormap = cm.linear.YlOrRd_09.scale(clusters_gdf['elec acces'].min(), clusters_gdf['elec acces'].max())

    # Add the clusters polygons to the map with color based on 'elec acces'
    folium.GeoJson(
        clusters_gdf,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['elec acces']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6,
        }
    ).add_to(m)

    # Add the colormap legend to the map
    colormap.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    folium.plugins.Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen',
                              force_separate_button=False).add_to(m)
                              
    # Display the map in Streamlit
    st_folium(m, width=1400, height=800)

# Load combined buildings
def load_combined_buildings():
    if os.path.exists(COMBINED_BUILDINGS_FILE):
        return gpd.read_file(COMBINED_BUILDINGS_FILE)
    return gpd.GeoDataFrame({'geometry': [], 'source': []}, crs='EPSG:4326')

# Save clustered points
def save_clustered_points(gdf):
    gdf.to_file(CLUSTERED_POINTS_FILE, driver='GeoJSON')

# Save buffered polygons
def save_buffered_polygons(gdf):
    gdf.to_file(BUFFERED_POLYGONS_FILE, driver='GeoJSON')

# Create clustering map
def create_clustering_map(clustered_gdf=None, buffered_gdf=None):
    m = folium.Map(location=[clustered_gdf.geometry.centroid.y.mean(), 
                             clustered_gdf.geometry.centroid.x.mean()], zoom_start=15)

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

    # Add clustered points
    if clustered_gdf is not None and not clustered_gdf.empty:
        clustered_gdf_4326 = clustered_gdf.to_crs(epsg=4326)
        for idx, row in clustered_gdf_4326.iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=2, color='black').add_to(m)

    # Add buffered polygons
    if buffered_gdf is not None and not buffered_gdf.empty:
        buffered_gdf_4326 = buffered_gdf.to_crs(epsg=4326)
        folium.GeoJson(
            buffered_gdf_4326,
            style_function=lambda x: {
                'fillColor': 'blue',
                'color': 'blue',
                'weight': 1,
                'fillOpacity': 0.4,
            }
        ).add_to(m)

    return m

def show():
    # Define paths
    COMBINED_BUILDINGS_FILE = os.path.join('data', '2_downloaded_input_data', 'buildings', 'combined_buildings.geojson')
    CLUSTERED_POINTS_FILE = os.path.join('data', '4_intermediate_output', 'clustering', 'clustered_points.geojson')
    BUFFERED_POLYGONS_FILE = os.path.join('data', '4_intermediate_output', 'clustering', 'buffered_polygons.geojson')

    # Radio button for method selection
    method = st.radio("Select Clustering Method", ('MIT', 'Standard'), index=0)
    
    with st.expander("Parameters", expanded=False):
        # Input fields for the user to specify CRS and whether to skip processing
        crs = st.number_input("CRS (Coordinate Reference System)", value=21095)
        flag = st.checkbox("Skip Processing", value=False)

    # Initialize session state for clusters, buildings, and paths
    if "clusters_gdf" not in st.session_state:
        st.session_state["clusters_gdf"] = None
    if "buildings_df" not in st.session_state:
        st.session_state["buildings_df"] = None
    if "output_path_clusters" not in st.session_state:
        st.session_state["output_path_clusters"] = None
    if "output_path_points_clipped" not in st.session_state:
        st.session_state["output_path_points_clipped"] = None

    if method == 'MIT':
        if st.button("Run Clustering"):
            clusters_gdf, buildings_df, output_path_clusters, output_path_points_clipped = building_to_cluster_v1(crs, flag)
            
            # Ensure that the output GeoDataFrames are in the correct CRS
            clusters_gdf = clusters_gdf.to_crs(epsg=crs)
            buildings_df = buildings_df.to_crs(epsg=crs)
            
            st.session_state["clusters_gdf"] = clusters_gdf
            st.session_state["buildings_df"] = buildings_df
            st.session_state["output_path_clusters"] = output_path_clusters
            st.session_state["output_path_points_clipped"] = output_path_points_clipped
            st.success("Clustering completed.")
    
    # Check if clusters_gdf exists in session state before trying to create the map
    if st.session_state["clusters_gdf"] is not None:
        clusters_gdf = st.session_state["clusters_gdf"]
        
        # Create and display the map using the new create_map function
        create_map(clusters_gdf)

        '''
        # Ensure clustering was performed before attempting to export
        if st.session_state["output_path_clusters"] and st.session_state["output_path_points_clipped"]:
            # Add a button to export the clusters and points as a ZIP file
            if st.button("Export Clusters and Points"):
                # File paths for export
                zip_path = "exported_data.zip"

                # Create a ZIP file containing the Shapefile
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    zipf.write(st.session_state["output_path_clusters"], os.path.basename(st.session_state["output_path_clusters"]))
                    for file in os.listdir(os.path.dirname(st.session_state["output_path_clusters"])):
                        if file.startswith("Communities_boundaries"):
                            zipf.write(os.path.join(os.path.dirname(st.session_state["output_path_clusters"]), file), file)

                st.success(f"Export completed! Files saved in '{zip_path}'.")

                # Provide a download button for the ZIP file
                with open(zip_path, "rb") as f:
                    st.download_button('Download Exported Data', f, file_name=zip_path)
        else:
            st.error("Error: Clustering data not found. Please run the clustering process first.")
        '''
        
    else:
    
    # Ensure output directory exists
        os.makedirs('data\4_intermediate_output\clustering', exist_ok=True)
        
        # Load combined buildings
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

        # Initialize session state for clustering results
        if 'clustered_gdf' not in st.session_state:
            st.session_state.clustered_gdf = None
        if 'buffered_gdf' not in st.session_state:
            st.session_state.buffered_gdf = None

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

            # Create buffered polygons instead of convex hulls
            buffered_polygons = []
            for cluster_id in cluster_summary['Cluster ID']:
                if cluster_id != -1:
                    cluster_points = building_centroids[building_centroids['cluster'] == cluster_id]
                    buffer = cluster_points.buffer(eps)
                    merged_polygon = unary_union(buffer)
                    buffered_polygons.append({'cluster': cluster_id, 'geometry': merged_polygon})

            buffered_gdf = gpd.GeoDataFrame(buffered_polygons, crs=building_centroids.crs)
            
            # Store results in session state
            st.session_state.clustered_gdf = building_centroids
            st.session_state.buffered_gdf = buffered_gdf

            save_clustered_points(building_centroids)
            save_buffered_polygons(buffered_gdf)

            st.success("Clustering completed. You can now review the results.")

        # Display map
        if st.session_state.clustered_gdf is not None and st.session_state.buffered_gdf is not None:
            m = create_clustering_map(st.session_state.clustered_gdf, st.session_state.buffered_gdf)
            st_folium(m, width=1400, height=800)
        else:
            st.warning("Please perform clustering first.")

if __name__ == "__main__":
    show()