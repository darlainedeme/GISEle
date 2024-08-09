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

# Function to perform clustering and cleaning on building data
def poles_clustering_and_cleaning(buildings_filter, crs, chain_upper_bound, pole_upper_bound):
    def create_clusters(buildings_filter, max_distance):
        coordinates = [(point.x, point.y) for point in buildings_filter.geometry]
        kdtree = cKDTree(coordinates)
        assigned = np.zeros(len(buildings_filter.geometry), dtype=bool)
        clusters = []

        def dfs(node, current_cluster):
            neighbors = kdtree.query_ball_point(coordinates[node], chain_upper_bound)
            unassigned_neighbors = [neighbor for neighbor in neighbors if not assigned[neighbor]]
            assigned[unassigned_neighbors] = True

            if node not in current_cluster:
                current_cluster.append(node)

            for neighbor in unassigned_neighbors:
                dfs(neighbor, current_cluster)

        for i, shapely_point in enumerate(buildings_filter.geometry):
            if not assigned[i]:
                current_cluster = []
                dfs(i, current_cluster)
                clusters.append(current_cluster)
        return clusters

    result_clusters = create_clusters(buildings_filter, chain_upper_bound)
    i = 0
    for clus in result_clusters:
        if len(clus) > 2:
            coords = [(point.x, point.y) for point in buildings_filter.loc[clus, 'geometry']]
            distances = squareform(pdist(coords))
            agg_cluster = AgglomerativeClustering(distance_threshold=pole_upper_bound, n_clusters=None, linkage='complete')
            cluster_labels = agg_cluster.fit_predict(distances)
            if len(set(cluster_labels)) > 1:
                for j in list(set(cluster_labels)):
                    indices = [index for index, value in enumerate(cluster_labels) if value == j]
                    buildings_filter.loc[[clus[k] for k in indices], 'Group2'] = i
                    i += 1
            else:
                buildings_filter.loc[clus, 'Group2'] = i
        else:
            buildings_filter.loc[clus, 'Group2'] = i
        i += 1

    buildings_adjusted = []
    area = []
    num = []
    elec_access = []
    cons = []

    for group in buildings_filter['Group2'].unique():
        buildings_adjusted.append(MultiPoint(buildings_filter.loc[buildings_filter['Group2'] == group, 'geometry'].values).centroid)
        area.append(buildings_filter.loc[buildings_filter['Group2'] == group, 'area'].sum())
        cons.append(buildings_filter.loc[buildings_filter['Group2'] == group, 'cons (kWh/'].sum())
        num.append(len(buildings_filter.loc[buildings_filter['Group2'] == group, 'area']))
        elec_access.append(buildings_filter.loc[buildings_filter['Group2'] == group, 'elec acces'].mean())

    buildings_adjusted_gdf = gpd.GeoDataFrame({'area': area, 'number': num, 'cons (kWh/': cons, 'elec acces': elec_access},
                                              geometry=buildings_adjusted, crs=crs)
    return buildings_adjusted_gdf

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

        Urbanity_final = rasterio.open(output_modified_raster)
        coords = [(point.x, point.y) for point in buildings_df['geometry']]
        buildings_df['urbanity'] = [x[0] for x in Urbanity_final.sample(coords)]

        if not os.path.exists(output_folder_pointsclipped):
            os.makedirs(output_folder_pointsclipped)
        buildings_df.to_file(output_path_points_clipped)

    else:
        buildings_df = gpd.read_file(output_path_points_clipped)

    buildings_df = buildings_df.reset_index(drop=True)
    x_interp = [55, 150]
    y_interp = [120, 50]

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

def show():
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
    else:
        st.write("Standard method not yet implemented.")

    # Display map if clustering has been run
    if st.session_state["clusters_gdf"] is not None and st.session_state["buildings_df"] is not None:
        clusters_gdf = st.session_state["clusters_gdf"]
        
        # Initialize map centered on the first cluster's centroid
        m = folium.Map(location=[clusters_gdf.geometry.centroid.y.mean(), clusters_gdf.geometry.centroid.x.mean()],
                       zoom_start=12)

        # Add tile layers
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

        # Add clusters polygons to map
        for _, row in clusters_gdf.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, color=row.name: {
                    'fillColor': '#0000ff',
                    'color': '#0000ff',
                    'weight': 1,
                    'fillOpacity': 0.2
                }
            ).add_to(m)

        folium.LayerControl().add_to(m)

        # Display map in Streamlit
        st_folium(m, width=1400, height=800)

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
                    zipf.write(st.session_state["output_path_points_clipped"], os.path.basename(st.session_state["output_path_points_clipped"]))
                    for file in os.listdir(os.path.dirname(st.session_state["output_path_points_clipped"])):
                        if file.startswith("points_clipped"):
                            zipf.write(os.path.join(os.path.dirname(st.session_state["output_path_points_clipped"]), file), file)

                st.success(f"Export completed! Files saved in '{zip_path}'.")

                # Provide a download button for the ZIP file
                with open(zip_path, "rb") as f:
                    st.download_button('Download Exported Data', f, file_name=zip_path)
        else:
            st.error("Error: Clustering data not found. Please run the clustering process first.")
