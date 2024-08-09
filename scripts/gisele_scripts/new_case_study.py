import os
import geopandas as gpd
import pandas as pd
import streamlit as st

def new_case_study(parameters, output_path_clusters):
    st.write("2. New case study creation")

    gisele_folder = "/mount/src/gisele"
    case_study = parameters["case_study"]
    crs = parameters["crs"]

    database = gisele_folder
    study_area_folder = os.path.join(database, 'data', '3_user_uploaded_data', 'selected_area.geojson')

    case_study_path = os.path.join('Case studies', case_study)
    if not os.path.exists(case_study_path):
        # Create new folders for the study case
        os.makedirs(case_study_path)
        os.makedirs(os.path.join(case_study_path, 'Intermediate'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Communities'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Microgrid'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Geospatial_Data'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'MILP_output'))
        os.makedirs(os.path.join(case_study_path, 'Output'))
        os.makedirs(os.path.join(case_study_path, 'Output', 'MILP_processed'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'all_data'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'MILP_input'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'all_data', 'Lines_connections'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'all_data', 'Lines_marked'))

        # Define the path to the configuration file
        configuration_path = os.path.join(database, 'data', '0_configuration_files', 'Configuration.csv')

        # Ensure the destination directory exists
        destination_dir = os.path.join(case_study_path, 'Input')
        os.makedirs(destination_dir, exist_ok=True)

        # Copy the configuration file to the destination directory
        pd.read_csv(configuration_path).to_csv(os.path.join(destination_dir, 'Configuration.csv'))

        # Read the possible connection points and write them in the case study's folder
        substations_path = os.path.join(database, 'data', '4_intermediate_output', 'connection_points', 'con_points.shp')
        Substations = gpd.read_file(substations_path)
        Substations_crs = Substations.to_crs(crs)
        Substations['X'] = [geom.xy[0][0] for geom in Substations_crs['geometry']]
        Substations['Y'] = [geom.xy[1][0] for geom in Substations_crs['geometry']]
        substations_output_path = os.path.join(database, 'data', '2_downloaded_input_data', 'substations', 'substations.shp')
        Substations.to_file(substations_output_path)

        # Read the polygon of the study area and write it in the local database
        study_area = gpd.read_file(study_area_folder)
        study_area.to_file(os.path.join(destination_dir, 'Study_area.geojson'))

        # Read the communities and write them in the local database
        Clusters = gpd.read_file(output_path_clusters)
        Clusters = Clusters.to_crs(crs)
        Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
        for i, row in Clusters.iterrows():
            if row['geometry'].geom_type == 'MultiPolygon':
                Clusters.at[i, 'geometry'] = row['geometry'][0]
        clusters_output_path = os.path.join(database, 'data', '4_intermediate_output', 'clustering', 'Communities_boundaries.shp')
        Clusters.to_file(clusters_output_path)
    else:
        destination_path = os.path.join(database, 'data', '4_intermediate_output', 'clustering', 'Communities_boundaries.shp')
        st.write(output_path_clusters)
        source_gdf = gpd.read_file(output_path_clusters)
        source_gdf.to_file(destination_path)
        Clusters = gpd.read_file(destination_path)
        Clusters = Clusters.to_crs(crs)
        Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
        study_area = gpd.read_file(os.path.join(database, 'data', '3_user_uploaded_data', 'selected_area.geojson'))
        Substations = gpd.read_file(os.path.join(database, 'data', '2_downloaded_input_data', 'substations', 'substations.shp'))

    st.write("New case study creation completed")
    return Clusters, study_area, Substations
