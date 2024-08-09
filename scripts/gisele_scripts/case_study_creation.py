import os
import geopandas as gpd
import pandas as pd
import streamlit as st

def new_case_study(parameters, output_path_clusters):
    try:
        st.write("2. New case study creation")

        gisele_folder = parameters["gisele_folder"]
        crs = parameters["crs"]

        # Define paths
        database = gisele_folder
        study_area_folder = os.path.join(database, 'data', '3_user_uploaded_data', 'selected_area.geojson')
        intermediate_output_folder = os.path.join(database, 'data', '4_intermediate_output')
        final_output_folder = os.path.join(database, 'data', '5_final_output')

        # Create necessary directories for the case study
        os.makedirs(os.path.join(intermediate_output_folder, 'Communities'), exist_ok=True)
        os.makedirs(os.path.join(intermediate_output_folder, 'Microgrid'), exist_ok=True)
        os.makedirs(os.path.join(intermediate_output_folder, 'Optimization', 'MILP_output'), exist_ok=True)
        os.makedirs(os.path.join(final_output_folder, 'MILP_processed'), exist_ok=True)
        os.makedirs(os.path.join(intermediate_output_folder, 'Optimization', 'all_data', 'Lines_connections'), exist_ok=True)
        os.makedirs(os.path.join(intermediate_output_folder, 'Optimization', 'all_data', 'Lines_marked'), exist_ok=True)

        # Copy the configuration file to the destination directory
        configuration_path = os.path.join(database, 'data', '0_configuration_files', 'Configuration.csv')
        pd.read_csv(configuration_path).to_csv(os.path.join(intermediate_output_folder, 'Configuration.csv'), index=False)

        # Process and save substations data
        substations_path = os.path.join(database, 'data', '4_intermediate_output', 'connection_points', 'con_points.shp')
        Substations = gpd.read_file(substations_path)
        Substations = Substations.to_crs(crs)
        Substations['X'] = Substations.geometry.apply(lambda geom: geom.xy[0][0])
        Substations['Y'] = Substations.geometry.apply(lambda geom: geom.xy[1][0])
        Substations.to_file(os.path.join(database, 'data', '2_downloaded_input_data', 'substations', 'substations.shp'))

        # Save the study area file
        study_area = gpd.read_file(study_area_folder)
        study_area.to_file(os.path.join(intermediate_output_folder, 'Study_area.geojson'))

        # Process and save the clusters data
        Clusters = gpd.read_file(output_path_clusters)
        Clusters = Clusters.to_crs(crs)
        Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
        for i, row in Clusters.iterrows():
            if row['geometry'].geom_type == 'MultiPolygon':
                Clusters.at[i, 'geometry'] = row['geometry'][0]
        Clusters.to_file(os.path.join(intermediate_output_folder, 'clustering', 'Communities_boundaries.shp'))

        st.write("New case study creation completed")
        return Clusters, study_area, Substations

    except Exception as e:
        st.error(f"An error occurred during case study creation: {e}")
        raise

def create():
    try:
        st.write("Initializing case study creation...")

        # Define the parameters required for the case study creation
        parameters = {
            "gisele_folder": "/mount/src/gisele",  # Base folder path
            "crs": "EPSG:4326"  # The coordinate reference system to be used
        }
        output_path_clusters = os.path.join(parameters["gisele_folder"], 'data', '4_intermediate_output', 'clustering', 'Communities_boundaries.shp')  # Path to the clusters file

        # Call the new_case_study function
        Clusters, study_area, Substations = new_case_study(parameters, output_path_clusters)

        # Display the results excluding the geometry column
        st.write("Case study created successfully.")
        st.write("Clusters:", Clusters.drop(columns='geometry'))  # Exclude geometry column
        st.write("Study Area:", study_area.drop(columns='geometry'))  # Exclude geometry column
        st.write("Substations:", Substations.drop(columns='geometry'))  # Exclude geometry column

    except Exception as e:
        st.error(f"An error occurred during the case study creation process: {e}")
        raise
