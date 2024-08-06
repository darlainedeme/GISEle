import os
import sys
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import nearest_points, unary_union
import numpy as np
import rasterio
from rasterio.mask import mask
from scipy.interpolate import interp1d
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import streamlit as st

# Get the current script path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the routing_scripts directory to the system path
routing_scripts_path = os.path.join(current_dir, 'routing_scripts')
sys.path.append(routing_scripts_path)

# Import custom modules
from cleaning import *
from functions import *
from functions2 import *

import initialization, clustering, processing, collecting, optimization, results, grid, branches
import QGIS_processing_polygon as qgis_process
import Local_area_optimization as LAO
import MILP_Input_creation, MILP_models, process_output, grid_routing, Secondary_substations
from OpenEnergyMapMIT_v1 import building_to_cluster_v1

# Define function to set parameters
def set_parameters():
    st.header("Set Parameters")
    
    gisele_folder = st.text_input("Gisele Folder", os.getcwd())
    country = st.text_input("Country", 'Uganda')
    case_study = st.text_input("Case Study", 'awach555')
    crs = st.number_input("CRS", value=21095)
    resolution = st.number_input("Resolution", value=200)
    voltage = st.number_input("Voltage (kV)", value=110)
    LV_base_cost = st.number_input("LV Base Cost", value=10000)
    load_capita = st.number_input("Load per Capita (kW)", value=0.6)
    pop_per_household = st.number_input("Population per Household", value=5)
    resolution_population = st.number_input("Resolution Population", value=30)
    ss_data = st.text_input("SS Data File", 'ss_data_evn.csv')
    coe = st.number_input("Cost of Energy (euro/MWh)", value=60)
    grid_lifetime = st.number_input("Grid Lifetime (years)", value=40)
    landcover_option = st.text_input("Landcover Option", 'ESA')

    local_database = st.checkbox("Local Database", value=True)
    MITBuilding = st.checkbox("MIT Building", value=True)
    losses = st.checkbox("Losses", value=False)
    reliability_option = st.checkbox("Reliability Option", value=False)
    mg_option = st.checkbox("Microgrid Option", value=False)
    multi_objective_option = st.checkbox("Multi Objective Option", value=False)
    n_line_type = st.number_input("Number of Line Types", value=1)
    MV_coincidence = st.number_input("MV Coincidence", value=0.8)
    mg_types = st.number_input("Microgrid Types", value=1)
    Roads_option = st.checkbox("Roads Option", value=True)
    Rivers_option = st.checkbox("Rivers Option", value=False)
    triangulation_logic = st.checkbox("Triangulation Logic", value=True)
    population_dataset_type = st.text_input("Population Dataset Type", 'buildings')

    return {
        "gisele_folder": gisele_folder,
        "country": country,
        "case_study": case_study,
        "crs": crs,
        "resolution": resolution,
        "voltage": voltage,
        "LV_base_cost": LV_base_cost,
        "load_capita": load_capita,
        "pop_per_household": pop_per_household,
        "resolution_population": resolution_population,
        "ss_data": ss_data,
        "coe": coe,
        "grid_lifetime": grid_lifetime,
        "landcover_option": landcover_option,
        "local_database": local_database,
        "MITBuilding": MITBuilding,
        "losses": losses,
        "reliability_option": reliability_option,
        "mg_option": mg_option,
        "multi_objective_option": multi_objective_option,
        "n_line_type": n_line_type,
        "MV_coincidence": MV_coincidence,
        "mg_types": mg_types,
        "Roads_option": Roads_option,
        "Rivers_option": Rivers_option,
        "triangulation_logic": triangulation_logic,
        "population_dataset_type": population_dataset_type
    }

def run_routing(parameters):
    gisele_folder = parameters["gisele_folder"]
    country = parameters["country"]
    case_study = parameters["case_study"]
    crs = parameters["crs"]
    resolution = parameters["resolution"]
    voltage = parameters["voltage"]
    LV_base_cost = parameters["LV_base_cost"]
    load_capita = parameters["load_capita"]
    pop_per_household = parameters["pop_per_household"]
    resolution_population = parameters["resolution_population"]
    ss_data = parameters["ss_data"]
    coe = parameters["coe"]
    grid_lifetime = parameters["grid_lifetime"]
    landcover_option = parameters["landcover_option"]
    local_database = parameters["local_database"]
    MITBuilding = parameters["MITBuilding"]
    losses = parameters["losses"]
    reliability_option = parameters["reliability_option"]
    mg_option = parameters["mg_option"]
    multi_objective_option = parameters["multi_objective_option"]
    n_line_type = parameters["n_line_type"]
    MV_coincidence = parameters["MV_coincidence"]
    mg_types = parameters["mg_types"]
    Roads_option = parameters["Roads_option"]
    Rivers_option = parameters["Rivers_option"]
    triangulation_logic = parameters["triangulation_logic"]
    population_dataset_type = parameters["population_dataset_type"]

    st.write('0. Clustering Procedures')

    shortProcedureFlag = False
    database = os.path.join(gisele_folder, 'scripts', 'routing_scripts', 'Database')
    study_area_folder = os.path.join(database, country, 'Study_area', 'small_area_5.shp')
    radius = 200
    density = 100

    try:
        output_path_points, output_path_clusters = building_to_cluster_v1(study_area_folder, crs, radius, density, shortProcedureFlag)
    except Exception as e:
        st.error(f"Error processing clustering: {e}")

    st.write("Processing completed")

    
    # 2- New case study creation
    case_study_path = os.path.join('Case studies', case_study)
    if not os.path.exists(case_study_path):
        # Create new folders for the study case
        os.makedirs(case_study_path)
        os.makedirs(os.path.join(case_study_path, 'Input'))
        os.makedirs(os.path.join(case_study_path, 'Output'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Communities'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Microgrid'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Geospatial_Data'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'MILP_output'))
        os.makedirs(os.path.join(case_study_path, 'Output', 'MILP_processed'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'all_data'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'MILP_input'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'all_data', 'Lines_connections'))
        os.makedirs(os.path.join(case_study_path, 'Intermediate', 'Optimization', 'all_data', 'Lines_marked'))

        # Define the path to the configuration file
        configuration_path = os.path.join('scripts', 'routing_scripts', 'Database', 'Configuration.csv')

        # Ensure the destination directory exists
        destination_dir = os.path.join(case_study_path, 'Input')
        os.makedirs(destination_dir, exist_ok=True)

        # Copy the configuration file to the destination directory
        pd.read_csv(configuration_path).to_csv(os.path.join(destination_dir, 'Configuration.csv'))

        # Read the possible connection points and write them in the case study's folder
        Substations = gpd.read_file(os.path.join(database, country, 'con_points_5'))
        Substations_crs = Substations.to_crs(crs)
        Substations['X'] = [geom.xy[0][0] for geom in Substations_crs['geometry']]
        Substations['Y'] = [geom.xy[1][0] for geom in Substations_crs['geometry']]
        Substations.to_file(os.path.join(case_study_path, 'Input', 'substations'))

        # Read the polygon of the study area and write it in the local database
        study_area = gpd.read_file(study_area_folder)
        study_area.to_file(os.path.join(case_study_path, 'Input', 'Study_area'))

        # Read the communities and write them in the local database
        Clusters = gpd.read_file(output_path_clusters)
        Clusters = Clusters.to_crs(crs)
        Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
        for i, row in Clusters.iterrows():
            if row['geometry'].geom_type == 'MultiPolygon':
                Clusters.at[i, 'geometry'] = row['geometry'][0]
        # Clusters.to_file(os.path.join(case_study_path, 'Input', 'Communities_boundaries'))
        Clusters.to_file(os.path.join(database, country, 'Input', 'Communities_boundaries'))
    else:
        destination_path = os.path.join(database, country, 'Input', 'Communities_boundaries', 'Communities_boundaries.shp')
        st.write(output_path_clusters)
        source_gdf = gpd.read_file(output_path_clusters)
        source_gdf.to_file(destination_path)
        Clusters = gpd.read_file(destination_path)
        Clusters = Clusters.to_crs(crs)
        Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
        # st.write(case_study_path)
        study_area = gpd.read_file(os.path.join('scripts', 'routing_scripts', 'Case studies', 'awach555', 'Input', 'Study_area', 'Study_area.shp'))
        Substations = gpd.read_file(os.path.join('scripts', 'routing_scripts', 'Case studies', 'awach555', 'Input', 'substations', 'substations.shp'))

    # Create the grid of points
    st.write('1. CREATE A WEIGHTED GRID OF POINTS')
    df_weighted = qgis_process.create_input_csv(crs, resolution, resolution_population, landcover_option, country, case_study, database, study_area)
    accepted_road_types = [
        'living_street', 'pedestrian', 'primary', 'primary_link', 'secondary', 'secondary_link',
        'tertiary', 'tertiary_link', 'unclassified', 'residential'
    ]
    Road_nodes, Road_lines = create_roads_new(gisele_folder, case_study, Clusters, crs, accepted_road_types, resolution, resolution_population)
    Merge_Roads_GridOfPoints(gisele_folder, case_study)

    # Clustering procedure
    st.write('2. LOCATE SECONDARY SUBSTATIONS INSIDE THE CLUSTERS.')
    el_th = 0.21
    Clusters = Clusters[Clusters['elec acces'] < el_th]
    Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
    destination_path1 = os.path.join(case_study_path, 'Input', 'Communities_boundaries', 'Communities_boundaries_2el.shp')
    Clusters.to_file(destination_path1)
    '''
    # Optimize local area
    start = time.time()
    LAO.optimize(
        crs, country, resolution_population, load_capita, pop_per_household, road_coef, Clusters, case_study, LV_distance,
        ss_data, landcover_option, gisele_folder, roads_weight, run_genetic, max_length_segment, simplify_coef, crit_dist,
        LV_base_cost, population_dataset_type
    )
    end = time.time()
    st.write(f"Optimization Time: {end - start}")

    # Create input for the MILP
    st.write('5. Create the input for the MILP.')
    MILP_Input_creation.create_input(
        gisele_folder, case_study, crs, cable_specs['line1']['line_cost'], resolution, reliability_option, Roads_option,
        simplify_road_coef_outside, Rivers_option, mg_option, mg_types, triangulation_logic
    )

    n_clusters = Clusters.shape[0]

    if reliability_option:
        MILP_Input_creation.create_MILP_input(gisele_folder, case_study, crs, mg_option, MV_coincidence)
    else:
        MILP_Input_creation.create_MILP_input_1way(gisele_folder, case_study, crs, mg_option, MV_coincidence)

    time.sleep(5)

    # Execute the MILP model
    st.write('6. Execute the MILP according to the selected options.')
    start = time.time()
    if not mg_option and reliability_option and n_line_type == 1:
        MILP_models.MILP_without_MG(gisele_folder, case_study, n_clusters, coe, voltage, cable_specs['line1']['resistance'], cable_specs['line1']['reactance'], cable_specs['line1']['Pmax'], cable_specs['line1']['line_cost'])
    elif mg_option and not reliability_option and n_line_type == 1:
        MILP_models.MILP_MG_noRel(gisele_folder, case_study, n_clusters, coe, voltage, cable_specs['line1'])
    elif mg_option and not reliability_option and n_line_type == 2:
        MILP_models.MILP_MG_2cables(gisele_folder, case_study, n_clusters, coe, voltage, cable_specs['line1'], cable_specs['line3'])
    elif not mg_option and not reliability_option and n_line_type == 1 and losses:
        MILP_models.MILP_base_losses2(gisele_folder, case_study, n_clusters, coe, voltage, cable_specs['line1'])
    elif not mg_option and not reliability_option and n_line_type == 1:
        MILP_models.MILP_base(gisele_folder, case_study, n_clusters, coe, voltage, cable_specs['line1'])
    elif not mg_option and not reliability_option and n_line_type == 2:
        MILP_models.MILP_2cables(gisele_folder, case_study, n_clusters, coe, voltage, cable_specs['line1'], cable_specs['line3'])
    elif not mg_option and not reliability_option and n_line_type == 3:
        MILP_models.MILP_3cables(gisele_folder, case_study, n_clusters, coe, voltage, cable_specs['line1'], cable_specs['line2'], cable_specs['line3'])

    end = time.time()
    st.write(f"MILP Execution Time: {end - start}")

    # Process the output from the MILP
    st.write('7. Process MILP output')
    process_output.process(gisele_folder, case_study, crs, mg_option, reliability_option)
    process_output.create_final_output(gisele_folder, case_study)
    if mg_option:
        process_output.analyze(gisele_folder, case_study, coe, mg_option, n_line_type)
    '''
# Main function
def show():
    st.title("Routing Procedures")

    parameters = set_parameters()

    if st.button("Run Routing"):
        run_routing(parameters)

if __name__ == "__main__":
    show()