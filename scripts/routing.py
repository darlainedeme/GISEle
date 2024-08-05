import os
import base64
import io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import plotly.graph_objs as go
import pyutilib.subprocess.GlobalData
import time
import rasterio
import sys

def set_stuff(): 
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

    # 0 - Setting input
    ############# INPUT ELECTRICAL PARAMETERS #############

    # Parameters for MV cables
    cable_specs = {
        'line1': {"resistance": 0.306, "reactance": 0.33, "Pmax": 6.69, "line_cost": 10000},
        'line2': {"resistance": 0.4132, "reactance": 0.339, "Pmax": 2, "line_cost": 7800},
        'line3': {"resistance": 0.59, "reactance": 0.35, "Pmax": 0.8, "line_cost": 6700}
    }

    ###########STARTING THE SCRIPT###########

    # Location information
    gisele_folder = os.getcwd()
    villages_file = 'Communities/Communities.shp'
    country = 'Uganda'
    case_study = 'awach555'

    #######################################################################
    ###############   Electrical and geographical parameters ##############
    #######################################################################

    crs = 21095
    resolution = 200
    voltage = 110  # [kV]
    LV_base_cost = 10000
    load_capita = 0.6  # [kW]
    pop_per_household = 5
    resolution_population = 30

    # Data used for local area optimization
    max_length_segment = resolution_population * 1.5
    simplify_coef = 5
    crit_dist = simplify_coef / 2
    LV_distance = 500
    ss_data = 'ss_data_evn.csv'
    simplify_road_coef_inside = 5
    simplify_road_coef_outside = 30
    road_coef = 2
    roads_weight = 0.3
    coe = 60  # euro/MWh of electrical energy supplied
    grid_lifetime = 40  # years
    landcover_option = 'ESA'

    #######################################################################
    ###############   Flags & Options #####################################
    #######################################################################

    local_database = True
    MITBuilding = True
    losses = False
    reliability_option = False
    mg_option = False
    multi_objective_option = False
    n_line_type = 1
    MV_coincidence = 0.8
    mg_types = 1
    Roads_option = True
    Rivers_option = False
    triangulation_logic = True
    population_dataset_type = 'buildings'

def show():

    #%%
    st.write('0. Clustering Procedures')
    '''
    if not local_database:
        database = r'C:\Users\alekd\Politecnico di Milano\Silvia Corigliano - Gisele shared\8.Case_Study'
        cluster_folder = r'Database/' + country + '/Clusters/villages_majaua.shp'
        substations_folder = r'Database/' + country + '/connection_points.shp'
        study_area_folder = r'Database/' + country + '/Study_area/majaua.shp'
    elif MITBuilding:
        shortProcedureFlag = False
        database = gisele_folder + '/Database'
        study_area_folder = database + '/' + country + '/Study_area/small_area_5.shp'
        radius = 200
        density = 100
        output_path_points, output_path_clusters = building_to_cluster_v1(study_area_folder, crs, radius, density, shortProcedureFlag)
        cluster_folder = output_path_clusters
        substations_folder = database + '/' + country + '/con_points_5'
    else:
        database = gisele_folder + '/Database'
        cluster_folder = database + '/' + country + '/' + villages_file
        substations_folder = database + '/' + country + '/con_points'
        study_area_folder = database + '/' + country + '/Study_area/Study_area_test1.shp'

    # 2- New case study creation
    case_study_path = r'Case studies/' + case_study
    if not os.path.exists(case_study_path):
        # Create new folders for the study case
        os.makedirs(case_study_path)
        os.makedirs(case_study_path + '/Input')
        os.makedirs(case_study_path + '/Output')
        os.makedirs(case_study_path + '/Intermediate')
        os.makedirs(case_study_path + '/Intermediate/Communities')
        os.makedirs(case_study_path + '/Intermediate/Microgrid')
        os.makedirs(case_study_path + '/Intermediate/Optimization')
        os.makedirs(case_study_path + '/Intermediate/Geospatial_Data')
        os.makedirs(case_study_path + '/Intermediate/Optimization/MILP_output')
        os.makedirs(case_study_path + '/Output/MILP_processed')
        os.makedirs(case_study_path + '/Intermediate/Optimization/all_data')
        os.makedirs(case_study_path + '/Intermediate/Optimization/MILP_input')
        os.makedirs(case_study_path + '/Intermediate/Optimization/all_data/Lines_connections')
        os.makedirs(case_study_path + '/Intermediate/Optimization/all_data/Lines_marked')

        # Copy the Configuration file from the general input
        pd.read_csv(r'general_input/Configuration.csv').to_csv(case_study_path + '/Input/Configuration.csv')

        # Read the possible connection points and write them in the case study's folder
        Substations = gpd.read_file(substations_folder)
        Substations_crs = Substations.to_crs(crs)
        Substations['X'] = [geom.xy[0][0] for geom in Substations_crs['geometry']]
        Substations['Y'] = [geom.xy[1][0] for geom in Substations_crs['geometry']]
        Substations.to_file(case_study_path + '/Input/substations')

        # Read the polygon of the study area and write it in the local database
        study_area = gpd.read_file(study_area_folder)
        study_area.to_file(case_study_path + '/Input/Study_area')

        # Read the communities and write them in the local database
        Clusters = gpd.read_file(cluster_folder)
        Clusters = Clusters.to_crs(crs)
        Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
        for i, row in Clusters.iterrows():
            if row['geometry'].geom_type == 'MultiPolygon':
                Clusters.at[i, 'geometry'] = row['geometry'][0]
        Clusters.to_file(case_study_path + '/Input/Communities_boundaries')
    else:
        destination_path = case_study_path + '/Input/Communities_boundaries/Communities_boundaries.shp'
        source_gdf = gpd.read_file(cluster_folder)
        source_gdf.to_file(destination_path)
        Clusters = gpd.read_file(destination_path)
        Clusters = Clusters.to_crs(crs)
        Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
        study_area = gpd.read_file(case_study_path + '/Input/Study_area/Study_area.shp')
        Substations = gpd.read_file(case_study_path + '/Input/substations/substations.shp')

    # Create the grid of points
    print('1. CREATE A WEIGHTED GRID OF POINTS')
    df_weighted = qgis_process.create_input_csv(crs, resolution, resolution_population, landcover_option, country, case_study, database, study_area)
    accepted_road_types = [
        'living_street', 'pedestrian', 'primary', 'primary_link', 'secondary', 'secondary_link',
        'tertiary', 'tertiary_link', 'unclassified', 'residential'
    ]
    Road_nodes, Road_lines = create_roads_new(gisele_folder, case_study, Clusters, crs, accepted_road_types, resolution, resolution_population)
    Merge_Roads_GridOfPoints(gisele_folder, case_study)

    # Clustering procedure
    print('2. LOCATE SECONDARY SUBSTATIONS INSIDE THE CLUSTERS.')
    el_th = 0.21
    Clusters = Clusters[Clusters['elec acces'] < el_th]
    Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
    destination_path1 = case_study_path + '/Input/Communities_boundaries/Communities_boundaries_○2el.shp'
    Clusters.to_file(destination_path1)

    # Optimize local area
    start = time.time()
    LAO.optimize(
        crs, country, resolution_population, load_capita, pop_per_household, road_coef, Clusters, case_study, LV_distance,
        ss_data, landcover_option, gisele_folder, roads_weight, run_genetic, max_length_segment, simplify_coef, crit_dist,
        LV_base_cost, population_dataset_type
    )
    end = time.time()
    print(f"Optimization Time: {end - start}")

    # Create input for the MILP
    print('5. Create the input for the MILP.')
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
    print('6. Execute the MILP according to the selected options.')
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
    print(f"MILP Execution Time: {end - start}")

    # Process the output from the MILP
    print('7. Process MILP output')
    process_output.process(gisele_folder, case_study, crs, mg_option, reliability_option)
    process_output.create_final_output(gisele_folder, case_study)
    if mg_option:
        process_output.analyze(gisele_folder, case_study, coe, mg_option, n_line_type)
    '''