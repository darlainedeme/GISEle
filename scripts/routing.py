import os
import base64
import io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import plotly.graph_objs as go
from shapely.geometry import Point
import pyutilib.subprocess.GlobalData
import time
import rasterio 
from gisele_scripts.cleaning import *

from gisele_scripts.functions import *
from gisele_scripts.functions2 import *
from gisele_scripts import initialization, clustering, processing, collecting, \
    optimization, results, grid, branches

from gisele_scripts import QGIS_processing_polygon as qgis_process
from gisele_scripts import Local_area_optimization as LAO
from gisele_scripts import MILP_Input_creation,MILP_models, process_output, grid_routing,Secondary_substations 
from gisele_scripts.OpenEnergyMapMIT_v1 import building_to_cluster_v1
#%%
#0 - Setting input
############# INPUT ELECTRICAL PARAMETERS #############

# parameters for the MV cable that is used
# generic cable
# resistance = 0.42  # [ohm/km]
# reactance = 0.39  # [ohm/km]
# Pmax = 11  # [MVA]

#CABLES FOR LESOTHO
#line1 - 94mm2 from panda power ( closest to 100mm2 - HARE )
resistance = 0.306
reactance= 0.33
Pmax = 6.69 # MVA, in amperes it is 350A
line_cost = 5800 +4200 # not clear if this is for one phase or for all 3
line1 = {"resistance":resistance,"reactance":reactance,"Pmax":Pmax,"line_cost":line_cost}
#line2 70mm2 from pandapower (closest to 70mm2 - MINK )
resistance2 = 0.4132
reactance2 = 0.339
Pmax2 = 2 # MVA, in amperes it is 290A   5.54 MVA
line_cost2 = 3600 + 4200 # assuming 4200 for poles etc.
line2 = {"resistance":resistance2,"reactance":reactance2,"Pmax":Pmax2,"line_cost":line_cost2}
#line3 48mm2 from pandapower (closest to 48mm2 - FOX)
resistance3 = 0.59
reactance3 = 0.35
Pmax3 = 0.8 # MVA, in amperes it is 290A   4MVA
line_cost3 = 2500+4200 # we are assuming 4200 for poles etc.
line3 = {"resistance":resistance3,"reactance":reactance3,"Pmax":Pmax3,"line_cost":line_cost3}
#cable of 48mm2
#resistance = 0.321  # [ohm/km]
##reactance = 0.372  # [ohm/km]
##Pmax = 4  # [MVA]
#line_cost = 10000  # Cost of MV voltage feeders [€/km]

#new cable - 70mm2
# resistance = 0.41
# reactance = 0.306
# Pmax=6
# line_cost=10000
# #new cable - 70mm2
# resistance = 0.305
# reactance = 0.35
# Pmax=6
# line_cost=10000
#second cable - this cable is used only in case the MILP option with 2 different lines is chosen
#resistance2 = 0.35
#reactance2 = 0.38
#Pmax2 = 1 #2
#line_cost2 = 8000 # Cost of MV voltage feeders [€/km]

# #cable of 70 mm2
# resistance = 0.413  # [ohm/km]
# reactance = 0.36  # [ohm/km]
# Pmax = 5.5  # [MVA]



###########STARTING THE SCRIPT###########

#Location information
gisele_folder=os.getcwd()
#villages_file = 'Villages_areas.geojson'
villages_file = 'Communities/Communities.shp'
country  = 'Uganda' #Mozambique
case_study='awach555' #majawa #Isola_Giglio2

#######################################################################
###############   Electrical and geographical parameters ##############
#######################################################################

crs = 21095 #32632 - Italy , 22287 - Lesotho, 21095 - Uganda
resolution = 200 #240

voltage = 110# [kV] 15kV - Italy , 11kV Lesotho
LV_base_cost=10000 # for the LV we are not checking electrical parameters
load_capita= 0.6 #0.2 #kW♠
pop_per_household=5
resolution_population = 30 # this should be automatic from the raster 

#data used for the local area optimization
max_length_segment = resolution_population*1.5
simplify_coef = 5
crit_dist = simplify_coef/2 

LV_distance=500 # Maximum length of the LV network.
ss_data = 'ss_data_evn.csv' # folder in which the costs for substations can be found
simplify_road_coef_inside = 5 # in meters, used for the routing inside the clusters.
simplify_road_coef_outside = 30 # in meters, used for creating connections among clusters/substations.
road_coef = 2
roads_weight=0.3

coe = 60  # euro/MWh of electrical energy supplied
grid_lifetime = 40 #years
landcover_option= 'ESA'#'ESACCI' 


#######################################################################
###############   Flags & Options ##################################### 
####################################################################### 

local_database =True 
MITBuilding = True
losses=False
reliability_option=False
mg_option = False
multi_objective_option = False
n_line_type= 1
MV_coincidence=0.8
mg_types =1 #if more than one mg for each cluster needs to be computed, with different reliability levels (needed for multiobjective: mg_types=3)


Roads_option=True # Boolean stating whether we want to use the roads for the steiner tree and dijkstra.
Rivers_option=False
n_line_type=1
run_genetic=False
triangulation_logic = True
population_dataset_type = 'buildings'



#%%
print('0. Clustering Procedures')
if local_database ==False: 
    
    database= r'C:\Users\alekd\Politecnico di Milano\Silvia Corigliano - Gisele shared\8.Case_Study'
    #cluster_folder = database+'\Lesotho\case_study_3_clusters/Villages_areas_3_copy.geojson'
    cluster_folder = r'Database/'+country+'/Clusters/villages_majaua.shp'
    substations_folder =r'Database/'+country+'/connection_points.shp'
    study_area_folder =r'Database/'+country+'/Study_area/majaua.shp'

    
elif MITBuilding == True:    
    shortProcedureFlag = False
    database =gisele_folder + '/Database'
    study_area_folder = database + '/' + country + '/Study_area/small_area_5.shp'
    radius = 200 #Not over 500 meters•
    density = 100
    output_path_points , output_path_clusters = building_to_cluster_v1(study_area_folder , crs, radius, density, shortProcedureFlag)
    cluster_folder = output_path_clusters
    substations_folder = database + '/' + country +'/con_points_5'

else:
    database =gisele_folder + '/Database'
    cluster_folder = database + '/' + country + '/'+villages_file
    substations_folder = database + '/' + country +'/con_points'
    study_area_folder = database + '/' + country + '/Study_area/Study_area_test1.shp'

# 2- New case study creation
if not os.path.exists(r'Case studies/'+case_study): # if this is a new project, create the starting point for the analysis
    # Create new folders for the study case
    os.makedirs(r'Case studies/'+case_study)
    os.makedirs(r'Case studies/'+case_study+'/Input')
    os.makedirs(r'Case studies/'+case_study+'/Output')
    os.makedirs(r'Case studies/'+case_study+'/Intermediate')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Communities  ')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Microgrid')
    os.makedirs(r'Case studies/'+case_study+'/Intermediate/Optimization')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Geospatial_Data')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/MILP_output')
    os.makedirs(r'Case studies/' + case_study + '/Output/MILP_processed')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/all_data')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/MILP_input')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/all_data/Lines_connections')
    os.makedirs(r'Case studies/' + case_study +'/Intermediate/Optimization/all_data/Lines_marked')
    # Copy the Configuration file from the general input
    pd.read_csv(r'general_input/Configuration.csv').to_csv(r'Case studies/'+case_study+'/Input/Configuration.csv')
    # Read the possible connection points and write them in the case study's folder
    Substations = gpd.read_file(substations_folder)
    Substations_crs = Substations.to_crs(crs)
    #Substations['X'] = [Substations_crs['geometry'].values[i][0].xy[0][0] for i in range(Substations.shape[0])]
    #Substations['Y'] = [Substations_crs['geometry'].values[i][0].xy[1][0] for i in range(Substations.shape[0])]
    Substations['X'] = [Substations_crs['geometry'].values[i].xy[0][0] for i in range(Substations.shape[0])]
    Substations['Y'] = [Substations_crs['geometry'].values[i].xy[1][0] for i in range(Substations.shape[0])]
    Substations.to_file(r'Case studies/' + case_study + '/Input/substations')
    # Read the polygon of the study area and write it in the local database.
    study_area = gpd.read_file(study_area_folder)
    study_area.to_file(r'Case studies/'+case_study+'/Input/Study_area')
    # Read the communities and write them in the local database
    Clusters = gpd.read_file(cluster_folder)
    Clusters = Clusters.to_crs(crs)
    Clusters['cluster_ID'] = [*range(1, Clusters.shape[0] + 1)]
    for i, row in Clusters.iterrows(): # this is just in case one of the polygons is saved as a MP with just 1 polygon
        if row['geometry'].geom_type == 'MultiPolygon':
            Clusters.loc[i, 'geometry'] = row['geometry'][0]
    Clusters.to_file(r'Case studies/'+case_study+'/Input/Communities_boundaries') 
    
    
else: # not a new project, just read the files from the local folder 
    destination_path = r'Case studies/'+case_study+'/Input/Communities_boundaries/Communities_boundaries.shp'
    source_gdf = gpd.read_file(cluster_folder)
    source_gdf.to_file(destination_path)
    Clusters = gpd.read_file(r'Case studies/'+case_study+'/Input/Communities_boundaries/Communities_boundaries.shp') 
    Clusters = Clusters.to_crs(crs)
    Clusters['cluster_ID'] = [*range(1, Clusters.shape[0] + 1)]
    study_area = gpd.read_file(r'Case studies/'+case_study+'/Input/Study_area/Study_area.shp')
    Substations = gpd.read_file(r'Case studies/' + case_study + '/Input/substations/substations.shp')   
    
# cleaning_MODULE_A(case_study)
#%%
'''Create the grid of points'''
print('1. CREATE A WEIGHTED GRID OF POINTS')
df_weighted = qgis_process.create_input_csv(crs,resolution,resolution_population,landcover_option,country,case_study,database,study_area)
accepted_road_types = ['living_street', 'pedestrian', 'primary', 'primary_link', 'secondary', 'secondary_link',
                          'tertiary', 'tertiary_link', 'unclassified','residential']
Road_nodes,Road_lines = create_roads_new(gisele_folder, case_study, Clusters,crs, accepted_road_types,resolution,resolution_population)
Merge_Roads_GridOfPoints(gisele_folder,case_study) 

# cleaning_MODULE_B(case_study)
#%%
''' CLUSTERING PROCEDURE'''
'''For each cluster, perform further aglomerative clustering, locate secondary substations and perform MV grid routing'''
print('2. LOCATE SECONDARY SUBSTATIONS INSIDE THE CLUSTERS.') 
el_th = 0.21
Clusters = Clusters[Clusters['elec acces'] < el_th]  
Clusters['cluster_ID'] = [*range(1, Clusters.shape[0] + 1)]  
destination_path1 = r'Case studies/'+case_study+'/Input/Communities_boundaries/Communities_boundaries_○2el.shp'
Clusters.to_file(destination_path1)
# Clusters = Clusters[Clusters['cluster_ID']==18] 
#%%
start = time.time()
LAO.optimize(crs,country, resolution_population, load_capita, pop_per_household, road_coef, Clusters, case_study, LV_distance,
            ss_data,landcover_option,gisele_folder,roads_weight,run_genetic, max_length_segment,simplify_coef, crit_dist,LV_base_cost,population_dataset_type)
end = time.time() 
print(end-start) 
#%% 
'''A prior elimination of clusters''' 

#Cost of conductors to the closest village 
#Cost of PS 
#Cost of energy 

#VS 

    
#Cost for MG

#%%
'''Create the input for the MILP'''
# print('5. Create the input for the MILP.')
MILP_Input_creation.create_input(gisele_folder,case_study,crs,line_cost,resolution,reliability_option,Roads_option,
                          simplify_road_coef_outside, Rivers_option,mg_option,mg_types,triangulation_logic)

n_clusters = Clusters.shape[0]

# if multi_objective_option:
#     calculate_mg_multiobjective(gisele_folder, case_study, crs)
# elif mg_option:
#      calculate_mg(gisele_folder, case_study, crs,mg_types)
if reliability_option:
    MILP_Input_creation.create_MILP_input(gisele_folder,case_study,crs,mg_option,MV_coincidence) # all the lines are from i-j and j-i, only positive powers to consider reliability
else:
    #pass
    MILP_Input_creation.create_MILP_input_1way(gisele_folder,case_study,crs,mg_option,MV_coincidence) # only i-j, without reliability
time.sleep(5) 

#%%
'''Here do an alternative of the MILP_Input_Creation, in which each "substation is actually the entire cluster"'''
'''Execute the desired MILP model'''
print('6. Execute the MILP according to the selected options.')
coe = 60
start = time.time()
if mg_option == False and reliability_option==True and n_line_type==1:
    MILP_models.MILP_without_MG(gisele_folder,case_study,n_clusters,coe,voltage,resistance,reactance,Pmax,line_cost)
elif mg_option == True and reliability_option==False and n_line_type==1:
    MILP_models.MILP_MG_noRel(gisele_folder, case_study, n_clusters, coe, voltage,line1)
elif mg_option == True and reliability_option==False and n_line_type==2:
    MILP_models.MILP_MG_2cables(gisele_folder, case_study, n_clusters, coe, voltage,line1,line3)
elif mg_option == False and reliability_option == False and n_line_type ==1 and losses==True:
    MILP_models.MILP_base_losses2(gisele_folder, case_study, n_clusters, coe, voltage, line1)
elif mg_option == False and reliability_option == False and n_line_type ==1:
    MILP_models.MILP_base(gisele_folder, case_study, n_clusters, coe, voltage, line1)

elif mg_option == False and reliability_option==False and n_line_type==2:
    MILP_models.MILP_2cables(gisele_folder, case_study, n_clusters, coe, voltage, line1,line3)

elif mg_option == False and reliability_option==False and n_line_type==3:
    MILP_models.MILP_3cables(gisele_folder, case_study, n_clusters, coe, voltage, line1,line2,line3)


end = time.time()
#print(end-start)
'''Process the output from the MILP'''
#%%
print('7. Process MILP output')
reliability_option=False
process_output.process(gisele_folder,case_study,crs,mg_option,reliability_option)
process_output.create_final_output(gisele_folder, case_study)
if mg_option == True:
    process_output.analyze(gisele_folder,case_study,coe,mg_option,n_line_type)
# end = time.time()
# print(end-start) 

