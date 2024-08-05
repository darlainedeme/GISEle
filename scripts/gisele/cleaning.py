# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:31:30 2024

@author: EPSla
"""
import shutil 
import os 
def cleaning_MODULE_A(case_study): 
    shutil.rmtree(r'Case studies/'+case_study+'/Output') 
    shutil.rmtree(r'Case studies/'+case_study+'/Intermediate') 
    os.makedirs(r'Case studies/'+case_study+'/Output')
    os.makedirs(r'Case studies/'+case_study+'/Intermediate')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Communities')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Microgrid')
    os.makedirs(r'Case studies/'+case_study+'/Intermediate/Optimization')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Geospatial_Data')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/MILP_output')
    os.makedirs(r'Case studies/' + case_study + '/Output/MILP_processed')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/all_data')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/MILP_input')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/all_data/Lines_connections')
    os.makedirs(r'Case studies/' + case_study +'/Intermediate/Optimization/all_data/Lines_marked')
    
def cleaning_MODULE_B(case_study):     
    shutil.rmtree(r'Case studies/'+case_study+'/Output')
    shutil.rmtree(r'Case studies/' + case_study + '/Intermediate/Communities')
    shutil.rmtree(r'Case studies/' + case_study + '/Intermediate/Microgrid')
    shutil.rmtree(r'Case studies/'+case_study+'/Intermediate/Optimization')
    os.makedirs(r'Case studies/'+case_study+'/Output')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Communities')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Microgrid')
    os.makedirs(r'Case studies/'+case_study+'/Intermediate/Optimization')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/MILP_output')
    os.makedirs(r'Case studies/' + case_study + '/Output/MILP_processed')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/all_data')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/MILP_input')
    os.makedirs(r'Case studies/' + case_study + '/Intermediate/Optimization/all_data/Lines_connections')
    os.makedirs(r'Case studies/' + case_study +'/Intermediate/Optimization/all_data/Lines_marked')