import streamlit as st
st.set_page_config(layout="wide")

from scripts import home, area_selection, data_retrieve, buildings, clustering_modes, data_visualization, costs, summary_analysis, demand_estimation, routing, results
from scripts.routing import routing_preparation, local_area_optimization, create_milp_input, execute_milp, process_milp_output

# Define navigation
main_nav = st.sidebar.radio("Navigation", [
    "Home", "Area Selection", "Data Retrieve", "Buildings", 
    "Clustering", "Data Visualization and Enhancement", 
    "Summary Analysis", "Costs", "Demand Estimation", 
    "Routing", "Results"
])

if main_nav == "Home":
    home.show()
elif main_nav == "Area Selection":
    area_selection.show()
elif main_nav == "Data Retrieve":
    data_retrieve.show()
elif main_nav == "Buildings":
    buildings.show()
elif main_nav == "Clustering":
    clustering_modes.show()
elif main_nav == "Data Visualization and Enhancement":
    data_visualization.show()
elif main_nav == "Summary Analysis":
    summary_analysis.show()
elif main_nav == "Costs":
    costs.show()
elif main_nav == "Demand Estimation":
    demand_estimation.show()
elif main_nav == "Routing":
    routing_nav = st.sidebar.radio("Routing Steps", [
        "Routing Preparation", "Local Area Optimization", 
        "Create MILP Input", "Execute MILP", "Process MILP Output"
    ])
    if routing_nav == "Routing Preparation":
        routing_preparation.show()
    elif routing_nav == "Local Area Optimization":
        local_area_optimization.show()
    elif routing_nav == "Create MILP Input":
        create_milp_input.show()
    elif routing_nav == "Execute MILP":
        execute_milp.show()
    elif routing_nav == "Process MILP Output":
        process_milp_output.show()
elif main_nav == "Results":
    results.show()
