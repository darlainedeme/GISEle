import streamlit as st
st.set_page_config(layout="wide")

from scripts import home, area_selection, data_retrieve, buildings, clustering_modes, data_visualization, costs, summary_analysis, demand_estimation, routing, results

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
    clustering.show()
elif main_nav == "Data Visualization and Enhancement":
    data_visualization.show()
elif main_nav == "Summary Analysis":
    summary_analysis.show()
elif main_nav == "Costs":
    costs.show()
elif main_nav == "Demand Estimation":
    demand_estimation.show()
elif main_nav == "Routing":
    routing.show()
elif main_nav == "Results":
    results.show()