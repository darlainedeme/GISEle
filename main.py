import streamlit as st

# Import your other pages
from scripts import home, area_selection, data_retrieve, buildings_mapping, clustering_modes, data_visualization, costs, summary_analysis, demand_estimation, routing, results, vania

# Display logo at the very top of the sidebar
st.sidebar.image("data/logo.png", width=200)  # Adjust the width as needed

# Define navigation with separators
main_nav = st.sidebar.radio("Navigation", [
    "Home",
    "_________________________",
    "Area Selection", 
    "Data Retrieve", 
    "_________________________",
    "Buildings", 
    "Clustering", 
    "Data Visualization and Enhancement", 
    "VANIA Report",
    "_________________________",
    "Summary Analysis", 
    "Costs", 
    "Demand Estimation", 
    "Routing", 
    "_________________________",
    "Results"
])

# Navigation dictionary for clean code
pages = {
    "Home": home.show,
    "Area Selection": area_selection.show,
    "Data Retrieve": data_retrieve.show,
    "Buildings": buildings_mapping.show,
    "Clustering": clustering_modes.show,
    "Data Visualization and Enhancement": data_visualization.show,
    "VANIA Report": vania.show,
    "Summary Analysis": summary_analysis.show,
    "Costs": costs.show,
    "Demand Estimation": demand_estimation.show,
    "Routing": routing.show,
    "Results": results.show
}

# Display the selected page, excluding separators
if main_nav not in ["_________________________"]:
    pages[main_nav]()
