import streamlit as st

# Import your other pages
from scripts import home, area_selection, data_retrieve, buildings_mapping, clustering_modes, data_visualization, costs, summary_analysis, demand_estimation, routing, results, vania

# Display logo at the very top of the sidebar
st.sidebar.image("data/logo.png", width=200)  # Adjust the width as needed

# Define navigation
main_nav = st.sidebar.radio("Navigation", [
    "Home", "Area Selection", "Data Retrieve", "Buildings", 
    "Clustering", "Data Visualization and Enhancement", 
    "VANIA Report", "Summary Analysis", "Costs", "Demand Estimation", 
    "Routing", "Results"
])

# Navigation dictionary for clean code
pages = {
    "Home": home.show,
    "Area Selection": area_selection.show,
    "Data Retrieve": data_retrieve.show,
    "Buildings": buildings_mapping.show,
    "Clustering": clustering_modes.show,
    "Data Visualization and Enhancement": data_visualization.show,
    "VANIA Report": vania.show,  # Add the VANIA report page here
    "Summary Analysis": summary_analysis.show,
    "Costs": costs.show,
    "Demand Estimation": demand_estimation.show,
    "Routing": routing.show,
    "Results": results.show
}

# Display the selected page
if main_nav in pages:
    pages[main_nav]()
