import streamlit as st

st.set_page_config(layout="wide")

# Import your other pages
from scripts import home, area_selection, timelapse, data_retrieve, data_visualization, vania, modelling_parameters, demand_estimation, minigrid_sizing, routing, results
from scripts.gisele_scripts import clustering_modes, case_study_creation

# Display logo at the very top of the sidebar
st.sidebar.image("data/logo.png", width=100)  # Adjust the width as needed

# Define the main sections with emojis
main_sections = {
    "🏠 Home": ["Home"],
    "📍 Area Identification": ["Area Selection", "Satellite Timelapse"],
    "📊 VANIA": ["Data Retrieve", "Data Visualization and Enhancement", "VANIA Report"],
    "⚙️ GISELE": ["Modelling Parameters", "Clustering", "Case Study", "Demand Estimation", "Mini-grid Sizing", "Routing", "Results"]
}


# Select the main section
main_section = st.sidebar.radio("Navigation", list(main_sections.keys()))

# Visual separation and styling for subpages
if main_section == "🏠 Home":
    # st.sidebar.markdown("**🏠 Home**", unsafe_allow_html=True)
    subpage = "Home"
    pages = {
        "Home": home.show
    }

elif main_section == "📍 Area Identification":
    st.sidebar.markdown("<hr style='border: none; border-bottom: 2px solid #ccc;'>", unsafe_allow_html=True)
    # st.sidebar.markdown("**📍 Area Identification**", unsafe_allow_html=True)
    subpage = st.sidebar.radio("Area Identification", main_sections[main_section], index=0, key="area_id")
    pages = {
        "Area Selection": area_selection.show,
        "Satellite Timelapse": timelapse.show
    }

elif main_section == "📊 VANIA":
    st.sidebar.markdown("<hr style='border: none; border-bottom: 2px solid #ccc;'>", unsafe_allow_html=True)
    # st.sidebar.markdown("**📊 VANIA**", unsafe_allow_html=True)
    subpage = st.sidebar.radio("VANIA", main_sections[main_section], index=0, key="vania")
    pages = {
        "Data Retrieve": data_retrieve.show,
        "Data Visualization and Enhancement": data_visualization.show,
        "VANIA Report": vania.show
    }

elif main_section == "⚙️ GISELE":
    st.sidebar.markdown("<hr style='border: none; border-bottom: 2px solid #ccc;'>", unsafe_allow_html=True)
    # st.sidebar.markdown("**⚙️ GISELE**", unsafe_allow_html=True)
    subpage = st.sidebar.radio("GISELE", main_sections[main_section], index=0, key="gisele")
    pages = {
        "Modelling Parameters": modelling_parameters.show,
        "Clustering": clustering_modes.show,
        "Case Study": case_study_creation.case_study_creation_function,
        "Demand Estimation": demand_estimation.show,
        "Mini-grid Sizing": minigrid_sizing.show,
        "Routing": routing.show,
        "Results": results.show
    }

# Display the selected page
if subpage in pages:
    pages[subpage]()
