import streamlit as st

# Set page configuration
#st.set_page_config(page_title="Local GISEle", page_icon="🗺️")

def show():
    # Add logo
    st.image("data/logo.png", use_column_width=True)

    st.markdown("""
    # Welcome to Local GISEle 🗺️

    Local GISEle is a comprehensive Geographic Information System (GIS) tool designed to facilitate local area analysis, data collection, visualization, and analysis for various geographical and infrastructural projects. This application integrates multiple data sources, including OpenStreetMap (OSM), Google, and potentially Microsoft in the future, to provide enriched datasets for your projects.

    ## Features
    """)
    
    with st.expander("🔍 **Area Selection**"):
        st.markdown("""
        Choose an area by address, coordinates, or upload a GeoJSON file.
        """)

    with st.expander("🌐 **Data Retrieval**"):
        st.markdown("""
        Download and visualize data such as buildings, roads, points of interest, and more.
        """)

    with st.expander("🏠 **Buildings Analysis**"):
        st.markdown("""
        Enhance building datasets using multiple sources.
        """)

    with st.expander("📊 **Clustering**"):
        st.markdown("""
        Identify and evaluate clusters within the target area.
        """)

    with st.expander("🎨 **Data Visualization and Enhancement**"):
        st.markdown("""
        Visualize and manually enhance various datasets.
        """)

    with st.expander("💸 **Cost Inputs**"):
        st.markdown("""
        Define various cost inputs for your analysis.
        """)

    with st.expander("📋 **Summary Analysis**"):
        st.markdown("""
        Combine all information into a comprehensive summary with maps and charts.
        """)

    with st.expander("⚡ **Demand Estimation**"):
        st.markdown("""
        Estimate energy demand using different methodologies.
        """)

    with st.expander("🔌 **Mini-grid Sizing**"):
        st.markdown("""
        Optimize mini-grid designs.
        """)

    with st.expander("🔋 **Grid Extension**"):
        st.markdown("""
        Estimate the extension of the national grid to the study area.
        """)

    with st.expander("📈 **Results**"):
        st.markdown("""
        View and download comprehensive results.
        """)

    st.markdown("""
    ## How to Use
    """)
    
    with st.expander("📘 **User Guide**"):
        st.markdown("""
        1. **Home**: This is the welcome page where you can get an overview of the application.
        2. **Area Selection**: Navigate to the 'Area Selection' page to select your area of interest by address, coordinates, or by uploading a GeoJSON file.
        3. **Data Retrieve**: Use this page to download all necessary datasets for your selected area. Visualize the progress and download the results as a zip file.
        4. **Buildings**: Analyze and enhance building data using OSM as the reference dataset and Google data for enhancements.
        5. **Clustering**: Perform clustering analysis to identify clusters within the target area.
        6. **Data Visualization and Enhancement**: Visualize and manually enhance the downloaded datasets.
        7. **Costs**: Define various cost inputs required for your analysis.
        8. **Summary Analysis**: View a comprehensive summary of all data with maps and charts.
        9. **Demand Estimation**: Estimate the energy demand using different methodologies.
        10. **Mini-grid Sizing**: Optimize the mini-grid design based on user inputs.
        11. **Grid**: Estimate the grid extension needed to connect the study area to the national grid.
        12. **Results**: View and download the final results of your analysis.
        """)

    st.markdown("""
    ## Resources 🌐
    - **Web App**: [Local GISEle](https://gisele.streamlit.app/)
    - **GitHub Repository**: [Local GISEle GitHub](https://github.com/darlainedeme/GISEle)
    - **Documentation**: Detailed documentation and usage instructions can be found in the GitHub repository.
    
    ## Contact 📬
    For any questions or support, please contact:
    - **Darlain Edeme**: [E4G Polimi](http://www.e4g.polimi.it/)
    - **GitHub**: [Darlain Edeme](https://github.com/darlainedeme)
    - **Twitter**: [@darlainedeme](https://twitter.com/darlainedeme)
    - **LinkedIn**: [Darlain Edeme](https://www.linkedin.com/in/darlain-edeme)
    
    We hope you find Local GISEle helpful for your GIS projects. Happy analyzing!
    """)

# Display the home page
if __name__ == "__main__":
    show()
