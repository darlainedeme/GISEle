import streamlit as st

def show():
    st.markdown("""
    # Welcome to Local GISEle 🗺️

    Local GISEle is a comprehensive Geographic Information System (GIS) tool designed to facilitate local area analysis, data collection, visualization, and analysis for various geographical and infrastructural projects. This application integrates multiple data sources, including OpenStreetMap (OSM), Google, and potentially Microsoft in the future, to provide enriched datasets for your projects.
    """)

    st.markdown("## Features")

    with st.expander("📍 **Area Identification**"):
        st.markdown("""
        ### 2.1. Area Selection
        This step allows you to define your area of interest. You can select the area by entering an address, using coordinates, or uploading a GeoJSON file that represents the region you're focusing on.

        ### 2.2. Satellite Timelapse
        Generate and visualize satellite timelapse for your selected area, giving you insight into changes over time within the region.
        """)

    with st.expander("📊 **VANIA**"):
        st.markdown("""
        ### 3.1. Data Retrieve
        In this step, you'll download all the necessary datasets for your selected area. The datasets include information about buildings, roads, points of interest, and more. You'll be able to monitor the download progress and save the results.

        ### 3.2. Data Visualization and Enhancement
        After retrieving the data, visualize the datasets on a map. This section also allows you to manually enhance the datasets by adding or modifying features directly on the map.

        ### 3.3. VANIA Report
        Generate comprehensive reports based on the visualized and enhanced data. These reports are crucial for in-depth analysis and decision-making.
        """)

    with st.expander("⚙️ **GISELE**"):
        st.markdown("""
        ### 4.1. Modelling Parameters
        Define various input parameters required for your analysis, including cost inputs, demand estimation assumptions, and other key factors that influence your GIS project.

        ### 4.2. Clustering
        Perform clustering analysis to identify significant patterns and groupings within the target area. This helps in understanding the distribution of various features and optimizing resources.

        ### 4.3. Demand Estimation
        Estimate the energy demand within your area of interest using different methodologies. This step is essential for planning energy distribution and infrastructure.

        ### 4.4. Mini-grid Sizing
        Based on the estimated demand, optimize the design and size of mini-grids to ensure efficient energy distribution.

        ### 4.5. Routing
        Plan the optimal routes for infrastructure such as roads, power lines, and pipelines, taking into account the geographical features and the existing infrastructure.

        ### 4.6. Results
        View and download the final results of your analysis, including maps, charts, and detailed reports that summarize all the steps and findings of your GIS project.
        """)

    st.markdown("## How to Use")

    with st.expander("📘 **User Guide**"):
        st.markdown("""
        - **Home**: The starting point where you can learn about the tool and its features.
        - **Area Identification**: Begin by selecting the area of interest and generating satellite timelapses.
        - **VANIA**: Retrieve data, visualize, and enhance it. Generate reports for comprehensive analysis.
        - **GISELE**: Set up your analysis parameters, perform clustering, estimate demand, design mini-grids, plan routing, and finally view your results.
        """)

    st.markdown("## Resources 🌐")
    st.markdown("""
    - **Web App**: [Local GISEle](https://gisele.streamlit.app/)
    - **GitHub Repository**: [Local GISEle GitHub](https://github.com/darlainedeme/GISEle)
    - **Documentation**: Detailed documentation and usage instructions can be found in the GitHub repository.
    """)

    st.markdown("## Contact 📬")
    st.markdown("""
    For any questions or support, please contact:
    - **Darlain Edeme**: [E4G Polimi](http://www.e4g.polimi.it/)
    - **GitHub**: [Darlain Edeme](https://github.com/darlainedeme)
    - **Twitter**: [@darlainedeme](https://twitter.com/darlainedeme)
    - **LinkedIn**: [Darlain Edeme](https://www.linkedin.com/in/darlain-edeme)
    
    We hope you find Local GISEle helpful for your GIS projects. Happy analyzing!
    """)

if __name__ == "__main__":
    show()
