def show():
    # Radio button for method selection
    method = st.radio("Select Clustering Method", ('MIT', 'Standard'), index=0)
    
    with st.expander("Parameters", expanded=False):
        # Input fields for the user to specify paths, CRS, etc.
        crs = st.number_input("CRS (Coordinate Reference System)", value=21095)
        radius = st.number_input("Radius", value=200)
        dens_filter = st.number_input("Density Filter", value=100)
        flag = st.checkbox("Skip Processing", value=False)

    # Initialize session state for clusters and buildings
    if "clusters_gdf" not in st.session_state:
        st.session_state["clusters_gdf"] = None
    if "buildings_df" not in st.session_state:
        st.session_state["buildings_df"] = None

    if method == 'MIT':
        if st.button("Run Clustering"):
            clusters_gdf, buildings_df = building_to_cluster_v1(crs, radius, dens_filter, flag)
            
            # Ensure that the output GeoDataFrames are in the correct CRS
            clusters_gdf = clusters_gdf.to_crs(epsg=crs)
            buildings_df = buildings_df.to_crs(epsg=crs)
            
            st.session_state["clusters_gdf"] = clusters_gdf
            st.session_state["buildings_df"] = buildings_df
            st.success("Clustering completed.")
    else:
        st.write("Standard method not yet implemented.")

    # Display map if clustering has been run
    if st.session_state["clusters_gdf"] is not None and st.session_state["buildings_df"] is not None:
        clusters_gdf = st.session_state["clusters_gdf"]
        buildings_df = st.session_state["buildings_df"]

        # Initialize map centered on the first cluster's centroid
        m = folium.Map(location=[clusters_gdf.geometry.centroid.y.mean(), clusters_gdf.geometry.centroid.x.mean()],
                       zoom_start=12)

        # Add tile layers
        folium.TileLayer('cartodbpositron', name="Positron").add_to(m)
        folium.TileLayer('cartodbdark_matter', name="Dark Matter").add_to(m)
        folium.TileLayer(
            tiles='http://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Maps',
            overlay=False,
            control=True
        ).add_to(m)
        folium.TileLayer(
            tiles='http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Hybrid',
            overlay=False,
            control=True
        ).add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(m)

        # Add clusters polygons to map
        for _, row in clusters_gdf.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, color=row.name: {
                    'fillColor': '#0000ff',
                    'color': '#0000ff',
                    'weight': 1,
                    'fillOpacity': 0.2
                }
            ).add_to(m)

        # Add marker clusters for points
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in buildings_df.iterrows():
            folium.Marker(location=[row.geometry.y, row.geometry.x]).add_to(marker_cluster)

        folium.LayerControl().add_to(m)

        # Display map in Streamlit
        st_data = st_folium(m, width=1400, height=800)

        # Add a button to export the clusters and points
        if st.button("Export Clusters and Points"):
            # Export Clusters
            clusters_gdf.to_file("clusters_export.geojson", driver='GeoJSON')
            # Export Points
            buildings_df.to_file("points_export.geojson", driver='GeoJSON')
            st.success("Export completed! Files saved as 'clusters_export.geojson' and 'points_export.geojson'.")
