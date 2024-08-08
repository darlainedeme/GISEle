import ee
import json
import os
import warnings
import datetime
import fiona
import geopandas as gpd
import folium
import streamlit as st
import geemap.colormaps as cm
import geemap.foliumap as geemap
from datetime import date
from shapely.geometry import Polygon

warnings.filterwarnings("ignore")

# Authenticate and initialize Earth Engine
@st.cache_data
def ee_authenticate(token_name="EARTHENGINE_TOKEN"):
    geemap.ee_initialize(token_name=token_name)

@st.cache_data
def uploaded_file_to_gdf(data):
    import tempfile
    import os
    import uuid

    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(data.getbuffer())

    if file_path.lower().endswith(".kml"):
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(file_path, driver="KML")
    else:
        gdf = gpd.read_file(file_path)

    return gdf

def app():
    today = date.today()

    row1_col1, row1_col2 = st.columns([2, 1])

    if st.session_state.get("zoom_level") is None:
        st.session_state["zoom_level"] = 4

    st.session_state["ee_asset_id"] = None
    st.session_state["bands"] = None
    st.session_state["palette"] = None
    st.session_state["vis_params"] = None

    with row1_col1:
        ee_authenticate(token_name="EARTHENGINE_TOKEN")
        m = geemap.Map(
            basemap="HYBRID",
            plugin_Draw=True,
            Draw_export=True,
            locate_control=True,
            plugin_LatLngPopup=False,
        )
        m.add_basemap("ROADMAP")

    with row1_col2:
        # Ensure the selected area is available
        selected_area_path = 'data/3_user_uploaded_data/selected_area.geojson'
        if os.path.exists(selected_area_path):
            with open(selected_area_path) as f:
                geojson_data = json.load(f)
            gdf = gpd.read_file(selected_area_path)

            if not gdf.empty:
                centroid = gdf.geometry.unary_union.centroid
                st.session_state["latitude"] = centroid.y
                st.session_state["longitude"] = centroid.x
                st.session_state["geojson_data"] = geojson_data

                # Display the map with the selected area
                m.add_gdf(gdf, "Selected Area")
                m.set_center(centroid.x, centroid.y, 12)
            else:
                st.error("Selected area file is empty or not valid GeoJSON.")
        else:
            st.error("No selected area found. Please select an area first.")

    with row1_col2:
        collection = st.selectbox(
            "Select a satellite image collection: ",
            [
                # "Any Earth Engine ImageCollection",
                "Landsat TM-ETM-OLI Surface Reflectance",
                # "Sentinel-2 MSI Surface Reflectance",
                # "Geostationary Operational Environmental Satellites (GOES)",
                # "MODIS Vegetation Indices (NDVI/EVI) 16-Day Global 1km",
                # "MODIS Gap filled Land Surface Temperature Daily",
                # "MODIS Ocean Color SMI",
                # "USDA National Agriculture Imagery Program (NAIP)",
            ],
            index=0,
        )

        roi = geemap.gdf_to_ee(gdf, geodesic=False) if gdf is not None else None
        st.session_state["roi"] = roi

        if collection == "Any Earth Engine ImageCollection":
            keyword = st.text_input("Enter a keyword to search (e.g., MODIS):", "")
            if keyword:
                assets = geemap.search_ee_data(keyword)
                ee_assets = []
                for asset in assets:
                    if "ee_id_snippet" in asset and asset["ee_id_snippet"].startswith("ee.ImageCollection"):
                        ee_assets.append(asset)

                asset_titles = [x["title"] for x in ee_assets]
                dataset = st.selectbox("Select a dataset:", asset_titles)
                if len(ee_assets) > 0:
                    st.session_state["ee_assets"] = ee_assets
                    st.session_state["asset_titles"] = asset_titles
                    index = asset_titles.index(dataset)
                    ee_id = ee_assets[index]["id"]
                else:
                    ee_id = ""

                if dataset is not None:
                    with st.expander("Show dataset details", False):
                        index = asset_titles.index(dataset)
                        html = geemap.ee_data_html(st.session_state["ee_assets"][index])
                        st.markdown(html, True)
            else:
                ee_id = ""

            asset_id = st.text_input("Enter an ee.ImageCollection asset ID:", ee_id)

            if asset_id:
                with st.expander("Customize band combination and color palette", True):
                    try:
                        col = ee.ImageCollection.load(asset_id)
                        st.session_state["ee_asset_id"] = asset_id
                    except:
                        st.error("Invalid Earth Engine asset ID.")
                        st.session_state["ee_asset_id"] = None
                        return

                    img_bands = col.first().bandNames().getInfo()
                    if len(img_bands) >= 3:
                        default_bands = img_bands[:3][::-1]
                    else:
                        default_bands = img_bands[:]
                    bands = st.multiselect(
                        "Select one or three bands (RGB):", img_bands, default_bands
                    )
                    st.session_state["bands"] = bands

                    if len(bands) == 1:
                        palette_options = st.selectbox(
                            "Color palette",
                            cm.list_colormaps(),
                            index=2,
                        )
                        palette_values = cm.get_palette(palette_options, 15)
                        palette = st.text_area(
                            "Enter a custom palette:",
                            palette_values,
                        )
                        st.write(
                            cm.plot_colormap(cmap=palette_options, return_fig=True)
                        )
                        st.session_state["palette"] = json.loads(
                            palette.replace("'", '"')
                        )

                    if bands:
                        vis_params = st.text_area(
                            "Enter visualization parameters",
                            "{'bands': ["
                            + ", ".join([f"'{band}'" for band in bands])
                            + "]}",
                        )
                    else:
                        vis_params = st.text_area(
                            "Enter visualization parameters",
                            "{}",
                        )
                    try:
                        st.session_state["vis_params"] = json.loads(
                            vis_params.replace("'", '"')
                        )
                        st.session_state["vis_params"]["palette"] = st.session_state[
                            "palette"
                        ]
                    except Exception as e:
                        st.session_state["vis_params"] = None
                        st.error(
                            f"Invalid visualization parameters. It must be a dictionary."
                        )

        # Add other collections as needed...

    with row1_col1:
        m.to_streamlit(height=600)

    with row1_col2:
        if collection in [
            "Landsat TM-ETM-OLI Surface Reflectance",
            "Sentinel-2 MSI Surface Reflectance",
        ]:

            if collection == "Landsat TM-ETM-OLI Surface Reflectance":
                sensor_start_year = 1984
                timelapse_title = "Landsat Timelapse"
                timelapse_speed = 1
            elif collection == "Sentinel-2 MSI Surface Reflectance":
                sensor_start_year = 2015
                timelapse_title = "Sentinel-2 Timelapse"
                timelapse_speed = 1

            with st.form("submit_landsat_form"):
                roi = st.session_state.get("roi")
                out_gif = 'results/timelapse.gif'

                title = st.text_input(
                    "Enter a title to show on the timelapse: ", timelapse_title
                )
                RGB = st.selectbox(
                    "Select an RGB band combination:",
                    [
                        "Red/Green/Blue",
                        "NIR/Red/Green",
                        "SWIR2/SWIR1/NIR",
                        "NIR/SWIR1/Red",
                        "SWIR2/NIR/Red",
                        "SWIR2/SWIR1/Red",
                        "SWIR1/NIR/Blue",
                        "NIR/SWIR1/Blue",
                        "SWIR2/NIR/Green",
                        "SWIR1/NIR/Red",
                        "SWIR2/NIR/SWIR1",
                        "SWIR1/NIR/SWIR2",
                    ],
                    index=9,
                )

                frequency = st.selectbox(
                    "Select a temporal frequency:",
                    ["year", "quarter", "month"],
                    index=0,
                )

                with st.expander("Customize timelapse"):
                    speed = st.slider("Frames per second:", 1, 30, timelapse_speed)
                    dimensions = st.slider(
                        "Maximum dimensions (Width*Height) in pixels", 768, 2000, 768
                    )
                    progress_bar_color = st.color_picker(
                        "Progress bar color:", "#0000ff"
                    )
                    years = st.slider(
                        "Start and end year:",
                        sensor_start_year,
                        today.year,
                        (sensor_start_year, today.year),
                    )
                    months = st.slider("Start and end month:", 1, 12, (1, 12))
                    font_size = st.slider("Font size:", 10, 50, 30)
                    font_color = st.color_picker("Font color:", "#ffffff")
                    apply_fmask = st.checkbox(
                        "Apply fmask (remove clouds, shadows, snow)", True
                    )
                    font_type = st.selectbox(
                        "Select the font type for the title:",
                        ["arial.ttf", "alibaba.otf"],
                        index=0,
                    )
                    fading = st.slider(
                        "Fading duration (seconds) for each frame:", 0.0, 3.0, 0.0
                    )
                    mp4 = st.checkbox("Save timelapse as MP4", True)

                empty_text = st.empty()
                empty_image = st.empty()
                empty_fire_image = st.empty()
                empty_video = st.container()
                submitted = st.form_submit_button("Submit")
                if submitted:
                    empty_text.text("Computing... Please wait...")

                    start_year = years[0]
                    end_year = years[1]
                    start_date = str(months[0]).zfill(2) + "-01"
                    end_date = str(months[1]).zfill(2) + "-30"
                    bands = RGB.split("/")

                    if collection == "Landsat TM-ETM-OLI Surface Reflectance":
                        out_gif = geemap.landsat_timelapse(
                            roi=roi,
                            out_gif=out_gif,
                            start_year=start_year,
                            end_year=end_year,
                            start_date=start_date,
                            end_date=end_date,
                            bands=bands,
                            apply_fmask=apply_fmask,
                            frames_per_second=speed,
                            dimensions=768,
                            overlay_data=None,
                            overlay_color="black",
                            overlay_width=1,
                            overlay_opacity=1,
                            frequency=frequency,
                            date_format=None,
                            title=title,
                            title_xy=("2%", "90%"),
                            add_text=True,
                            text_xy=("2%", "2%"),
                            text_sequence=None,
                            font_type=font_type,
                            font_size=font_size,
                            font_color=font_color,
                            add_progress_bar=True,
                            progress_bar_color=progress_bar_color,
                            progress_bar_height=5,
                            loop=0,
                            mp4=mp4,
                            fading=fading,
                        )
                    elif collection == "Sentinel-2 MSI Surface Reflectance":
                        out_gif = geemap.sentinel2_timelapse(
                            roi=roi,
                            out_gif=out_gif,
                            start_year=start_year,
                            end_year=end_year,
                            start_date=start_date,
                            end_date=end_date,
                            bands=bands,
                            apply_fmask=apply_fmask,
                            frames_per_second=speed,
                            dimensions=768,
                            overlay_data=None,
                            overlay_color="black",
                            overlay_width=1,
                            overlay_opacity=1,
                            frequency=frequency,
                            date_format=None,
                            title=title,
                            title_xy=("2%", "90%"),
                            add_text=True,
                            text_xy=("2%", "2%"),
                            text_sequence=None,
                            font_type=font_type,
                            font_size=font_size,
                            font_color=font_color,
                            add_progress_bar=True,
                            progress_bar_color=progress_bar_color,
                            progress_bar_height=5,
                            loop=0,
                            mp4=mp4,
                            fading=fading,
                        )

                    if out_gif is not None and os.path.exists(out_gif):
                        empty_text.text(
                            "Right click the GIF to save it to your computer👇"
                        )
                        empty_image.image(out_gif)

                        out_mp4 = out_gif.replace(".gif", ".mp4")
                        if mp4 and os.path.exists(out_mp4):
                            with empty_video:
                                st.text(
                                    "Right click the MP4 to save it to your computer👇"
                                )
                                st.video(out_gif.replace(".gif", ".mp4"))

                    else:
                        empty_text.error(
                            "Something went wrong. You probably requested too much data. Try reducing the ROI or timespan."
                        )

# Run the app
if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        st.error(f"An error occurred: {e}")
