import planetary_computer as pc
import pystac_client
import requests
import rioxarray as riox
import fiona
import rasterio
from rasterio.mask import mask
import streamlit as st
from shapely.geometry import mapping

def download_nighttime_lights_mpc(polygon, nighttime_lights_path, clipped_nighttime_lights_path):
    try:
        aoi = mapping(polygon)

        daterange = {"interval": ["2019-01-01", "2019-12-31"]}

        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

        search = catalog.search(filter_lang="cql2-json", filter={
            "op": "and",
            "args": [
                {"op": "s_intersects", "args": [{"property": "geometry"}, aoi]},
                {"op": "anyinteracts", "args": [{"property": "datetime"}, daterange]},
                {"op": "=", "args": [{"property": "collection"}, "hrea"]}
            ]
        })

        items = search.item_collection()
        if not items:
            st.write("No nighttime lights data found for the selected area.")
            return None

        first_item = items[0]
        asset = first_item.assets["lightscore"]

        # Download the nighttime lights data
        signed_asset = pc.sign(asset)
        data = riox.open_rasterio(signed_asset.href)
        data.values[data.values < 0] = np.nan

        # Save the downloaded raster
        data.rio.to_raster(nighttime_lights_path)

        # Clip the downloaded raster to the polygon
        with fiona.open('data/input/selected_area.geojson', "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        with rasterio.open(nighttime_lights_path) as src:
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(clipped_nighttime_lights_path, "w", **out_meta) as dest:
            dest.write(out_image)

        st.write("Nighttime lights data downloaded and clipped to the selected area.")
        return clipped_nighttime_lights_path
    except Exception as e:
        st.error(f"Error downloading nighttime lights data: {e}")
        return None
