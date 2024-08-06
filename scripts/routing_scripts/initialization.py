"""
GIS For Electrification (GISEle)
Developed by the Energy Department of Politecnico di Milano
Initialization Code

Code for importing input GIS files, perform the weighting strategy and creating
the initial Point geodataframe.
"""

import os
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from functions import s
import osmnx as ox


def roads_import(geo_df,crs):
    '''
    Download road layers from OpenStreetMaps, and simplify geometries to use
    them for the grid routing algorithms
    :param df:
    :return:
    '''
    geo_df.to_crs(epsg=4326,inplace=True)
    bounds = geo_df.geometry.total_bounds
    print('Downloading roads, from OpenStreetMap..')
    graph = ox.graph_from_bbox(bounds[1], bounds[3],
                               bounds[0], bounds[2], network_type='drive_service')
    ox.save_graph_shapefile(graph,
                            filepath='Output/Datasets/Roads')  # crs is 4326
    #simplify geometry
    roads = gpd.read_file('Output/Datasets/Roads/edges.shp')
    roads_simple = roads.geometry.simplify(tolerance=0.0005)
    roads_simple = roads_simple.to_crs(epsg=int(crs))
    roads_simple.to_file('Output/Datasets/Roads/roads.shp')

def create_landcover_csv(landcover_tif, output_csv):
    """
    Create a CSV file from a land cover TIFF file.
    
    Parameters:
    - landcover_tif: Path to the land cover TIFF file.
    - output_csv: Path to the output CSV file.
    """
    with rasterio.open(landcover_tif) as src:
        data = src.read(1)  # Read the first band
        affine = src.transform
        
        rows, cols = data.shape
        records = []

        for row in range(rows):
            for col in range(cols):
                value = data[row, col]
                if value != src.nodata:
                    x, y = rasterio.transform.xy(affine, row, col)
                    records.append({
                        'ID': f'{row}_{col}',
                        'X': x,
                        'Y': y,
                        'Land_cover': value
                    })

        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"Landcover data has been written to {output_csv}")
        
def weighting(df, resolution, landcover_option):
    """
    Assign weights to all points of a dataframe according to the terrain
    characteristics and the distance to the nearest road.
    :param df: DataFrame containing the points
    :param resolution: Resolution of the dataframe df
    :param landcover_option: Land cover option to use for weighting
    :return df_weighted: DataFrame with weight attributes assigned
    """
    df_weighted = df.dropna(subset=['Elevation'])
    df_weighted.reset_index(drop=True, inplace=True)
    df_weighted['Slope'].fillna(value=0, inplace=True)
    df_weighted['Land_cover'].fillna(method='bfill', inplace=True)
    df_weighted['Land_cover'] = df_weighted['Land_cover'].round(0)
    df_weighted['Weight'] = 0

    print('Weighting the Dataframe..')
    files_folder = os.path.join('scripts', 'routing_scripts', 'Case studies', 'awach555', 'Input', 'Geospatial_Data')
    landcover_csv = os.path.join(files_folder, 'Landcover.csv')

    # Create the landcover CSV if it does not exist
    if not os.path.exists(landcover_csv):
        create_landcover_csv(os.path.join(files_folder, 'LandCover.tif'), landcover_csv)

    landcover_data = pd.read_csv(landcover_csv)

    # Weighting section
    # Slope conditions
    df_weighted['Weight'] = math.e**(0.01732867951 * df_weighted['Slope'])

    # Land cover using the column Other or GLC to compute the weight
    for i, row in landcover_data.iterrows():
        if landcover_option == 'GLC':
            df_weighted.loc[df_weighted['Land_cover'] == row['GLC2000'], 'Weight'] += row['WeightGLC']
        elif landcover_option == 'ESACCI':
            df_weighted.loc[df_weighted['Land_cover'] == row['ESACCI'], 'Weight'] += row['WeightESACCI']
        else:
            df_weighted.loc[df_weighted['Land_cover'] == row['Other'], 'Weight'] += row['WeightOther']

    # Road distance conditions
    df_weighted.loc[df_weighted['Road_dist'] < 1000, 'Weight'] += 5 * df_weighted.loc[df_weighted['Road_dist'] < 1000, 'Road_dist'] / 1000
    df_weighted.loc[df_weighted['Road_dist'] >= 1000, 'Weight'] += 5
    df_weighted.loc[df_weighted['Road_dist'] < resolution / 2, 'Weight'] = 1.5

    # Protected areas condition
    df_weighted.loc[df_weighted['Protected_area'] == True, 'Weight'] += 5

    # Select valid fields
    valid_fields = ['ID', 'X', 'Y', 'Elevation', 'Weight']
    blacklist = [x for x in df_weighted.columns if x not in valid_fields]
    df_weighted.drop(blacklist, axis=1, inplace=True)
    df_weighted.drop_duplicates(['ID'], keep='last', inplace=True)

    print("Cleaning and weighting process completed")
    s()
    return df_weighted

def creating_geodataframe(df_weighted, crs, unit, input_csv, step):
    """
    Based on the input weighted dataframe, creates a geodataframe assigning to
    it the Coordinate Reference System and a Point geometry
    :param df_weighted: Input Point dataframe with weight attributes assigned
    :param crs: Coordinate Reference System of the electrification project
    :param unit: Type of CRS unit used if degree or meters
    :param input_csv: Name of the input file for exporting the geodataframe
    :param step: From which part of GISEle the user is starting
    :return geo_df: Point geodataframe with weight, crs and geometry defined
    :return pop_points: Dataframe containing only the coordinates of the points
    """
    print("Creating the GeoDataFrame..")
    geometry = [Point(xy) for xy in zip(df_weighted['X'], df_weighted['Y'])]
    geo_df = gpd.GeoDataFrame(df_weighted, geometry=geometry,
                              crs=int(crs))

    # - Check crs conformity for the clustering procedure
    # if unit == '0':
    #     print('Attention: the clustering procedure requires measurements in '
    #           'meters. Therefore, your dataset must be reprojected.')
    #     crs = int(input("Provide a valid crs code to which reproject:"))
    #     geo_df['geometry'] = geo_df['geometry'].to_crs(epsg=crs)
    #     print("Done")
    # print("GeoDataFrame ready!")
    # s()
    # if step == 1:
    #     os.chdir(r'Output//Datasets')
    #     geo_df.to_csv(input_csv + "_weighted.csv")
    #     geo_df.to_file(input_csv + "_weighted.shp")
    #     os.chdir(r'..//..')
    #     print("GeoDataFrame exported. You can find it in the output folder.")
    #     l()
    #
    # print("In the considered area there are " + str(
    #     len(geo_df)) + " points, for a total of " +
    #       str(int(geo_df['Population'].sum(axis=0))) +
    #       " people which could gain access to electricity.")

    loc = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
    pop_points = pd.DataFrame(data=loc).values

    return geo_df, pop_points
