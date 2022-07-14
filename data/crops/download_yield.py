import ee 
# Trigger the authentication flow.
# ee.Authenticate()

# Initialize the library.
ee.Initialize()

import geopandas as gpd
from tqdm import trange
import requests
from os import path
import pandas as pd 
import matplotlib.pyplot as plt

def extract_poly_coords(geom):
    """
    From https://gis.stackexchange.com/questions/300857/how-to-import-the-geometry-feature-collection-which-is-in-geojson-format-into-go
    """
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        interior_coords = []
        for interior in geom.interiors:
            interior_coords += interior.coords[:]
    elif geom.type == 'MultiPolygon':
        exterior_coords = []
        interior_coords = []
        for part in geom.geoms:
            epc = extract_poly_coords(part)  # Recursive call
            exterior_coords += epc['exterior_coords']
            interior_coords += epc['interior_coords']
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.type))
    return {'exterior_coords': exterior_coords,
            'interior_coords': interior_coords}

def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame.
    Modification of 
    https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api-guiattard
    """
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    print(df.head())

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    # Keep the columns of interest.
    df = df[['datetime', "longitude", "latitude", *list_of_bands]]

    return df

def ee_nodate_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame.
    Modification of 
    https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api-guiattard
    """
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Keep the columns of interest.
    df = df[["longitude", "latitude", *list_of_bands]]

    return df

for year in [2015, 2017]:
    gdf = gpd.read_file("yield_data_final.geojson")
    gdf["key"] = gdf["County"] + "-" + gdf["State"]
    counties = pd.read_csv("counties-states.csv")
    gdf = gpd.GeoDataFrame(gdf[gdf["key"].isin(counties["key"])])

    """## Crop mask"""

    crop_mask = ee.ImageCollection('USDA/NASS/CDL').filterDate(ee.Date("{year}-01-01"), ee.Date("{year}-12-31"))
    crop_mask = crop_mask.select('cropland')

    # just look at soybeans
    crop_mask = crop_mask.first().eq(5)

    crop_mask = crop_mask.reduceResolution(
        reducer=ee.Reducer.max(),
        maxPixels=30000
    )


    """## MODIS data
    """

    root_dir = "MODIS"
    bands = ["NDVI", "EVI"]
    date_start = f"{year}-04-01"
    date_end = f"{year}-10-31"
    scale = 1000 # for computational convenience
    # for testing purposes
    # gdf = gpd.read_file("illinois-counties.geojson")
    collection = ee.ImageCollection("MODIS/006/MOD13Q1").filterDate(
            ee.Date(date_start), ee.Date(date_end)
        ).select(bands)
    collection = collection.map(lambda img: img.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=30000
    ))
    modisProjection = collection.first().projection()
    crop_mask_modis = crop_mask.reproject(crs=modisProjection)
    collection = collection.map(lambda img: img.updateMask(crop_mask_modis))

    iterator = trange(len(gdf))
    max_pixels = 0
    min_pixels = 10000
    for i in iterator:
        geometry = ee.Geometry.MultiPolygon(extract_poly_coords(gdf.geometry.values[i])["exterior_coords"])
        collection_area_of_interest = collection.getRegion(geometry, scale=scale).getInfo()
        df_area_of_interest = ee_array_to_df(collection_area_of_interest, bands)
        df_area_of_interest["County"] = gdf.County.values[i]
        df_area_of_interest["State"] = gdf.State.values[i]
        filename = f"MOD13Q1_{date_start}_{date_end}_{gdf.name.values[i]}-{gdf.state_name.values[i]}.csv"
        num_pixels = df_area_of_interest[df_area_of_interest["datetime"]==f"{year}-04-07"].shape[0]
        max_pixels = max(max_pixels, num_pixels)
        min_pixels = min(min_pixels, num_pixels)
        df_area_of_interest.to_csv(f"{root_dir}/{filename}")
        iterator.set_description(f"Location: {i}/{len(gdf)}, num_pixels: {num_pixels}, max_pixels: {max_pixels}, min_pixels: {min_pixels}")

    """## Gridmet"""

    root_dir = "GRIDMET" 
    bands = ["tmmx", "tmmn", "pr", "rmax", "rmin", "eto"]
    date_start = f"{year}-04-01"
    date_end = f"{year}-10-31"
    scale = 4638.3 # for computational convenience
    # for testing purposes
    # gdf = gpd.read_file("illinois-counties.geojson")
    collection = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate(
            ee.Date(date_start), ee.Date(date_end)
        ).select(bands)
    # no masking because not many pixels anyway and temperature shouldn't vary too much 
    # whether it's cropland or not2
    gridmetProjection = collection.first().projection();
    crop_mask_gridmet = crop_mask.reproject(crs=gridmetProjection)
    collection = collection.map(lambda img: img.updateMask(crop_mask_gridmet))

    iterator = trange(len(gdf))
    max_pixels = 0
    min_pixels = 10000
    for i in iterator:
        geometry = ee.Geometry.MultiPolygon(extract_poly_coords(gdf.geometry.values[i])["exterior_coords"])
        collection_area_of_interest = collection.getRegion(geometry, scale=scale).getInfo()
        df_area_of_interest = ee_array_to_df(collection_area_of_interest, bands)
        df_area_of_interest["County"] = gdf.County.values[i]
        df_area_of_interest["State"] = gdf.State.values[i]
        # reformat datetime
        df_area_of_interest["datetime"] = df_area_of_interest["datetime"].apply(lambda i: str(i)[:10])
        filename = f"GRIDMET_{date_start}_{date_end}_{gdf.name.values[i]}-{gdf.state_name.values[i]}.csv"
        num_pixels = df_area_of_interest[df_area_of_interest["datetime"]==f"{year}-04-07"].shape[0]
        max_pixels = max(max_pixels, num_pixels)
        min_pixels = min(min_pixels, num_pixels)
        df_area_of_interest.to_csv(f"{root_dir}/{filename}")
        iterator.set_description(f"Location: {i}/{len(gdf)}, num_pixels: {num_pixels}, max_pixels: {max_pixels}, min_pixels: {min_pixels}")


"""## Download Latlon"""

root_dir = "latlon"

bands = ["cropland"]
scale = 500 # for computational convenience
# date doesn't matter here
collection = ee.ImageCollection('USDA/NASS/CDL').filterDate(ee.Date(f"{year}-01-01"), ee.Date(f"{year}-12-31"))
collection = collection.select('cropland')
collection = collection.map(lambda img: img.eq(5))
collection = collection.map(lambda img: img.reduceResolution(
    reducer=ee.Reducer.mean(),
    maxPixels=30000
))

iterator = trange(len(gdf))
max_pixels = 0
min_pixels = 10000
for i in iterator:
    geometry = ee.Geometry.MultiPolygon(extract_poly_coords(gdf.geometry.values[i])["exterior_coords"])
    collection_area_of_interest = collection.getRegion(geometry, scale=scale).getInfo()
    df_area_of_interest = ee_nodate_array_to_df(collection_area_of_interest, bands)
    df_area_of_interest["County"] = gdf.County.values[i]
    df_area_of_interest["State"] = gdf.State.values[i]
    filename = f"latlon_{gdf.name.values[i]}-{gdf.state_name.values[i]}.csv"
    df_area_of_interest.to_csv(f"{root_dir}/{filename}")
    iterator.set_description(f"Location: {i}/{len(gdf)}, shape {df_area_of_interest.shape}")

