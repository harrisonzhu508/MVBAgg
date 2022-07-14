from shapely.geometry import Polygon, Point
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import contextily as ctx
def create_pixel_square(lonlat_centroid, lonlat_sep):
    x1 = [lonlat_centroid[0] - lonlat_sep / 2, lonlat_centroid[1] + lonlat_sep / 2]
    x2 = [lonlat_centroid[0] + lonlat_sep / 2, lonlat_centroid[1] + lonlat_sep / 2]
    x3 = [lonlat_centroid[0] + lonlat_sep / 2, lonlat_centroid[1] - lonlat_sep / 2]
    x4 = [lonlat_centroid[0] - lonlat_sep / 2, lonlat_centroid[1] - lonlat_sep / 2]
    return Polygon([x1, x2, x3, x4])

def create_point(lonlat_centroid):
    return Point([lonlat_centroid[0], lonlat_centroid[1]])

def plot_full_disaggregation(m, gdf_scaled, all_features, scaler_y = None, bag_shapefile = None):
    
    f_mean, _ = m.predict_f(gdf_scaled.loc[:, all_features].values)
    gdf_scaled["f_mean"] = scaler_y.inverse_transform(f_mean)

    gdf_scaled.crs = 4326
    gdf_scaled = gdf_scaled.to_crs(epsg=3857)

    pars = {"size": 50}
    plt.rc("font", **pars)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    gdf_scaled.plot(column="f_mean",ax=ax,cax=cax,
                edgecolor='black', alpha=0.8, legend=True)
    if bag_shapefile is not None:
        bag_shapefile.crs = 4326
        county_shapes = bag_shapefile.to_crs(epsg=3857)
        county_shapes.plot(ax=ax, facecolor="none",
                    edgecolor='black')
    ax.set_aspect("equal")
    ax.set_title("Posterior Mean of $f(x)$")
    ctx.add_basemap(ax)
    ax.axis('off')

def plot_i_disaggregation(m, gdf_scaled, i, features_i, scaler_y = None, bag_shapefile = None):
    
    f_mean, _ = m.predict_f_i(gdf_scaled.loc[:, features_i].values, i)
    gdf_scaled["f_mean"] = scaler_y.inverse_transform(f_mean)

    gdf_scaled = gdf_scaled.to_crs(epsg=3857)
    pars = {"size": 50}
    plt.rc("font", **pars)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    gdf_scaled.plot(column="f_mean",ax=ax,cax=cax,
                edgecolor='black', alpha=0.8, legend=True)
    if bag_shapefile is not None:
        bag_shapefile.crs = 4326
        county_shapes = bag_shapefile.to_crs(epsg=3857)
        county_shapes.plot(ax=ax, facecolor="none",
                    edgecolor='red')
    ax.set_aspect("equal")
    ax.set_title(f"Posterior Mean of $f^{i}(x_{i})$")
    ctx.add_basemap(ax)
    ax.axis('off')