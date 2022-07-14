# %%
import sys 
sys.path.append("../..")
import pandas as pd 
import geopandas as gpd
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as ctx
from src.bagData import BagData, create_tensorflow_iterator, create_bag_multiresolution_dictionary
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import gpflow 
from scipy.stats import sem
import json
import random 
import time 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from src.svgp import MVBAgg
from src.util import optimize_adam, optimize_natgrad_adam, save_model, load_model
from src.plot_util import create_pixel_square,create_point
from sklearn.cluster import KMeans
seed = 0


# %%
# read in user-specified counties and states to study
dates = ['04-07', '04-23', '05-09', '05-25',
       '06-10', '06-26', '07-12', '07-28',
       '08-13', '08-29', '09-14', '09-30',
       '10-16']
counties_states = pd.read_csv("../../data/crops/counties-states.csv")
States = ["OHIO","ILLINOIS", "IOWA", "MICHIGAN", "MISSOURI", "NORTH DAKOTA", "SOUTH DAKOTA"]
col_latlon = ["longitude", "latitude"]
col_modis = [f"EVI_{date}" for date in dates]
col_gridmet = [f"pr_{date}" for date in dates] + [f"tmmx_{date}" for date in dates]

# read in and take subset of yield data
df_yield = pd.read_csv("../../data/crops/soybean_yield_2015_2017.csv")
df_yield["key"] = df_yield["County"] + "-" + df_yield["State"]
df_yield = df_yield[df_yield["key"].isin(counties_states["key"])]
df_yield.rename(columns={"Year": "year"}, inplace=True)
df_yield = df_yield.drop_duplicates(["County", "State", "year"])
df_yield["Value"] = np.log(df_yield["Value"])

# read in latlon
df_latlon = pd.read_csv("../../data/crops/processed_covariates/latlon-500_data_500points.csv")
df_latlon["key"] = df_latlon["County"] + "-" + df_latlon["State"]
df_latlon = df_latlon[df_latlon["key"].isin(counties_states["key"])]
df_latlon = df_latlon.drop_duplicates(["County", "State", "longitude", "latitude"])

# read in and take subset of features data
df_modis = pd.read_csv("../../data/crops/processed_covariates/MOD13Q1-1000_data.csv")
df_modis["key"] = df_modis["County"] + "-" + df_modis["State"]
df_modis = df_modis[df_modis["key"].isin(counties_states["key"])]
features = ["year", "key", "County", "State", "longitude", "latitude"] + [f"EVI_{date}" for date in dates]
df_modis = df_modis[features]
df_modis = df_modis.dropna()
df_modis = df_modis.drop_duplicates(["County", "State", "longitude", "latitude", "year"])

# read in and take subset of features data
df_gridmet = pd.read_csv("../../data/crops/processed_covariates/GRIDMET_data.csv")
df_gridmet["key"] = df_gridmet["County"] + "-" + df_gridmet["State"]
df_gridmet = df_gridmet[df_gridmet["key"].isin(counties_states["key"])]

# %%


# %%
# col_gridmet = list(df_gridmet.columns[225:-1])
features = ["key", "year", "County", "State", "longitude", "latitude"] + col_gridmet
df_gridmet = df_gridmet[features]
df_gridmet = df_gridmet.drop_duplicates(["County", "State", "longitude", "latitude", "year"])

print(f"num_yields {df_yield.shape[0]}")

assert df_yield.shape[0] == 768

## create inducing points
gridmet_resolution = 0.04166816804999707


# %% [markdown]
# ## Experimental Setup with Data-Agg
df_modis_joined = df_modis.copy()
df_modis_joined = df_modis_joined[df_modis_joined["year"]==2015]
df_modis_joined["geometry"] = df_modis_joined[["longitude", "latitude"]].apply(lambda x: create_point(x), axis=1)
df_modis_joined = gpd.GeoDataFrame(df_modis_joined)
df_modis_joined = df_modis_joined.drop(["key", "County", "State"], axis=1)
df_modis_joined.drop(["longitude", "latitude"], inplace=True, axis=1)
df_gridmet_joined = df_gridmet.copy()
df_gridmet_joined = df_gridmet_joined[df_gridmet_joined["year"]==2015]
df_gridmet_joined["geometry"] = df_gridmet[["longitude", "latitude"]].apply(lambda x: create_pixel_square(x, gridmet_resolution), axis=1)
df_gridmet_joined = df_gridmet_joined.drop(["latitude", "longitude"], axis=1)
df_for_inducing = gpd.sjoin(gpd.GeoDataFrame(df_gridmet_joined), gpd.GeoDataFrame(df_modis_joined))
# %%
kf = KFold(n_splits=5, random_state=seed, shuffle=True)
output_types=(
    tf.int64, 
    tf.int64, 
    tf.int64, 
    tf.int64, 
    tf.float64, 
    tf.float64, 
    tf.float64, 
    tf.float64, 
    tf.float64, 
    tf.float64, 
    tf.float64, 
    tf.float64, 
    tf.float64
)
keys = list(set(df_yield[df_yield["year"]==2015].key).intersection(set(df_yield[df_yield["year"]==2017].key)))
keys.sort()

## create inducing points (continued)
latlon_cols = ["longitude", "latitude"]
modis_cols = [f"EVI_{date}" for date in dates]
pr_cols = [f"pr_{date}" for date in dates] 
tmmx_cols = [f"tmmx_{date}" for date in dates] 
all_features = latlon_cols + modis_cols + pr_cols + tmmx_cols
for fold, (train_index, test_index) in tqdm(enumerate(kf.split(keys))):
    train_keys = [keys[key] for key in train_index]
    df_for_inducing_fold = df_for_inducing[df_for_inducing["key"].isin(train_keys)]
    scaler_x = StandardScaler().fit(df_for_inducing_fold.loc[:, all_features].values)
    df_for_inducing_fold.loc[:, all_features] = scaler_x.transform(df_for_inducing_fold.loc[:, all_features].values)

    df_Z = pd.DataFrame(columns=["key"] + ["longitude", "latitude"] + [f"EVI_{date}" for date in dates] + col_gridmet)
    df_Z["key"] = train_keys
    all_features = ["longitude", "latitude"] + [f"EVI_{date}" for date in dates] + col_gridmet
    kmeans = KMeans(1)
    for key in tqdm(train_keys):
        X_tmp = df_for_inducing_fold.loc[df_for_inducing_fold["key"] == key, all_features].values
        Z_tmp = kmeans.fit(X_tmp).cluster_centers_
        df_Z.loc[df_Z["key"]==key, all_features] = Z_tmp
    df_Z.to_csv(f"../../data/crops/processed_covariates/df_Z_MVBAgg_fold{fold}.csv", index=False)

# %%
RMSE = []
MAPE  = []
LL  = []
training_time = []
iterations = 20000
lr_adam = 0.001
num_minibatch = 50
latlon_cols = ["longitude", "latitude"]
modis_cols = [f"EVI_{date}" for date in dates]
pr_cols = [f"pr_{date}" for date in dates] 
tmmx_cols = [f"tmmx_{date}" for date in dates] 
all_features = latlon_cols + modis_cols + pr_cols + tmmx_cols
num_resolutions = 4
# write down index of the features
col_index_space = [0, 1]
col_index_modis = [2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12,13,14]
col_index_pr = [15+i for i in range(13)]
col_index_tmmx = [28+i for i in range(13)]
col_index_all = col_index_space + col_index_modis + col_index_pr + col_index_tmmx
max_pixels = [500, 100, 100, 100]

for fold, (train_index, test_index) in tqdm(enumerate(kf.split(keys))):
    county_keys = [keys[key] for key in train_index]
    df_Z = pd.read_csv(f"../../data/crops/processed_covariates/df_Z_MVBAgg_fold{fold}.csv")
    Z = df_Z[df_Z["key"].isin(county_keys)][all_features].values
    
    X_latlon_train = df_latlon[df_latlon["key"].isin(county_keys)].loc[:, latlon_cols].values
    X_latlon_test = df_latlon[df_latlon["key"].isin(county_keys)].loc[:, latlon_cols].values
    X_modis_train = df_modis[df_modis["key"].isin(county_keys) & (df_modis["year"]==2015)].loc[:, modis_cols].values
    X_modis_test = df_modis[df_modis["key"].isin(county_keys) & (df_modis["year"]==2017)].loc[:, modis_cols].values
    X_pr_train = df_gridmet[df_gridmet["key"].isin(county_keys) & (df_gridmet["year"]==2015)].loc[:, pr_cols].values
    X_pr_test = df_gridmet[df_gridmet["key"].isin(county_keys) & (df_gridmet["year"]==2017)].loc[:, pr_cols].values
    X_tmmx_train = df_gridmet[df_gridmet["key"].isin(county_keys) & (df_gridmet["year"]==2015)].loc[:, tmmx_cols].values
    X_tmmx_test = df_gridmet[df_gridmet["key"].isin(county_keys) & (df_gridmet["year"]==2017)].loc[:, tmmx_cols].values
    y_train = df_yield[df_yield["key"].isin(county_keys) & (df_yield["year"]==2015)].Value.values[:, None]
    y_test = df_yield[df_yield["key"].isin(county_keys) & (df_yield["year"]==2017)].Value.values[:, None]

    scaler_y = StandardScaler().fit(y_train)
    y_train = scaler_y.transform(y_train)

    scaler_x = StandardScaler().fit(X_latlon_train)
    X_latlon_train = scaler_x.transform(X_latlon_train)
    X_latlon_test = scaler_x.transform(X_latlon_test)
    
    scaler_x = StandardScaler().fit(X_modis_train)
    X_modis_train = scaler_x.transform(X_modis_train)
    X_modis_test = scaler_x.transform(X_modis_test)

    scaler_x = StandardScaler().fit(X_pr_train)
    X_pr_train = scaler_x.transform(X_pr_train)
    X_pr_test = scaler_x.transform(X_pr_test)
    
    scaler_x = StandardScaler().fit(X_tmmx_train)
    X_tmmx_train = scaler_x.transform(X_tmmx_train)
    X_tmmx_test = scaler_x.transform(X_tmmx_test)

    df_features_list = [df_latlon, df_modis[df_modis["year"]==2015], df_gridmet[df_gridmet["year"]==2015], df_gridmet[df_gridmet["year"]==2015]]

    train_dict = create_bag_multiresolution_dictionary(
        [X_latlon_train, X_modis_train, X_pr_train, X_tmmx_train], 
        y_train, 
        [df_features[df_features["key"].isin(county_keys)] for df_features in df_features_list],
        df_yield[df_yield["key"].isin(county_keys) & (df_yield["year"]==2015)],
        county_keys,
        max_pixels
    )
    df_features_list = [df_latlon, df_modis[df_modis["year"]==2017], df_gridmet[df_gridmet["year"]==2017], df_gridmet[df_gridmet["year"]==2017]]
    test_dict = create_bag_multiresolution_dictionary(
        [X_latlon_test, X_modis_test, X_pr_test, X_tmmx_test], 
        y_test, 
        [df_features[df_features["key"].isin(county_keys)] for df_features in df_features_list],
        df_yield[df_yield["key"].isin(county_keys) & (df_yield["year"]==2017)],
        county_keys,
        max_pixels
    )

    train_bags = BagData(bag_data=train_dict, bags_metadata=county_keys)
    test_bags = BagData(bag_data=test_dict, bags_metadata=county_keys)

    ds_train = create_tensorflow_iterator(train_bags.gen_bags, output_types=output_types)


    # fit and train GP regression model
    k_space = gpflow.kernels.Matern32(active_dims=col_index_space)
    k_modis = gpflow.kernels.RBF(active_dims=col_index_modis)
    k_pr = gpflow.kernels.RBF(active_dims=col_index_pr)
    k_tmmx = gpflow.kernels.RBF(active_dims=col_index_tmmx)

    k = k_space + k_modis + k_pr + k_tmmx
    m = MVBAgg(kernel=k, likelihood=gpflow.likelihoods.Gaussian(), inducing_variable=Z, num_resolution=num_resolutions, num_data=train_bags.num_bags)
    gpflow.set_trainable(m.inducing_variable, False)
    
    print("Begin Training")
    t0 = time.time()
    logf = optimize_natgrad_adam(m, ds_train, num_data=train_bags.num_bags, iterations=iterations, minibatch_size=num_minibatch, learning_rate=lr_adam)
    t1 = time.time()
    save_model(m, save_path="../../results/usa_crops", model_name=f"MVBAgg_fold{fold}")

    # make predictions
    y_pred = np.zeros(test_bags.num_bags)
    y_std = np.zeros(test_bags.num_bags)
    for i, bag in enumerate(test_bags.bags):
        f_mean, f_var = m.predict_aggregated(test_bags[bag])
        y_pred[i] = f_mean[0][0]
        y_std[i] = np.sqrt(f_var[0][0])
    loglik = np.mean(m.predict_log_density(y_pred[:, None], y_std[:, None]**2, scaler_y.transform(test_bags.y)))
    y_pred_rescaled = scaler_y.inverse_transform(y_pred[:, None])

    lower = np.reshape((1.96 * y_std)*scaler_y.scale_, y_test.shape)
    upper = np.reshape((1.96 * y_std)*scaler_y.scale_, y_test.shape)
    errors = np.concatenate((lower, upper), axis=1)
    errors = errors.T
    
    # compute metrics 
    loglik = np.mean(m.predict_log_density(y_pred[:, None], y_std[:, None]**2, scaler_y.transform(test_bags.y)))
    RMSE.append(np.sqrt(np.mean((y_pred_rescaled - test_bags.y)**2)))
    MAPE.append(np.mean(np.abs(( y_pred_rescaled - test_bags.y) / test_bags.y)))
    LL.append(loglik)
    training_time.append(t1-t0)

    # plot predictions
    plt.figure(figsize=(8,8))
    plt.scatter(y_pred_rescaled, test_bags.y, color="red")
    plt.plot(np.linspace(-100,100,201), np.linspace(-100,100,201), color="black")

    plt.errorbar(
        y_pred_rescaled[:,0],
        test_bags.y[:,0],
        xerr=errors,
        fmt="o",
        ls="none",
        capsize=5,
        markersize=4,
        color="blue",
        alpha=0.2
        )
    plt.xlim((test_bags.y.min()-1, test_bags.y.max()+1))
    plt.ylim((test_bags.y.min()-1, test_bags.y.max()+1))
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.savefig(f"../../results/usa_crops/MVBAgg_fold{fold}.png")
    plt.savefig(f"../../results/usa_crops/MVBAgg_fold{fold}.pdf")
    plt.close()

    print(f"MAPE", MAPE)
    print(f"RMSE", RMSE)
    print(f"LL", LL)
    print(f"Training Time", training_time)

json_file = json.dumps({"CV-RMSE": sum(RMSE) / 5, "CV-MAPE": sum(MAPE) / 5, 
                       "CV-sd-RMSE": sem(RMSE), "CV-sd-MAPE": sem(MAPE), "CV-LL": sum(LL) / 5, "CV-sd-LL": sem(LL),
                        "Training Time": sum(training_time)/5, "Training Time se": sem(training_time)}
                       )
f = open(f"../../results/usa_crops/MVBAgg.json", "w")
f.write(json_file)
f.close()
