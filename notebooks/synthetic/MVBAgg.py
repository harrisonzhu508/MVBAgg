import sys 
sys.path.append("../..")
import pandas as pd 
import geopandas as gpd
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as ctx
from src.bagData import BagData, create_tensorflow_iterator, create_bag_multiresolution_dictionary
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
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

# read in user-specified counties and states to study
dates = ['04-07']
counties_states = pd.read_csv("../../data/crops/counties-states.csv")
States = ["OHIO","ILLINOIS", "IOWA", "MICHIGAN", "MISSOURI", "NORTH DAKOTA", "SOUTH DAKOTA"]
col_latlon = ["longitude", "latitude"]
col_modis = [f"EVI_{date}" for date in dates]
col_gridmet = [f"pr_{date}" for date in dates] + [f"tmmx_{date}" for date in dates]

# read in latlon
df_latlon = pd.read_csv("../../data/crops/processed_covariates/latlon-500_data_500points.csv")
df_latlon["key"] = df_latlon["County"] + "-" + df_latlon["State"]
df_latlon = df_latlon[df_latlon["key"].isin(counties_states["key"])]
df_latlon = df_latlon.drop_duplicates(["County", "State", "longitude", "latitude"])

# read in and take subset of features data
df_modis = pd.read_csv("../../data/crops/processed_covariates/MOD13Q1-1000_data.csv")
df_modis = df_modis[df_modis["year"]==2015]
df_modis["key"] = df_modis["County"] + "-" + df_modis["State"]
df_modis = df_modis[df_modis["key"].isin(counties_states["key"])]
features = ["year", "key", "County", "State", "longitude", "latitude"] + [f"EVI_{date}" for date in dates]
df_modis = df_modis[features]
df_modis = df_modis.dropna()
df_modis = df_modis.drop_duplicates(["County", "State", "longitude", "latitude", "year"])


# read in and take subset of features data
df_gridmet = pd.read_csv("../../data/crops/processed_covariates/GRIDMET_data.csv")
df_gridmet = df_gridmet[df_gridmet["year"]==2015]
df_gridmet["key"] = df_gridmet["County"] + "-" + df_gridmet["State"]
df_gridmet = df_gridmet[df_gridmet["key"].isin(counties_states["key"])]
features = ["key", "year", "County", "State", "longitude", "latitude"] + col_gridmet
df_gridmet = df_gridmet[features]
df_gridmet = df_gridmet.drop_duplicates(["County", "State", "longitude", "latitude", "year"])

df_response = pd.read_csv("../../data/synthetic/synthetic_yields.csv")

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
keys = list(set(df_response.key))
keys.sort()


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
col_index_modis = [2]
col_index_pr = [3]
col_index_tmmx = [4]
max_pixels = [500, 100, 100, 100]


for fold, (train_index, test_index) in tqdm(enumerate(kf.split(keys))):
    df_Z = pd.read_csv(f"../../data/crops/processed_covariates/df_Z_MVBAgg_fold{fold}.csv")
    train_keys = [keys[key] for key in train_index]
    test_keys = [keys[key] for key in test_index]
    Z = df_Z[df_Z["key"].isin(train_keys)][all_features].values
    
    X_latlon_train = df_latlon[df_latlon["key"].isin(train_keys)].loc[:, latlon_cols].values
    X_latlon_test = df_latlon[df_latlon["key"].isin(test_keys)].loc[:, latlon_cols].values
    X_modis_train = df_modis[df_modis["key"].isin(train_keys)].loc[:, modis_cols].values
    X_modis_test = df_modis[df_modis["key"].isin(test_keys)].loc[:, modis_cols].values
    X_pr_train = df_gridmet[df_gridmet["key"].isin(train_keys)].loc[:, pr_cols].values
    X_pr_test = df_gridmet[df_gridmet["key"].isin(test_keys)].loc[:, pr_cols].values
    X_tmmx_train = df_gridmet[df_gridmet["key"].isin(train_keys)].loc[:, tmmx_cols].values
    X_tmmx_test = df_gridmet[df_gridmet["key"].isin(test_keys)].loc[:, tmmx_cols].values
    y_train = df_response[df_response["key"].isin(train_keys)].y.values[:, None]
    y_test = df_response[df_response["key"].isin(test_keys)].y.values[:, None]

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

    df_features_list = [df_latlon, df_modis, df_gridmet, df_gridmet]

    train_dict = create_bag_multiresolution_dictionary(
        [X_latlon_train, X_modis_train, X_pr_train, X_tmmx_train], 
        y_train, 
        [df_features[df_features["key"].isin(train_keys)] for df_features in df_features_list],
        df_response[df_response["key"].isin(train_keys)],
        train_keys,
        max_pixels
    )
    test_dict = create_bag_multiresolution_dictionary(
        [X_latlon_test, X_modis_test, X_pr_test, X_tmmx_test], 
        y_test, 
        [df_features[df_features["key"].isin(test_keys)] for df_features in df_features_list],
        df_response[df_response["key"].isin(test_keys)],
        test_keys,
        max_pixels
    )

    train_bags = BagData(bag_data=train_dict, bags_metadata=train_keys)
    test_bags = BagData(bag_data=test_dict, bags_metadata=test_keys)

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
    save_model(m, save_path="../../results/synthetic", model_name=f"MVBAgg_fold{fold}")

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
    RMSE.append(np.sqrt(np.mean((y_pred_rescaled - y_test)**2)))
    MAPE.append(np.mean(np.abs(( y_pred_rescaled - y_test) / y_test)))
    LL.append(loglik)
    training_time.append(t1-t0)

    # plot predictions
    plt.figure(figsize=(8,8))
    plt.scatter(y_pred_rescaled, y_test, color="red")
    plt.plot(np.linspace(-100,100,201), np.linspace(-100,100,201), color="black")

    plt.errorbar(
        y_pred_rescaled[:,0],
        y_test[:,0],
        xerr=errors,
        fmt="o",
        ls="none",
        capsize=5,
        markersize=4,
        color="blue",
        alpha=0.2
        )
    plt.xlim((y_test.min()-10, y_test.max()+10))
    plt.ylim((y_test.min()-10, y_test.max()+10))
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.savefig(f"../../results/synthetic/MVBAgg_fold{fold}.png")
    plt.savefig(f"../../results/synthetic/MVBAgg_fold{fold}.pdf")
    plt.close()

    print(f"MAPE", MAPE)
    print(f"RMSE", RMSE)
    print(f"LL", LL)
    print(f"Training Time", training_time)

json_file = json.dumps({"CV-RMSE": sum(RMSE) / 5, "CV-MAPE": sum(MAPE) / 5, 
                       "CV-sd-RMSE": sem(RMSE), "CV-sd-MAPE": sem(MAPE), "CV-LL": sum(LL) / 5, "CV-sd-LL": sem(LL),
                        "Training Time": sum(training_time)/5, "Training Time se": sem(training_time)}
                       )
f = open(f"../../results/synthetic/MVBAgg.json", "w")
f.write(json_file)
f.close()