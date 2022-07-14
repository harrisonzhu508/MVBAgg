# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
import sys 
sys.path.append("../..")
import pandas as pd 
import geopandas as gpd
from tqdm import tqdm
from src.bagData import BagData, create_tensorflow_iterator, create_bag_sameresolution_dictionary
import os
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
from src.svgp import VBagg
from src.util import optimize_adam, optimize_natgrad_adam, save_model
from sklearn.cluster import KMeans
seed = 0

# %%
# read in user-specified counties and states to study
counties_states = pd.read_csv("../../data/crops/counties-states.csv")
dates = ['04-07', '04-23', '05-09', '05-25',
       '06-10', '06-26', '07-12', '07-28',
       '08-13', '08-29', '09-14', '09-30',
       '10-16']

# read in and take subset of yield data
df_yield = pd.read_csv("../../data/crops/soybean_yield_2015_2017.csv")
df_yield["key"] = df_yield["County"] + "-" + df_yield["State"]
df_yield = df_yield[df_yield["key"].isin(counties_states["key"])]
df_yield.rename(columns={"Year": "year"}, inplace=True)
df_yield.rename(columns={"Year": "year"}, inplace=True)
df_yield = df_yield.drop_duplicates(["County", "State", "year"])
df_yield["Value"] = np.log(df_yield["Value"])

# read in and take subset of features data
df_features = pd.read_csv("../../data/crops/processed_covariates/MOD13Q1-GRIDMET-downsampled_data.csv")
df_features["key"] = df_features["County"] + "-" + df_features["State"]
df_features = df_features[df_features["key"].isin(counties_states["key"])]

# dates of the features
features = ["key", "County", "State", "year", "longitude", "latitude"] + [f"EVI_{date}" for date in dates] + [f"pr_{date}" for date in dates] + [f"tmmx_{date}" for date in dates]
df_features = df_features[features]
df_features = df_features.drop_duplicates(["County", "State", "year", "longitude", "latitude"])

assert df_yield.shape[0] == 768

# %%
# write down index of the features
col_index_space = [0, 1]
col_index_modis = [2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12,13,14]
col_index_pr = [15,16,17,18,19,20,21,22,23,24,25,26,27]
col_index_tmmx = [28 + i for i in range(13)]
col_index_all = col_index_space + col_index_modis + col_index_pr + col_index_tmmx

# %% [markdown]
# ## Experimental Setup

kf = KFold(n_splits=5, random_state=seed, shuffle=True)
output_types=(
    tf.int64, 
    tf.float64, 
    tf.float64, 
    tf.float64
)
keys = list(set(df_yield[df_yield["year"]==2015].key).intersection(set(df_yield[df_yield["year"]==2017].key)))
keys.sort()

# %% [markdown]
# Now we create a dictionary in the required format of src.bagData.bagData

# %%
RMSE = []
MAPE  = []
LL  = []
training_time = []
iterations = 20000
num_minibatch = 50
lr_adam = 0.001#
max_pixels = 100
latlon_cols = ["longitude", "latitude"]
modis_cols = [f"EVI_{date}" for date in dates]
pr_cols = [f"pr_{date}" for date in dates] 
tmmx_cols = [f"tmmx_{date}" for date in dates] 
all_features = latlon_cols + modis_cols + pr_cols + tmmx_cols

for fold, (train_index, test_index) in tqdm(enumerate(kf.split(keys))):
    county_keys = [keys[key] for key in train_index]
    
    X_train = df_features.loc[(df_features["key"].isin(county_keys)) & (df_features["year"]==2015), all_features].values
    X_test = df_features.loc[(df_features["key"].isin(county_keys)) & (df_features["year"]==2017), all_features].values
    y_train = df_yield[(df_yield["key"].isin(county_keys)) & (df_yield["year"]==2015)].Value.values[:, None]
    y_test = df_yield[(df_yield["key"].isin(county_keys)) & (df_yield["year"]==2017)].Value.values[:, None]

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    X_train, y_train = scaler_x.transform(X_train), scaler_y.transform(y_train)
    X_test = scaler_x.transform(X_test)

    train_dict = create_bag_sameresolution_dictionary(
        X_train, 
        y_train, 
        df_features[(df_features["key"].isin(county_keys)) & (df_features["year"]==2015)],
        df_yield[(df_yield["key"].isin(county_keys))& (df_yield["year"]==2015)],
        county_keys,
        max_pixels
    )
    test_dict = create_bag_sameresolution_dictionary(
        X_test, 
        y_test, 
        df_features[(df_features["key"].isin(county_keys)) & (df_features["year"]==2017)],
        df_yield[(df_yield["key"].isin(county_keys)) & (df_yield["year"]==2017)],
        county_keys,
        max_pixels
    )

    train_bags = BagData(bag_data=train_dict, bags_metadata="2015 yields")
    test_bags = BagData(bag_data=test_dict, bags_metadata="2017 yields")

    ds_train = create_tensorflow_iterator(train_bags.gen_bags, output_types=output_types)

    # create inducing points
    kmeans = KMeans(1)
    Z = np.zeros((train_bags.num_bags, len(col_index_all)))
    for i, bag in enumerate(train_bags.bags):
        data = train_bags[bag]
        Z_tmp = kmeans.fit(data[2]).cluster_centers_
        Z[i] = Z_tmp

    # fit and train GP regression model
    k_space = gpflow.kernels.Matern32(active_dims=col_index_space)
    k_modis = gpflow.kernels.RBF(active_dims=col_index_modis)
    k_pr = gpflow.kernels.RBF(active_dims=col_index_pr)
    k_tmmx = gpflow.kernels.RBF(active_dims=col_index_tmmx)
    k = k_space + k_modis + k_pr + k_tmmx
    m = VBagg(kernel=k, likelihood=gpflow.likelihoods.Gaussian(), inducing_variable=Z,num_data=train_bags.num_bags)
    gpflow.set_trainable(m.inducing_variable, False)
    
    print("Begin Training")
    t0 = time.time()
    logf = optimize_natgrad_adam(m, ds_train, num_data=train_bags.num_bags, iterations=iterations, minibatch_size=num_minibatch, learning_rate=lr_adam)
    t1 = time.time()
    save_model(m, save_path="../../results/usa_crops", model_name=f"VBagg_fold{fold}")



    # make predictions
    y_pred = np.zeros(test_bags.num_bags)
    y_std = np.zeros(test_bags.num_bags)
    for i, bag in enumerate(test_bags.bags):
        f_mean, f_var = m.predict_aggregated(test_bags[bag][1], test_bags[bag][2])
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
    plt.savefig(f"../../results/usa_crops/VBAgg_fold{fold}.png")
    plt.savefig(f"../../results/usa_crops/VBAgg_fold{fold}.pdf")
    plt.close()

    print(f"MAPE", MAPE)
    print(f"RMSE", RMSE)
    print(f"LL", LL)
    print(f"Training Time", training_time)

json_file = json.dumps({"CV-RMSE": sum(RMSE) / 5, "CV-MAPE": sum(MAPE) / 5, 
                       "CV-sd-RMSE": sem(RMSE), "CV-sd-MAPE": sem(MAPE), "CV-LL": sum(LL) / 5, "CV-sd-LL": sem(LL),
                        "Training Time": sum(training_time)/5, "Training Time se": sem(training_time)}
                       )
f = open(f"../../results/usa_crops/VBAgg.json", "w")
f.write(json_file)
f.close()