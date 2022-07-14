import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from typing import Callable, Tuple
import tensorflow as tf
from tqdm import tqdm
class BagData:
    """All the datasets are the same resolution

    Input:

    bag_data: bag_data = {
        "N": N, # int
        "weights": weights, # (Ni, 1) array
        "x": x, # (Ni, d) array
        "y": y, # (1, 1) array
    } 
    """

    def __init__(self, bag_data: dict, bags_metadata = None) -> None:

        self.bag_data = bag_data
        self.num_bags = len(bag_data.keys())
        self.bags = list(bag_data.keys())

        y = np.zeros((self.num_bags, 1))
        for i, bag in enumerate(self.bags):
            y[i] = bag_data[bag]["y"]

        self.y = y
        self.bags_metadata = bags_metadata

    def __str__(self):
        msg = "BagData with {} bags".format(self.num_bags)
        return msg

    def __len__(self):
        return self.num_bags

    def __getitem__(self, bag):
        """
        bag: bag index from indexing system
        """
        return tuple(self.bag_data[bag].values())

    def gen_bags(self):
        for bag in self.bags:
            yield self.__getitem__(bag)

def create_tensorflow_iterator(gen_bags_func: Callable, output_types: Tuple):
    ds = tf.data.Dataset.from_generator(
        gen_bags_func, 
        output_types=output_types
    )
    return ds

def create_bag_sameresolution_dictionary(X_all, y_all, df_features, df_yield, keys, max_pixels, key_column="key"):
    data_dict = {}
    for key in tqdm(keys):
        x = X_all[df_features[key_column] == key, :]
        N = x.shape[0]
        x_padded = np.zeros((max_pixels, x.shape[1]))
        x_padded[:N, :] = x
        y = y_all[df_yield[key_column]==key]

        # use uniform weights
        weights = np.ones((max_pixels, 1)) / N
        weights[N:] = 0
        data_dict[key] = {
            "N": N,
            "weights": weights,
            "x": x_padded,
            "y": y
        }

    return data_dict

def create_bag_multiresolution_dictionary(X_all_list, y_all, df_features_list, df_yield, keys, max_pixels, key_column="key"):
    data_dict = {}
    for key in tqdm(keys):
        N_list = []
        x_list = []
        weights_list = []
        for i, (X_all, df_features) in enumerate(zip(X_all_list, df_features_list)):
            
            x = X_all[df_features[key_column]==key, :]
            N = x.shape[0]
            # use uniform weights
            weights = np.ones((max_pixels[i], 1)) / N
            weights[N:] = 0
            x_padded = np.zeros((max_pixels[i], x.shape[1]))
            x_padded[:N, :] = x

            N_list.append(N)
            weights_list.append(weights)
            x_list.append(x_padded)

        y = y_all[df_yield["key"]==key]
        data_dict[key] = {f"N{i}":val for i, val in enumerate(N_list)}
        data_dict[key].update({f"weights{i}":val for i, val in enumerate(weights_list)})
        data_dict[key].update({f"x{i}":val for i, val in enumerate(x_list)})
        data_dict[key].update({"y": y})
        
    return data_dict

def create_bag_sameresolution_binomial_dictionary(X_all, y_all, df_features, df_yield, keys, max_pixels, key_column="key", count_column="n_obs"):
    data_dict = {}
    for key in tqdm(keys):
        N_list = []
        x_list = []
        weights_list = []
        x = X_all[df_features[key_column] == key]
        N = x.shape[0]
        # use uniform weights
        weights = np.ones((max_pixels, 1)) / N
        weights[N:] = 0
        x_padded = np.zeros((max_pixels, x.shape[1]))
        x_padded[:N, :] = x

        N_list.append(N)
        weights_list.append(weights)
        x_list.append(x_padded)

        y = y_all[df_yield[key_column]==key]
        counts = df_yield[df_yield[key_column]==key][count_column].values[:, None]

        data_dict[key] = {f"N{i}":val for i, val in enumerate(N_list)}
        data_dict[key].update({f"weights{i}":val for i, val in enumerate(weights_list)})
        data_dict[key].update({f"x{i}":val for i, val in enumerate(x_list)})
        data_dict[key].update({"counts": counts})
        data_dict[key].update({"y": y})
        
    return data_dict

def create_bag_multiresolution_binomial_dictionary(X_all_list, y_all, df_features_list, df_yield, keys, max_pixels):
    data_dict = {}
    for key in tqdm(keys):
        N_list = []
        x_list = []
        weights_list = []
        for X_all, df_features in zip(X_all_list, df_features_list):
            x = X_all[df_features["name_2"] == key]
            N = x.shape[0]
            # use uniform weights
            weights = np.ones((max_pixels, 1)) / N
            weights[N:] = 0
            x_padded = np.zeros((max_pixels, x.shape[1]))
            x_padded[:N, :] = x

            N_list.append(N)
            weights_list.append(weights)
            x_list.append(x_padded)

        y = y_all[df_yield["name_2"]==key]
        counts = df_yield[df_yield["name_2"]==key]["n_obs"].values[:, None]
        data_dict[key] = {f"N{i}":val for i, val in enumerate(N_list)}
        data_dict[key].update({f"weights{i}":val for i, val in enumerate(weights_list)})
        data_dict[key].update({f"x{i}":val for i, val in enumerate(x_list)})
        data_dict[key].update({"counts": counts})
        data_dict[key].update({"y": y})
        
    return data_dict