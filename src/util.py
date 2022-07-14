from tqdm import trange
import tensorflow as tf
import gpflow
from gpflow.optimizers import NaturalGradient
from os import path
import pickle
from shapely.geometry import Point
import random
import numpy as np

def save_model(model: gpflow.models.GPModel, save_path="./", model_name=None):
    if model_name is None:
        model_name = "model"

    with open(f"{path.join(save_path, model_name)}.pickle", "wb") as handle:
        pickle.dump(model.parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model: gpflow.models.GPModel, model_path="./"):
    with open(model_path, "rb") as handle:
        parameters = pickle.load(handle)
    for i in range(len(model.parameters)):
        model.parameters[i].assign(parameters[i])


def optimize_adam(
    model, ds_train, iterations, num_data, minibatch_size=50, learning_rate=0.001
):
    """
    Utility function running the Adam optimizer
    Modified from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(ds_train.repeat().shuffle(num_data).batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    iterator = trange(iterations)
    for step in iterator:
        optimization_step()
        if step % 1000 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            iterator.set_description(f"EPOCH: {step}, ELBO: {elbo}")

    return logf


def optimize_natgrad_adam(
    model, ds_train, iterations, num_data, minibatch_size=50, learning_rate=0.001
):
    """
    Utility function running the Adam optimizer for hyperparameters and NaturalGradient for variational
    parameters
    Modified from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(ds_train.repeat().shuffle(num_data).batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    natgrad_opt = NaturalGradient(gamma=0.1)

    gpflow.set_trainable(model.q_mu, False)
    gpflow.set_trainable(model.q_sqrt, False)
    variational_params = [(model.q_mu, model.q_sqrt)]

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
        natgrad_opt.minimize(training_loss, variational_params)

    iterator = trange(iterations)
    for step in iterator:
        optimization_step()
        if step % 1000 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            iterator.set_description(f"EPOCH: {step}, ELBO: {elbo}")
    return logf


def random_points_in_polygon(number, polygon):
    """https://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon"""
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i = 0
    while i < number:
        long = random.uniform(min_x, max_x)
        lat = random.uniform(min_y, max_y)
        point = Point(long, lat)
        if polygon.contains(point):
            points.append([long, lat])
            i += 1
    return np.array(points)  # returns list of points