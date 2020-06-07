import plot
import pinn as PINNs
import prep_data
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import scipy.io
import numpy as np
from datetime import datetime
from pyDOE import lhs
import logging
import os


logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import lbfgs

np.random.seed(1234)
tf.random.set_seed(1234)

# Turning off INFO and WARNING messages for tf


# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
tf_epochs = 1000
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
nt_epochs = 0

nt_config = lbfgs.Struct()
nt_config.learningRate = 0.8
nt_config.maxIter = nt_epochs
nt_config.nCorrection = 50
nt_config.tolFun = 1.0 * np.finfo(float).eps

# OPTIMIZERS
tf_optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-3,
    beta_1=0.999,
    epsilon=1e-1)
# tf_optimizer = tf.keras.optimizers.Adadelta(
#    learning_rate=1e-3,
#    epsilon=1e-1)

# Collect data
npz = np.load('data/single_random_pendulum_data.npz', allow_pickle=True)
# 50000 states of the pendulum
states_data = npz['states']
rewards_data = npz['rewards']
# 50000 corresponding times for each state of the pendulum
time_data = npz['time']

# find theta from states array (cos(theta))
#theta = [np.arccos(x) for x in states_data[:,0]]
#theta = np.array(theta)

# using sin(theta) data
theta = states_data[:, 1]
# Getting the data
# x and t => replaced by one t
# usol => replace by theta
# X and T => unnecessary
# Exact_u => unnecessary
# X_star => t
# u_star => theta
# X_u_train => t_train
# u_train => theta_train
# X_f => t_f
# ub same
# lb same

# Data size on the solution u
N_u = 500
# Collocation points size, where we’ll check for f = 0
N_f = 1500


print("Entering 'data_prep' script")
set_size = 500  # reduce set size to accommodate computational limitations
mu = 0
sigma = 0.2
t, theta, t_train, theta_train, t_f, ub, lb = prep_data.prep_data(
    theta[:set_size], time_data[:set_size], N_u, N_f, mu=mu, sigma=sigma)
print("Exited 'data_prep' script")
# Simple test confirming theta_train and t_train go together
# print(t_train.shape)
# print(t_train)
# print('-------------')
# print(theta_train.shape)
# print(theta_train)
# print('-------------')
#index_data = (float(t_train[0])/0.05)+1
# print(np.arccos(states_data[int(index_data)][0]),time_data[int(index_data)])

# DeepNN topology (1-sized input [t], 8 hidden layer of 20-width, 1-sized output [theta]
layers = [1, 20, 20, 1]

# Creating the model and training
# Variables needed: t_f, ub, lb, t, theta, t_train, theta_train
logger = prep_data.Logger(frequency=10)
print("Entering 'pinn' script")
pinn = PINNs.PhysicsInformedNN(layers, tf_optimizer, logger, t_f, ub, lb)
print("Exited 'pinn' script")


def error():
    theta_pred, _ = pinn.predict(t)
    return np.linalg.norm(theta - theta_pred, 2) / np.linalg.norm(theta, 2)


logger.set_error_fn(error)

# print((t_train[:,None].shape))
pinn.fit(t_train, theta_train, nt_config, tf_epochs=tf_epochs)

# Getting the model predictions, from the same t that the predictions were previously obtained from
theta_pred, f_pred = pinn.predict(t)

# Obtaining the predicted theta values
print("f_pred shape: ")
print(f_pred.shape)
f_predict_list = []
i = 0
for i in range(f_pred.shape[0]):
    f_predict_list.append(f_pred[i, 0, 0])

f_predict_array = np.array(f_predict_list)

# Logging the predicted theta values
f = open("logs/predicted_theta.out", "w+")
f.write("time \t theta \t theta_predicted \t f_predicted\n")
i = 0
for i in range(theta_pred.shape[0]):
    f.write("%0.3f \t %0.3f \t %0.3f \t \t \t %0.3f \n" %
            (t[i, 0], theta[i, 0], theta_pred[i, 0], f_predict_array[i]))


# Plotting
filename = f"exact_vs_pred_{tf_epochs}epochs_{set_size}size_{len(layers)-2}hidden_{mu}mu_{sigma}sigma.jpg"
plot.exact_vs_pred(t, t_train, theta, theta_train, theta_pred, filename)
filename = f"pendulum_loss_{tf_epochs}epochs_{set_size}size_{len(layers)-2}hidden_{mu}mu_{sigma}sigma.jpg"
plot.loss(filename)
