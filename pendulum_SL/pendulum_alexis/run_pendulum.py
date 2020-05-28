

import scipy.io
import numpy as np
import tensorflow as tf
from datetime import datetime
from pyDOE import lhs
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

import prep_data
#import lbfgs
import pinn as PINNs
import plot

np.random.seed(1234)
tf.random.set_seed(1234)

epoch = 500
# Data size on the solution u
N_u = 500
# Collocation points size, where we’ll check for f = 0
N_f = 1500
# DeepNN topology (1-sized input [t], 8 hidden layer of 20-width, 1-sized output [theta]
layers = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
tf_epochs = epoch

# OPTIMIZERS
tf_optimizer = tf.keras.optimizers.Adam(
  learning_rate=1e-3,
  beta_1=0.999,
  epsilon= 1e-1)
#tf_optimizer = tf.keras.optimizers.Adadelta(
#    learning_rate=1e-3,
#    epsilon=1e-1)

# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
nt_epochs = epoch


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
theta = states_data[:,1]
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

print("Entering 'data_prep' script")
set_size = 1500 # reduce set size to accommodate computational limitations
t, theta, t_train, theta_train, t_f, ub, lb = prep_data.prep_data(theta[:set_size], time_data[:set_size], N_u, N_f, noise=0.0)
print("Exited 'data_prep' script")
# Simple test confirming theta_train and t_train go together
#print(t_train.shape)
#print(t_train)
#print('-------------')
#print(theta_train.shape)
#print(theta_train)
#print('-------------')
#index_data = (float(t_train[0])/0.05)+1
#print(np.arccos(states_data[int(index_data)][0]),time_data[int(index_data)])


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

#print((t_train[:,None].shape))
pinn.fit(t_train[:], theta_train[:], tf_epochs=tf_epochs)

# Getting the model predictions, from the same t that the predictions were previously obtained from
theta_pred, f_pred = pinn.predict(t)

print(f_pred.shape)
f_predict_list = []
i = 0
for i in range(f_pred.shape[0]):
  f_predict_list.append(f_pred[i, i, 0])

f_predict_array = np.array(f_predict_list)

f = open("logs/predicted_theta.out", "w+")
f.write("time \t theta \t theta_predicted \t f_predicted\n")
i = 0
for i in range(theta_pred.shape[0]):
  f.write("%0.3f \t %0.3f \t %0.3f \t \t \t %0.3f \n" %(t[i,0], theta[i,0], theta_pred[i,0], f_predict_array[i]))



# Plotting 
plot.exact_vs_pred(t, t_train, theta, theta_train, theta_pred)
filename = "pendulum_loss"
plot.loss(filename)
