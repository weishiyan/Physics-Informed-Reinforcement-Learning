import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

import Logger
import pinn
import lbfgs
import prep_data


def run_pendulum(tf_ep, nt_ep):
    # Data size on the solution u
    N_u = 500
    # Collocation points size, where we’ll check for f = 0
    N_f = 600
    # DeepNN 1-sized input [t], 8 hidden layer of 20-width, 1-sized output [u]
    layers = [1, 80, 80, 80, 80, 80, 80, 80, 80, 1]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    tf_epochs = int(tf_ep)
    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.007,
                                            epsilon=1e-1)
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
    nt_epochs = int(nt_ep)
    nt_config = lbfgs.Struct()
    nt_config.learningRate = 0.8
    nt_config.maxIter = nt_epochs
    nt_config.nCorrection = 50
    nt_config.tolFun = 1.0 * np.finfo(float).eps


    # Creating the model and training
    X_f, Exact_u, X_u_train, u_train, lb, ub = prep_data.prep_data(
        N_u, N_f, noise=0.01)
    plt.scatter(X_u_train, u_train, marker='.')
    plt.show()

    logger = Logger.Logger(frequency=10)
    pinns = pinn.PhysicsInformedNN(
        layers, tf_optimizer, logger, X_u_train, ub, lb, nu=(10/100))


    def error():
        u_pred, _ = pinns.predict(X_f)
        return np.linalg.norm(Exact_u - u_pred, 2) / np.linalg.norm(Exact_u, 2)


    logger.set_error_fn(error)
    pinns.fit(X_u_train, u_train, tf_epochs, nt_config)

    u_pred, f_pred = pinns.predict(X_f)

    plt.scatter(X_f, u_pred, marker='.',c='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta")
    plt.title("Predicted Data from Physics Informed Neural Network")
    plt.savefig("plots/Predicted_Data.png")
    # plt.show()

tf_eps = sys.argv[1]
nt_eps = sys.argv[2]

data = run_pendulum(tf_eps, nt_eps)