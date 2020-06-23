import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

import Logger
import pinn
import nn
import prep_data


def run_pendulum(network, tf_ep, pendulum_length):
    dataset = 'data/single_action_2_pendulum_data_L%s.npz' % pendulum_length
    g = 10.0  # default gravity value in openAI
    length = float(pendulum_length)
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
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it

    # Creating the model and training
    X_f, Exact_u, X_u_train, u_train, lb, ub = prep_data.prep_data(
        dataset, N_u, N_f, noise=0.01)
    plt.scatter(X_u_train, u_train, marker='.')
    plt.show()
    logger = Logger.Logger(frequency=10)

    # Train with physics informed network
    if network == 'pinn':
        pinns = pinn.PhysicsInformedNN(
            layers, tf_optimizer, logger, X_u_train, ub, lb, g, length)

        def error():
            u_pred, _ = pinns.predict(X_f)
            return np.linalg.norm(
                Exact_u - u_pred, 2) / np.linalg.norm(Exact_u, 2)
        logger.set_error_fn(error)
        pinns.fit(X_u_train, u_train, tf_epochs)
        u_pred, f_pred = pinns.predict(X_f)
        plt.scatter(X_f, u_pred, marker='.', c='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Theta")
        plt.title("Predicted Data from Physics Informed NeuralenNetwork")
        plt.savefig("plots/PINN_Predicted_Data.png")
    # Train without physics
    else:
        nns = nn.NN(
            layers, tf_optimizer, logger, X_u_train, ub, lb, g, length)

        def error():
            u_pred, _ = nns.predict(X_f)
            return np.linalg.norm(
                Exact_u - u_pred, 2) / np.linalg.norm(Exact_u, 2)
        logger.set_error_fn(error)
        nns.fit(X_u_train, u_train, tf_epochs)
        u_pred, f_pred = nns.predict(X_f)
        plt.scatter(X_f, u_pred, marker='.', c='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Theta")
        plt.title("Predicted Data from Physics Uninformed Neural Network")
        plt.savefig("plots/NN_Predicted_Data.png")
        # plt.show()
        pass


nnetwork = sys.argv[1]
tf_eps = sys.argv[2]
pen_leng = sys.argv[3]

data = run_pendulum(nnetwork, tf_eps, pen_leng)
