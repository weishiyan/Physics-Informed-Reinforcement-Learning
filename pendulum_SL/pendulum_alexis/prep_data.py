import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from pyDOE import lhs
import os
import sys


# "." for Colab/VSCode, and ".." for GitHub
repoPath = os.path.join(".", "PINNs")
# repoPath = os.path.join("..", "PINNs")
utilsPath = os.path.join(repoPath, "Utilities")
dataPath = os.path.join(repoPath, "main", "Data")
appDataPath = os.path.join(repoPath, "appendix", "Data")

sys.path.insert(0, utilsPath)

# from plotting import newfig, savefig


def prep_data(theta, time, N_u=None, N_f=None, N_n=None, q=None, ub=None,
              lb=None, noise=0.1, idx_t_0=None, idx_t_1=None, N_0=None,
              N_1=None):

    # Jan's data set
    #OMEGA = 2
    #THETA_0 = 0.33
#
    #lb = 0
    #ub = 4*np.pi 
    #t_f = lb + (ub - lb) * lhs(1, N_f)
    #theta = THETA_0*np.cos(OMEGA*t_f)
    #t_train = t_f
    ##theta_train = theta[:N_u]
    #theta_train = THETA_0*np.cos(OMEGA*t_train) + noise*np.random.randn(N_f,1)    
    
    
    # Extracting data
    t = time[:, None]  # variable
    theta = theta[:, None]  # solution

    # Adding noise to data - awm
    mu, sigma = 0, noise
    # give same shape as data
    random_noise = np.random.normal(mu, sigma, [len(theta)])
    theta = theta #+ random_noise

    #idx = np.random.choice(t.shape[0], N_u, replace=False)
    # training data
    #t_train = t[idx, 0]  # variable train
    #theta_train = theta[idx, 0]  # solution train
    t_train = t[:N_u, 0]  # variable train
    theta_train = theta[:N_u, 0]  # solution train

    # t = t[points:,0]
    # theta = theta[points:,0]

    # Domain bounds (lowerbounds upperbounds) [t] which is [0, x]
    lb = t.min(axis=0)
    ub = t.max(axis=0)

    # Generating the x and t collocation points for f, with each N_f size
    # We pointwise add and multiply to spread the LHS over the 2D domain
    t_f = lb + (ub - lb) * lhs(1, N_f)

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

    return t, theta, t_train, theta_train, t_f, ub, lb


class Logger(object):
    def __init__(self, frequency=10):
        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

        self.start_time = time.time()
        self.frequency = frequency

    def __get_elapsed(self):
        return datetime.fromtimestamp(
            time.time() - self.start_time).strftime("%M:%S")

    def __get_error_u(self):
        return self.error_fn()

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, model):
        print("\nTraining started")
        print("================")
        self.model = model
        print(self.model.summary())

    def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
        if epoch % self.frequency == 0:
            print(f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d} \
            elapsed = {self.__get_elapsed()} \
            loss = {loss:.4e} \
            error = {self.__get_error_u():.4e}  " + custom)

    def log_train_opt(self, name):
        print(f"—— Starting {name} optimization ——")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): \
        duration = {self.__get_elapsed()}  \
        error = {self.__get_error_u():.4e}  " + custom)
