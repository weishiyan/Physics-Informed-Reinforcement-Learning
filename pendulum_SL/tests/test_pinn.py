'''
This is the unittest for pinn module
'''
import unittest

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import tensorflow as tf

import prep_data
import Logger
import nn

dataset = '../data/single_action_2_pendulum_data_L100.npz'

class UnitTests(unittest.TestCase):
    '''
    
    '''
    
    def test_pinn_layer_shape(self):
        tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.007,
                                            epsilon=1e-1)
        logger = Logger.Logger(frequency=10)
        g = 10.0  # default gravity value in openAI
        length = float(100)
        N_f = 1500
        N_u = 1000

        X_f, Exact_u, X_u_train, u_train, lb, ub = \
                prep_data.prep_data(dataset, N_u, N_f, noise=0.1)
        try:
            layers = [2, 80, 80, 80, 80, 80, 80, 80, 80, 1]
            
            pinns= pinn.PhysicsInformedNN(layers, tf_optimizer, logger, X_u_train, ub, lb, g, length)
            pinns.fit(X_u_train, u_train, 100)
        except (ValueError):
            pass
        else:
            raise Exception("Input data must match input layer shape")
    
    def test_pinn_f_model(self):
        N_f = 1500
        N_u = 1000

        X_f, Exact_u, X_u_train, u_train, lb, ub = \
                prep_data.prep_data(dataset, N_u, N_f, noise=0.1)
        
        self.t_f = tf.convert_to_tensor(X_u_train, dtype=self.dtype)
        pinns= pinn.PhysicsInformedNN(layers, tf_optimizer, logger, X_u_train, ub, lb, g, length)
        
        solution = pinns.f_model()

    
suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
_ = unittest.TextTestRunner().run(suite)
