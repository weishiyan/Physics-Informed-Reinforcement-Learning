'''
This is the unittest for prep_data module
'''
import unittest

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import prep_data

class UnitTests(unittest.TestCase):
    '''
    Unittest suite
    '''
    
    def test_prep_data(self):
        '''
        Test exception of prep_data when handling 
        sample size larger than data size
        '''
        dataset = '../data/single_action_2_pendulum_data_L100.npz'
        
        try:
            N_f = 1500
            N_u = N_f + 100

            t, theta, t_train, theta_train, t_f, ub, lb = \
                prep_data.prep_data(dataset, N_u, N_f, noise=0.1)
        except (ValueError):
            print("Passed when random size is larger than data size.")
            pass
        else:
            raise Exception("Sample size larger than data size")

    
suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
_ = unittest.TextTestRunner().run(suite)
