'''
This is the unittest for modified_dqn module
'''
import unittest

import gym
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
    
from agents.modified_dqn import DQN, _time_reward

env = gym.make('CartPole-v0')
agent = DQN(env)

class UnitTests(unittest.TestCase):
    '''
    Unittest suite
    '''

    def test_x(self):
        x_threshold = env.x_threshold
        low_reward_state = np.array([x_threshold*0.9, 0.1, 0, 0.1])
        high_reward_state = np.array([0, 0.1, 0, 0.1])
        low_reward = _time_reward(low_reward_state,env)
        high_reward = _time_reward(high_reward_state,env)
        assert low_reward < high_reward, "The reward structure does not follow desired physics"
        
    def test_x_dot(self):
        low_reward_state = np.array([1, 0.2, 0, 0.1])
        high_reward_state = np.array([1, 0.1, 0, 0.1])
        low_reward = _time_reward(low_reward_state,env)
        high_reward = _time_reward(high_reward_state,env)
        assert low_reward < high_reward, "The reward structure does not follow desired physics"
        
    def test_theta(self):
        theta_threshold = env.theta_threshold_radians
        low_reward_state = np.array([0, 0.1, theta_threshold*0.9, 0.1])
        high_reward_state = np.array([0, 0.1, 0, 0.1])
        low_reward = _time_reward(low_reward_state,env)
        high_reward = _time_reward(high_reward_state,env)
        assert low_reward < high_reward, "The reward structure does not follow desired physics"
        
    def test_theta_dot(self):
        low_reward_state = np.array([1, 0.1, 0, 0.2])
        high_reward_state = np.array([1, 0.1, 0, 0.1])
        low_reward = _time_reward(low_reward_state,env)
        high_reward = _time_reward(high_reward_state,env)
        assert low_reward < high_reward, "The reward structure does not follow desired physics"
        
        
suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
_ = unittest.TextTestRunner().run(suite)

agent.train_file.close()
os.remove(agent.train_file.name)