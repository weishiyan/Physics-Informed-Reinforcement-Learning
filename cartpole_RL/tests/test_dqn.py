'''
This is the unittest for dqn module
'''
import unittest

import gym
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from collections import deque
import numpy as np
    
from agents.dqn import DQN

env = gym.make('CartPole-v0')
agent = DQN(env)

class UnitTests(unittest.TestCase):
    '''
    Unittest suite
    '''

    def test_build_model(self):
        '''
        Unittest for _build_model function
        Check if the model has input and output layers that
        match the observation space and action space 
        specifically for cartpole problem
        '''
        random_model = agent._build_model()
        assert agent._build_model().input_shape[1] == 4, "The model is not compatible for cartpole with observation space equal to 4"
        assert agent._build_model().output_shape[1] == 2, "The model is not compatible for cartpole with action space equal to 2"
    
    def test_action_output(self):
        '''
        Unittest for action function
        Check if action function can return 1 or 0 based on
        _build_model prediction
        '''
        random_state = env.reset()
        random_reward_values = agent._build_model().predict(np.array(random_state).reshape(1, len(random_state)))
        random_action = np.argmax(random_reward_values[0])
        assert len(random_reward_values[0]) == 2, "The model is not able to predict rewards for two actions"
        assert random_action == 1 or random_action ==0, "The predicted action has failed"
    
    def test_remember(self):
        '''
        Unittest for remember function
        Check if the remember function can store training data 
        into memory
        '''
        agent.memory = deque(maxlen=2000)
        agent.memory.clear()
        random_state = env.reset()
        random_action = agent.action(random_state)
        next_state, reward, done, _ = env.step(random_action)
        agent.remember(random_state, random_action, reward, next_state, done)
        to_be_remembered = random_state, random_action, reward, next_state, done
        assert agent.memory[-1] == to_be_remembered, "The remember function is not appending lastest training data into last row of memory"
        
    def test_train_batch_size(self):
        '''
        Unittest for train function
        Check if the train function can randomly select training data 
        when the memory don't have sufficient data.
        threshold is set at 5 times of batch size.
        '''
        agent.memory.clear()
        batch_size = 100
        for i in range(batch_size):
            random_state = env.reset()
            random_action = agent.action(random_state)
            next_state, reward, done, _ = env.step(random_action)
            agent.remember(random_state, random_action, reward, next_state, done)
            test_return = agent.train()
        assert test_return == None, "The agent is trainning on insufficient training data that is smaller than 5 times of batch size"
        
    def test_train_counter(self):
        '''
        Unittest for train function
        Check if the train function can update reward through iterations
        '''
        agent.memory.clear()
        agent.target_train_counter = 0
        batch_size = 200
        for i in range(batch_size):
            random_state = env.reset()
            random_action = agent.action(random_state)
            next_state, reward, done, _ = env.step(random_action)
            agent.remember(random_state, random_action, reward, next_state, done)
            test_return = agent.train()
        assert agent.target_train_counter + 160 - 1 == batch_size, "Training failed"

suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
_ = unittest.TextTestRunner().run(suite)

agent.train_file.close()
os.remove(agent.train_file.name)