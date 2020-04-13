import random,sys
import numpy as np
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,BatchNormalization
from keras.optimizers import Adam
from keras import losses as krls
from keras import backend as K
import csv
import json
import math

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

#The Deep Q-Network (DQN)
class DQN:
    def __init__(self, env,cfg='cfg/dqn_setup.json'):
        self.env = env

        ## Implement the UCB approach
        self.sigma = 2 # confidence level
        self.total_actions_taken = 1
        self.individual_action_taken = np.ones(self.env.action_space.n)
        
        ## Get hyper-parameters from json cfg file
        data = []
        with open(cfg) as json_file:
            data = json.load(json_file)
            
        self.search_method =  (data['search_method']).lower() if 'search_method' in data.keys() else "epsilon"  # discount rate
        self.gamma =  float(data['gamma']) if 'gamma' in data.keys() else 0.95  # discount rate
        self.epsilon = float(data['epsilon']) if 'epsilon' in data.keys() else 1.0  # exploration rate
        self.epsilon_min = float(data['epsilon_min']) if 'epsilon_min' in data.keys() else 0.05
        self.epsilon_decay = float(data['epsilon_decay']) if 'epsilon_decay' in data.keys() else 0.995
        self.learning_rate =  float(data['learning_rate']) if 'learning_rate' in data.keys() else  0.001
        self.batch_size = int(data['batch_size']) if 'batch_size' in data.keys() else 32
        self.tau = float(data['tau']) if 'tau' in data.keys() else 0.5
        self.memory_length = int(data['memory_length']) if 'memory_length' in data.keys() else 2000
        
        self.memory = deque(maxlen = self.memory_length) 
        
        ##
        self.model = self._build_model()
        self.target_model = self._build_model()

        ## Save infomation ##
        train_file_name = "dqn_mse_cartpole_%s_lr%s__tau%s_v1.log" % (self.search_method, str(self.learning_rate) ,str(self.tau) )
        self.train_file = open(train_file_name, 'w')
        self.train_writer = csv.writer(self.train_file, delimiter = " ")
        self.target_train_counter = 0
    
    def _build_model(self):
        ## Input: state ##       
        state_input = Input(self.env.observation_space.shape)
        h1 = Dense(32, activation='relu')(state_input)
        h2 = Dense(32)(h1)
        output = Dense(self.env.action_space.n)(h2)
        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.learning_rate)
        model.compile(optimizer=adam,loss='mse')
        return model       
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        action = -1
        self.random_action = 0
        ## TODO: Update greed-epsilon to something like UBC
        self.epsilon_adj()
        if self.search_method=="epsilon" and np.random.rand() <= self.epsilon:
            action = random.randrange(self.env.action_space.n)        
            self.random_action = 1
        else: 
            np_state = np.array(state).reshape(1,len(state))
            act_values = self.target_model.predict(np_state)
            action = np.argmax(act_values[0])
        self.total_actions_taken += 1
        self.individual_action_taken[action]+=1
        return action

    def play(self,state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def train(self):
        if len(self.memory)<(self.batch_size*5):
            return
        #print('### TRAINING ###')
        training_losses = []
        minibatch = random.sample(self.memory, self.batch_size)
        batch_states = []
        batch_target = []
        for state, action, reward, next_state, done in minibatch:
            np_state = np.array(state).reshape(1,len(state))
            np_next_state = np.array(next_state).reshape(1,len(next_state))
            expectedQ =0 
            if not done:
                expectedQ = self.gamma*np.amax(self.target_model.predict(np_next_state)[0])
            target = reward + expectedQ
            
            target_f = self.target_model.predict(np_state)
            target_f[0][action] = target
            
            if batch_states==[]:
                batch_states=np_state
                batch_target=target_f
            else:
                batch_states=np.append(batch_states,np_state,axis=0)
                batch_target=np.append(batch_target,target_f,axis=0)
            
        history = self.model.fit(batch_states, batch_target, epochs = 1, verbose = 0)
        training_losses.append(history.history['loss'][0])
        
        self.target_train_counter += 1
            
        self.target_train()  
        return np.mean(training_losses)

    def target_train(self):
        model_weights  = self.model.get_weights()
        target_weights =self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau*model_weights[i] + (1-self.tau)*target_weights[i]
        self.target_model.set_weights(target_weights)

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
