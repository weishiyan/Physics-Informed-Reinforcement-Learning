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
<<<<<<< ours

        ## Setup GPU cfg
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

=======
        
>>>>>>> theirs
        ## Get hyper-parameters from json cfg file
        data = []
        with open(cfg) as json_file:
            data = json.load(json_file)
<<<<<<< ours

        self.search_method =  (data['search_method']).lower() if (data['search_method']) else "epsilon"  # discount rate
        self.search_method = "epsilon"
        self.gamma =  float(data['gamma']) if float(data['gamma']) else 0.95  # discount rate
        self.epsilon = float(data['epsilon']) if float(data['epsilon']) else 1.0  # exploration rate
        self.epsilon_min = float(data['epsilon_min']) if float(data['epsilon_min']) else 0.05
        self.epsilon_decay = float(data['epsilon_decay']) if float(data['epsilon_decay']) else 0.995
        self.learning_rate =  float(data['learning_rate']) if float(data['learning_rate']) else  0.001
        self.batch_size = int(data['batch_size']) if int(data['batch_size']) else 32
        self.target_train_interval = int(data['target_train_interval']) if int(data['target_train_interval']) else 50
        self.tau = float(data['tau']) if float(data['tau']) else 1.0
        self.save_model = data['save_model'] if str(data['save_model']) else './model'

=======
            
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
>>>>>>> theirs
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
<<<<<<< ours
        h1 = Dense(128, activation='relu')(state_input)
        h2 = Dense(256, activation='relu')(h1)
        h3 = Dense(128, activation='relu')(h2)
        ## Output: action ##
        output = Dense(self.env.action_space.n,activation='relu')(h3)
        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.learning_rate)
        #model.compile(loss='mean_squared_logarithmic_error', optimizer=adam)
        model.compile(loss='mae', optimizer=adam)
        return model

=======
        h1 = Dense(32, activation='relu')(state_input)
        h2 = Dense(32)(h1)
        output = Dense(self.env.action_space.n)(h2)
        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.learning_rate)
        model.compile(optimizer=adam,loss='mse')
        return model       
    
>>>>>>> theirs
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
<<<<<<< ours
            ## Adding the UCB
            if self.search_method=="ucb":
                logger.info('START UCB')
                logger.info( 'Default values')
                logger.info( (act_values))
                logger.info( (action))
                act_values +=  self.sigma*np.sqrt(math.log(self.total_actions_taken)/self.individual_action_taken)
                action = np.argmax(act_values[0])
                logger.info( 'UCB values')
                logger.info( (act_values))
                logger.info( (action))
                ## Check if there are multiple candidates and select one randomly
                mask = [i for i in range(len(act_values[0])) if act_values[0][i] == act_values[0][action]]
                ncands=len(mask)
                logger.info( 'Number of cands: %s' % str(ncands))
                if ncands>1:
                    action = mask[random.randint(0,ncands-1)]
                logger.info( (action))
                logger.info('END UCB')

        ## Capture the action statistics for the UBC methods
        logger.info('total_actions_taken: %s' % self.total_actions_taken)
        logger.info('individual_action_taken[%s]: %s' % (action,self.individual_action_taken[action]))
=======
>>>>>>> theirs
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
<<<<<<< ours

=======
>>>>>>> theirs
            np_state = np.array(state).reshape(1,len(state))
            np_next_state = np.array(next_state).reshape(1,len(next_state))
            expectedQ =0
            if not done:
                expectedQ = self.gamma*np.amax(self.target_model.predict(np_next_state)[0])
            target = reward + expectedQ
<<<<<<< ours
            #print('reward %s' % reward)
            #print('expectedQ %s' % expectedQ)
            #print('target %s' % target)
                #target = max(target,-1)
                #print ("reward:{},expectedQ:{},target:{}".format(reward,expectedQ,target))

=======
            
>>>>>>> theirs
            target_f = self.target_model.predict(np_state)
            target_f[0][action] = target

            if batch_states==[]:
                batch_states=np_state
                batch_target=target_f
            else:
                batch_states=np.append(batch_states,np_state,axis=0)
                batch_target=np.append(batch_target,target_f,axis=0)
<<<<<<< ours

            #history = self.model.fit(np_state, target_f, epochs = 1, verbose = 0)
            #losses.append(history.history['loss'])
        history = self.model.fit(batch_states, batch_target, epochs = 1, verbose = 0)
        losses.append(history.history['loss'][0])
        self.train_writer.writerow([np.mean(losses)])
        self.train_file.flush()

        #if len(self.memory)%(self.batch_size)==0:
        if self.target_train_counter%self.target_train_interval == 0:
            logger.info('### TRAINING TARGET MODEL ###')
            self.target_train()

        return np.mean(losses)
=======
            
        history = self.model.fit(batch_states, batch_target, epochs = 1, verbose = 0)
        training_losses.append(history.history['loss'][0])
        
        self.target_train_counter += 1
            
        self.target_train()  
        return np.mean(training_losses)
>>>>>>> theirs

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
<<<<<<< ours
        abspath = os.path.abspath(self.save_model + name)
        path = os.path.dirname(abspath)
        if not os.path.exists(path):os.makedirs(path)
        # Save JSON config to disk
        model_json_name = self.save_model + name + '.json'
        json_config = self.model.to_json()
        with open(model_json_name, 'w') as json_file:
            json_file.write(json_config)
        # Save weights to disk
        self.model.save_weights(self.save_model + name+'.weights.h5')
        self.model.save(self.save_model + name+'.modelall.h5')
        logger.info('### SAVING MODEL '+abspath+'###')
=======
        self.model.save_weights(name)
>>>>>>> theirs
