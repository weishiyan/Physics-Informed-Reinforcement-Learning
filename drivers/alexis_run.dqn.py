# -*- coding: utf-8 -*-
import random, math
import gym
from tqdm import tqdm
import time
import numpy as np

##
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)

from agents.dqn import DQN

import csv

if __name__ == "__main__":
    import sys
    # for i in range(10):
    ###########
    ## Train ##
    ###########
    EPISODES = 500
    NSTEPS   = 200
    #######################
    ## Setup environment ##
    #######################
    estart = time.time()
    env = gym.make('CartPole-v0')
    env._max_episode_steps=NSTEPS
    env.seed(1)
    end = time.time()
    #logger.info('Time init environment: %s' % str( (end - estart)/60.0))
    #logger.info('Using environment: %s' % env)
    #logger.info('Observation_space: %s' % env.observation_space.shape)
    #logger.info('Action_size: %s' % env.action_space)
   
    #################
    ## Setup agent ##
    #################
    agent = DQN(env)
    
    ## Save infomation ##
    filename = "alexis3_dqn_cartpole_mse_episode%s_memory%s_noMod_" % (str(EPISODES),str(NSTEPS)) #,count)
    train_file = open(filename, 'w')
    train_writer = csv.writer(train_file, delimiter = " ")
    
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        #logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward=0
        step = 0
        done = False
        while done!=True:
            action = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)
            # Scale reward based on how close theta/x are to their respective thresholds
            theta_r = (env.theta_threshold_radians - np.abs(current_state[2])) / env.theta_threshold_radians 
            x_r = (env.x_threshold - np.abs(current_state[0])) / env.x_threshold 
            reward = (theta_r + x_r)/2
            agent.remember(current_state, action, reward, next_state, done)
            agent.train()
           # logger.info('Current state: %s' % str(current_state))
           # logger.info('Action: %s' % str(action))
           # logger.info('Next state: %s' % str(next_state))
           # logger.info('Reward: %s' % str(reward))
           # logger.info('Done: %s' % str(done))
            
            ##
            current_state = next_state
            ##
            total_reward+=reward
            step += 1
           # logger.info("Current episode reward: %s " % str(total_reward))
   
            ## Save memory
            train_writer.writerow([step,current_state,action,reward,next_state,total_reward,done])
            train_file.flush()
       
    agent.save("%s.h5" %filename)
    train_file.close()
