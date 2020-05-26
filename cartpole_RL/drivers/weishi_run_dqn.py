# -*- coding: utf-8 -*-
import random, math
import gym
from tqdm import tqdm
import time
import pandas as pd
import numpy as np

from scipy.stats import tukeylambda

##
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)

from agents.dqn import DQN

import csv

if __name__ == "__main__":
    import sys

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
    logger.info('Time init environment: %s' % str( (end - estart)/60.0))
    logger.info('Using environment: %s' % env)
    logger.info('Observation_space: %s' % env.observation_space.shape)
    logger.info('Action_size: %s' % env.action_space)

    #################
    ## Setup agent ##
    #################
    agent = DQN(env)
    
    ## Save infomation ##
    filename = "dqn_cartpole_mse_episode%s_memory%s_1" % (str(EPISODES),str(NSTEPS))
    train_file = open(filename, 'w')
    train_writer = csv.writer(train_file, delimiter = " ")
    
    def new_reward(state,lam=-0.5):
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
    
        # triangular function for x
        x_upper = env.x_threshold - x
        x_lower = env.x_threshold + x
    
        x_r = max(0.0, min(x_upper, x_lower))
    
        # triangular function for theta
        theta_upper = env.theta_threshold_radians - theta
        theta_lower = env.theta_threshold_radians + theta
    
        theta_r = max(0.0, min(theta_upper, theta_lower))
    
        # tukey-lambda function for x_dot
        lam = -0.5
        x_dot_threshold = 1.86
        x_dot_tukey = x_dot*tukeylambda.ppf(0.99,-0.5)/x_dot_threshold
        x_dot_r = tukeylambda.pdf(x_dot_tukey,lam)
    
        # tukey_lambda function for theta_dot
        theta_dot_threshold = 0.72
        theta_dot_tukey = theta_dot*tukeylambda.ppf(0.99,-0.5)/theta_dot_threshold
        theta_dot_r = tukeylambda.pdf(theta_dot_tukey,lam)
    
        new_reward = x_r * theta_r * x_dot_r * theta_dot_r

        return new_reward
    
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward=0
        done = False
        while done!=True:
            action = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)
            
            reward = new_reward(current_state)
            
            agent.remember(current_state, action, reward, next_state, done)
            
            agent.train()
            logger.info('Current state: %s' % str(current_state))
            logger.info('Action: %s' % str(action))
            logger.info('Next state: %s' % str(next_state))
            logger.info('Reward: %s' % str(reward))
            logger.info('Done: %s' % str(done))
            
            ##
            current_state = next_state
            ##
            total_reward+=reward
            logger.info("Current episode reward: %s " % str(total_reward))

            ## Save memory
            train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
            train_file.flush()
            
    agent.save("%s.h5" %filename)
    train_file.close()
