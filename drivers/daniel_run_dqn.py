# -*- coding: utf-8 -*-
import math
import gym
from tqdm import tqdm
import time

##
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)

from agents.dqn import DQN

import csv

if __name__ == "__main__":

    ###########
    ## Train ##
    ###########
    EPISODES = 500
    NSTEPS = 200
    #######################
    ## Setup environment ##
    #######################
    estart = time.time()
    env = gym.make('CartPole-v0')
    env._max_episode_steps = NSTEPS
    env.seed(1)
    end = time.time()
    logger.info('Time init environment: %s' % str((end-estart)/60.0))
    logger.info('Using environment: %s' % env)
    logger.info('Observation_space: %s' % env.observation_space.shape)
    logger.info('Action_size: %s' % env.action_space)

    #################
    ## Setup agent ##
    #################
    agent = DQN(env)

    ## Save infomation ##
    reward_method = "None"
    month = time.localtime().tm_mon
    day = time.localtime().tm_mday
    hour = time.localtime().tm_hour
    min = time.localtime().tm_min
    filename = "cartpole_episode%s_memory%s_%s_%s_%s_%s_%s" % \
        (EPISODES, NSTEPS, reward_method, month, day, hour, min)
    train_file = open(filename, 'w')
    train_writer = csv.writer(train_file, delimiter = " ")

    def _normal(std, x):
        return math.exp(-(x**2)/(2*std**2))/(std*math.sqrt(2*math.pi))


    def _step(bin, x, threshold):
        return 1-(1/bins * math.floor(math.fabs(x)/(threshold/bins)))

    def _linear(x, threshold):
        return 1 - (x/threshold)

    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward=0
        done = False
        while done!=True:
            action = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)

            # Incorporating environment data in reward structure
            # Trying out a normal distribution of rewards
            if reward_method=="Normal":
                x = current_state[0]
                theta = current_state[2]
                std_x = env.x_threshold / 3  # dividing by 3 for 3*sigma
                std_theta = env.theta_threshold_radians / 3 # dividing by 3 for 3*sigma

                # normalizing the reward to 1
                x_reward = _normal(std_x, x) / _normal(std_x, 0)
                theta_reward = _normal(std_theta, theta) / _normal(std_theta, 0)
                reward = (x_reward + theta_reward) / 2

            # Dividing stepwise regions of x and theta with certain number of bins
            elif reward_method=="Step":
                x = current_state[0]
                theta = current_state[2]
                x_threshold = env.x_threshold
                theta_threshold = env.theta_threshold_radians
                bins = 5
                x_reward = _step(bins, x, x_threshold)
                theta_reward = _step(bins, theta, theta_threshold)
                reward = (x_reward + theta_reward) / 2

            # If the cart and the pole have the same sign, the reward is 1
            elif reward_method=="Same_sign":
                x = current_state[0]
                theta = current_state[2]
                if x < 0 and theta < 0:
                    reward = 1
                elif x > 0 and theta > 0:
                    reward = 1
                elif x == 0 and theta == 0:
                    reward = 1
                else:
                    reward == 0

            # Only thinking about the pole angle
            elif reward_method=="Normal_Angle":
                theta=current_state[2]
                std_theta = env.theta_threshold_radians / 3 # dividing by 3 for 3*sigma
                theta_reward = _normal(std_theta, theta) / _normal(std_theta, 0)
                reward = theta_reward
            else:
                pass

            agent.remember(current_state, action, reward, next_state, done)
            agent.train()


            ##
            current_state = next_state
            ##
            total_reward+=reward


            ## Save memory
            train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
            train_file.flush()

        logger.info("Current episode reward: %s " % str(total_reward))

    agent.save('data/'+filename)
    train_file.close()
