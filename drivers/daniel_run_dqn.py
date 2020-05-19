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

from agents.daniel_dqn import DQN

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
    reward_method = "Time"
    month = time.localtime().tm_mon
    day = time.localtime().tm_mday
    hour = time.localtime().tm_hour
    min = time.localtime().tm_min
    filename = "cartpole_episode%s_memory%s_%s_%s_%s_%s_%s" % \
        (EPISODES, NSTEPS, reward_method, month, day, hour, min)
    train_file = open('./data/'+filename, 'w')
    train_writer = csv.writer(train_file, delimiter = " ")

    def _linear(x, threshold):
        return 1 - (math.fabs(x)/threshold)

    logger.info('Testing ' + reward_method + ' reward method')
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward=0
        done = False
        while done!=True:
            action = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)

            # Incorporating environment data in reward structure
            # Linear reward map
            # Angle at which to fail the episode
            # self.theta_threshold_radians = 12 * 2 * math.pi / 360
            # self.x_threshold = 2.4
            if reward_method=="Linear":
                x, x_dot, theta, theta_dot = current_state
                x_threshold = env.x_threshold
                theta_threshold = env.theta_threshold_radians
                x_reward = _linear(x, x_threshold)
                theta_reward = _linear(theta, theta_threshold)

                # Correlation coefficient against total reward
                reward = 0.426 * x_reward + 0.574 * theta_reward
            elif reward_method=="Time":
                x, x_dot, theta, theta_dot = current_state
                x_threshold = env.x_threshold
                theta_threshold = env.theta_threshold_radians
                theta_time = (theta_threshold - math.fabs(theta)) / theta_dot
                theta_reward = math.fabs(math.tanh(theta_time))
                x_time = (x_threshold - math.fabs(x)) / x_dot
                x_reward = math.fabs(math.tanh(x_time))

                # Correlation coefficient
                reward = 0.3 * x_reward + 0.7 * theta_reward
            else:
                pass

            # logger.info('Current state: %s' % str(current_state))
            # logger.info('Action: %s' % str(action))
            # logger.info('Next state: %s' % str(next_state))
            # logger.info('Reward: %s' % str(reward))
            # logger.info('Done: %s' % str(done))

            agent.remember(current_state, action, reward, next_state, done)
            agent.train()


            ##
            current_state = next_state
            ##
            total_reward += reward


            ## Save memory
            train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
            train_file.flush()

        logger.info("Current episode reward: %s " % str(total_reward))

    agent.save(filename+'_agent')
    train_file.close()
