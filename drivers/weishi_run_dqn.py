for i in range(1):
    
    # -*- coding: utf-8 -*-
    import random, math
    import gym
    from tqdm import tqdm
    import time
    import math
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
        EPISODES = 250
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
        filename = "dqn_cartpole_mse_episode%s_memory%s_try%s" % (str(EPISODES),str(NSTEPS),str(i+1))
        ## added time stamp
        train_file = open(filename, 'w')
        train_writer = csv.writer(train_file, delimiter = " ")
        
        def add_reward(param, param_type, lam = 0.5):
            if param_type == "position":
                param_boundary = 2.4
            elif param_type == "angle":
                param_boundary = math.pi/15
            factor = 1/(lam*param_boundary)
            param_tl = param*factor
            new_reward = tukeylambda.pdf(param_tl, lam) 
            return new_reward
    
        for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
            logger.info('Starting new episode: %s' % str(e))
            current_state = env.reset()
            total_reward=0
            done = False
            while done!=True:
                action = agent.action(current_state)
                next_state, reward, done, _ = env.step(action)
                
                ##add reward
                x = current_state[0]
                theta = current_state[2]
                new_x_reward = add_reward(x, param_type = "position")
                new_theta_reward = add_reward(theta, param_type = "angle")
                new_reward = reward + new_x_reward + new_theta_reward
                
                reward = new_reward
                
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
                
            ##weishi added
            ##count number of sucessful episodes
            #if total_reward >=195:
                #count_success_episode+=1
                    
            ##break the loop if reward reach over 195 for 10 times
            #if count_success_episode == 10:
            #    break
            #else:
            #    continue
            
        agent.save("%s.h5" %filename)
        train_file.close()
