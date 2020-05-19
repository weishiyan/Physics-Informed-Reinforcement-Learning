# -*- coding: utf-8 -*-
import random, math
import gym
from gym.utils import seeding
from tqdm import tqdm
import time
import numpy as np
import keras

##
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)

from agents.alexis_diff_dqn import DQN

import csv

import tensorflow as tf
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Adam = tf.keras.optimizers.Adam

GAMMA = 0.95
env = gym.make("CartPole-v0")
state_size = 4
num_actions = env.action_space.n
# building a new neural network here
network = Sequential([
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='tanh')
]) # third layer is a 'softmax'
network.compile(loss='mse',optimizer=Adam())

# A.Mills Functions: -----------------------------------------

def train_reward(current_state): # from Julia script
    limits = np.empty((2,2), dtype = float)
    x, x_dot, theta, theta_dot = current_state[0], current_state[1], current_state[2], current_state[3]
    # Scale reward based on how close theta/x are to their respective thresholds
    limits[0,0] = env.x_threshold - x # x upper
    limits[1,0] = env.x_threshold + x # x lower
    x_r = max(float(0),min(limits[:,0]))
    
    limits[0,1] = env.theta_threshold_radians - theta # theta upper
    limits[1,1] = env.theta_threshold_radians + theta # theta lower
    theta_r = max(float(0),min(limits[:,1]))

    reward = x_r*theta_r
    return reward

# JULIA SCRIPT: in julia Flux.back!(loss, update_param) is backprop. so our gradientTape?
def replay(episode_memory):
    rewards = []
    for step in range(episode_memory.shape[0]):
        rewards.append(train_reward(episode_memory[step,0]))
    rewards = np.array(rewards)
    with tf.GradientTape() as t:
        t.watch(agent.model.trainable_variables)
    gradient = t.gradient(loss(rewards),agent.model.trainable_variables)
    return agent.grad_train_model(gradient)

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards

def loss(rewards): # converted to python from Julia
    episode_length = len(rewards)
    maximum_r = np.ones((episode_length), dtype=float)*MAX_TRAIN_REWARD
    reward_0 = (maximum_r)
    rewards = (rewards)
    loss = np.square(np.subtract(rewards,reward_0))
    loss_np = np.asarray(loss, np.float32)
    loss_tf = tf.convert_to_tensor(loss_np, np.float32)
    #loss = tf.reduce_sum(tf.square(reward_0 - rewards))
    return loss_tf

def seed(seed=None):
       np_random, seed = seeding.np_random(seed)
       return [seed]

def step(state, action): # from cartpole openai
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = (masspole + masscart)
        length = 0.5 # actually half the pole's length
        polemass_length = (masspole * length)
        force_mag = 10.0
        tau = 0.02  # seconds between state updates
        kinematics_integrator = 'euler'
        # Angle at which to fail the episode
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([x_threshold * 2,
                         np.finfo(np.float32).max,
                         theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        seed()
        viewer = None
        #state = None
        steps_beyond_done = None

        assert action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = state
        x, x_dot, theta, theta_dot = state
        force = force_mag if action==1 else -force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
        xacc  = temp - polemass_length * thetaacc * costheta / total_mass
        if kinematics_integrator == 'euler':
            x  = x + tau * x_dot
            x_dot = x_dot + tau * xacc
            theta = theta + tau * theta_dot
            theta_dot = theta_dot + tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + tau * xacc
            x  = x + tau * x_dot
            theta_dot = theta_dot + tau * thetaacc
            theta = theta + tau * theta_dot
        state = (x,x_dot,theta,theta_dot)
        done =  x < -x_threshold \
                or x > x_threshold \
                or theta < -theta_threshold_radians \
                or theta > theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif steps_beyond_done is None:
            # Pole just fell!
            steps_beyond_done = 0
            reward = 1.0
        else:
            if steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            steps_beyond_done += 1
            reward = 0.0

        return np.array(state), reward, done, {}

def get_action(network, state, num_actions): # for defined network above (a little diff from our model)
    softmax_out = network(state.reshape((1, -1)))
    selected_action = np.random.choice(num_actions, p=softmax_out.numpy()[0])
    return selected_action

def update_network(network, rewards, states, actions, num_actions): # for defined network
    reward_sum = 0
    discounted_rewards = []
    for reward in rewards[::-1]:  # reverse buffer r
        reward_sum = reward + GAMMA * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards)
    # standardise the rewards
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    states = np.vstack(states)
    loss = network.train_on_batch(states, discounted_rewards)
    return loss
# A.Mills functions end -----------------------------------------


# FROM SLACK CHANNEL
#def function(x,y):
#    output = 1.0 
#    for i in range(y):
#        if i > 1 and i < 5:
#            output = tf.multiply(output, x)
#    return output
#
#def gradient(x,y):
#    with tf.GradientTape() as tape: # keeps track of gradients
#        tape.watch(x)
#        out = function(x,y)
#    return tape.gradient(out,x)


if __name__ == "__main__":
    import sys

    ###########
    ## Train ##
    ###########
    EPISODES = 100
    NSTEPS   = 200
    MAX_TRAIN_REWARD = env.x_threshold*env.theta_threshold_radians


    #######################
    ## Setup environment ##
    #######################
    estart = time.time()
    #env = gym.make('CartPole-v0')
    env._max_episode_steps=NSTEPS
    env.seed(1)
    end = time.time()
    gamma = 0.95
    #logger.info('Time init environment: %s' % str( (end - estart)/60.0))
    #logger.info('Using environment: %s' % env)
    #logger.info('Observation_space: %s' % env.observation_space.shape)
    #logger.info('Action_size: %s' % env.action_space)

    #################
    ## Setup agent ##
    #################
    agent = DQN(env)


    
    ## Save infomation ##
    filename = "keras_tensor_dqn_cartpole_mse_episode%s_memory%s_" % (str(EPISODES),str(NSTEPS))
    train_file = open(filename, 'w')
    train_writer = csv.writer(train_file, delimiter = " ")
    
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        #logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        #rewards = []
        #states = []
        #actions = []
        loss_memory = []
        #state_memory = []
        episode_mem = []
        total_reward=0
        total_rewards = []
        done = False
        while done!=True:
            state = current_state
          
            action = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)
            episode_mem.append([current_state,action,reward,next_state])
            loss_memory.append(agent.train())

            agent.remember(current_state, action, reward, next_state, done)


            if done:
                episode_mem = np.array(episode_mem)
                loss_memory = np.array(loss_memory)

                total_rewards.append(episode_mem[:,2].sum())
                discounted_rewards = discount_rewards(episode_mem[:,2],gamma)

                replay(episode_mem)

            #state = (current_state)
            #action = get_action(network, state, num_actions)
            #new_state, reward, done, _ = step(current_state,action)
            #states.append(state)
            #rewards.append(reward)
            #actions.append(action)
            #if len(rewards) > 1:
            #    loss = update_network(network, rewards, states, actions, num_actions)
            ##
            current_state = next_state
            ##
            total_reward+=reward
            #logger.info("Current episode reward: %s " % str(total_reward))

            ## Save memory
            train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
            train_file.flush()
            if done:
                
                tot_reward = sum(rewards)
            else:
                pass
                
       
    #agent.save("%s.h5" %filename)
    train_file.close()
