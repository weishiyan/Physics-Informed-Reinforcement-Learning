import random
import gym
import numpy as np
import math
from collections import deque
import tensorflow as tf
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Adam = tf.keras.optimizers.Adam


# from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4
MAX_TRAIN_REWARD = x_threshold * theta_threshold_radians
SEQ_LEN = 8


def loss(rewards):
    return (rewards - MAX_TRAIN_REWARD)**2


def train_reward(env):
    x, x_dot, theta, theta_dot = env.state
    # Custom reward for training
    # Product of Triangular function over x-axis and θ-axis
    # Min reward = 0, Max reward = env.x_threshold * env.θ_threshold_radians
    x_upper = env.x_threshold - x
    x_lower = env.x_threshold + x
    r_x     = max(0, min(x_upper, x_lower))

    theta_upper = env.theta_threshold_radians - theta
    theta_lower = env.theta_threshold_radians + theta
    r_θ     = max(0, min(theta_upper, theta_lower))
    return r_x * r_θ


def muEpisode(env):
    l = tf.convert_to_tensor(0.0)
    for frames in range(BATCH_SIZE):
        act = action(env.state)
        a = 1 if act > 0 else -1
        s, r, done, _ = env.step(a)
        if done:
            break
        l += loss(train_reward(env))
    return l


def action(state):
    state = np.reshape(state, [1, observation_space])
    # print(model(state))
    # print(model.predict(state))
    return (3 + model(state)) / 2


def episode(env):
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    step = 0
    total_reward = 0
    while True:
        step += 1
        with tf.GradientTape() as gt:
            gt.watch(model.trainable_variables)
            reward = muEpisode(env)
        grad = gt.gradient(reward, model.trainable_variables)
        total_reward += reward
        print(grad)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return total_reward


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    # score_logger = ScoreLogger(ENV_NAME)
    global observation_space
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    # dqn_solver = DQNSolver(observation_space, action_space)
    global model
    model = Sequential()
    model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(1, activation="tanh"))
    global optimizer
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss="mse", optimizer=optimizer)

    run = 0
    while True:
        total_reward = episode(env)
        print(f"Episode: {run} | Score: {total_reward:6.2f} | ", end="")

        score_mean = test(env)
        print("Mean score over 100 test episodes: {:6.2f}".format(score_mean))
        if score_mean > env.reward_threshold:
            print("CartPole-v0 solved!")
            break
        run += 1



# def test(env)
#     score_mean = 0
#     env.testmode()
#     for i in range(100):
#         total_reward = episode!(env)
#         score_mean += total_reward / 100
#     env.testmode(False)
#     return score_mean
