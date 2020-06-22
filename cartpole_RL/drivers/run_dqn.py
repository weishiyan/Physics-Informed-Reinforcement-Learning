import gym
from tqdm import tqdm
import time
import csv
import logging

from cartpole_RL.agents.dqn import DQN

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)

if __name__ == "__main__":
    # Training parameters
    EPISODES = 10
    NSTEPS = 200

    # Environment variables
    estart = time.time()
    env = gym.make('CartPole-v0')
    env._max_episode_steps = NSTEPS
    env.seed(1)
    end = time.time()
    logger.info('Time init environment: %s' % str((end - estart)/60.0))
    logger.info('Using environment: %s' % env)
    logger.info('Observation_space: %s' % env.observation_space.shape)
    logger.info('Action_size: %s' % env.action_space)

    # Agent setup
    agent = DQN(env)

    # Saving the information to ./data folder
    filename = "./data/dqn_cartpole_mse_episode%s_memory%s" % (str(EPISODES),
                                                        str(NSTEPS))
    train_file = open(filename, 'w')
    train_writer = csv.writer(train_file, delimiter=" ")

    # Training start
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward = 0
        done = False
        while done is not True:
            # Taking an action
            action = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(current_state, action, reward, next_state, done)
            agent.train()

            logger.info('Current state: %s' % str(current_state))
            logger.info('Action: %s' % str(action))
            logger.info('Next state: %s' % str(next_state))
            logger.info('Reward: %s' % str(reward))
            logger.info('Done: %s' % str(done))

            current_state = next_state
            total_reward += reward
            logger.info("Current episode reward: %s " % str(total_reward))

            # Save memory
            train_writer.writerow([current_state, action, reward, next_state,
                                  total_reward, done])
            train_file.flush()

    # Saving the agent information
    # agent.save("./data/%s.h5" % filename)
    train_file.close()
