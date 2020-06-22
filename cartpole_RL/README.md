# Reinforcement Learning environments and agents/policies for the cartpole problem

## Software Requirement
* Python 3.7
* The environment framework is built of [OpenAI Gym](https://gym.openai.com). Cartpole environment can be [found here](https://gym.openai.com/envs/CartPole-v0/).
* Packages listed in setup.py

## Introduction
> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

~ From OpenAI website

## Instructions
1. Make sure to install the packages from setup.py
2. Run from this directory `$ cartpole_RL`
3. There are two files within drivers folder. One is `run_dqn.py`, the other `modified_run_dqn.py`. Choose which one you would like to run.
4. Run the wanted file from the current directory by using `python drivers/run_dqn.py`.

## Notes
1. There will be an update on the status of training within the console. The training length can be changed within the driver files.
2. Training parameters are obtained from the `dqn_setup.json` file within `cfg` folder.
3. The training outputs data to the `data` folder. If the data folder does not exist, an error will be thrown.
