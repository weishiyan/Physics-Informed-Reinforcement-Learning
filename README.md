# Reinforcement Learning environments and agents/policies used for the CDi Capstone 2020 project

## Software Requirement
* Python 3.7 
* The cartpole environment framework and pendulum data set is built using [OpenAI Gym](https://gym.openai.com) 
* Additional python packages are defined in the setup.py  

## Installing 

## Directory Organization
```
cartpole_RL                            : cartpole problem using reinforcement learning
 ├── setup.py                          : python dependency info
 ├── drivers                           : a folder contains RL steering scripts
 |   └── run_dqn.py                    : steering script to debug changes in the dqn agent using cartpole en
 ├── agents                            : a folder contains agent codes
 |   └── dqn.py                        : agent used for discrete action space 
 ├── cfg                               : a folder contains the agent and environment configuration
 ├── utils                             : a folder contains utilities

pendulum_SL                            : pendulum problem using supervised learning
(organize this)
```
