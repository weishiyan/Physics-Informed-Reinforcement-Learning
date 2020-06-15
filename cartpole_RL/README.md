# Reinforcement Learning environments and agents/policies for the cartpole problem

## Software Requirement
* Python 3.7 
* The environemnt framework is built of [OpenAI Gym](https://gym.openai.com) 
* Additional python packages are defined in the setup.py 
* For now, we assumes you are running at the top directory 

## Installing 
* Pull code from repo
```
git clone https://github.com/schr476/uw_capstone_2020.git
```
* Install uw_capstone_2020 (via pip):
```
cd uw_capstone_2020
pip install -e . --user
```
## Directory Organization
```
├── setup.py                          : python dependency info
├── drivers                            : a folder contains RL steering scripts
|   └── run_dqn.py                    : steering script to debug changes in the dqn agent using cartpole en
├── agents                            : a folder contains agent codes
    └── dqn.py                        : agent used for discrete action space 
├── cfg                               : a folder contains the agent and environment configuration
├── utils                             : a folder contains utilities
          
```
