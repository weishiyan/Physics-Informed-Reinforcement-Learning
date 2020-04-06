# Reinforcement Learning environments and agents/policies used for the CDi Capstone 2020 project

## Software Requirement
* Python 3.7 
* The environemnt framework is built of [OpenAI Gym](https://gym.openai.com) 
* Additional python packages are defined in the setup.py 
* For now, we assumes you are running at the top directory 

## Installing 
* Pull code from repo
```
git clone https://gitlab.pnnl.gov/schr476/cdi_capstone_2020.git
```
* Install cdi_capstone_2020 (via pip):
```
cd cdi_capstone_2020
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
