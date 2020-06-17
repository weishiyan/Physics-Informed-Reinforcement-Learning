# Physics Informed Machine Learning 
Our objective is to improve machine learning (ML) methods by devising an algorithm that utilizes the laws of physics. Incorporating physics through auto-differentiation in ML has a number of benefits including the need for less data and the inclusion of well-defined physics equations. This will ultimately increase the rate of convergence and prediction accuracy. This technology will aid in directing research projects in that systems can be better understood through predictions made by physics informed machine learning. 

This repository is divided into two parts, each of which explore different machine learning approaches. 

#### Part I: Solving the Cartpole Problem Using Reinforcement Learning (RL)
Reinforcement learning is utilized to train an "agent" and solve the cartpole problem. The problem is considered solved once the agent is capable of making an informed decision to balance the pole upright. 
![solved cartpole demo](https://github.com/schr476/uw_capstone_2020/tree/master/cartpole_RL/static/solved_cartpole.gif)

For more information click [here](https://github.com/schr476/uw_capstone_2020/tree/master/cartpole_RL)

#### Part II: Building a Pendulum Environment Using Supervised Learning (SL)
In order to implement the pendulum problem in RL, an environment must first be defined to respond to the agent's actions. An environment can be built using supervised learning responding to both a well-defined neural network and auto-differentiation that solve the simple pendulum equation: 
$$\frac{d^2\theta}{d^2t}+\frac{g}{l}\sin\theta=0$$


## Directory Organization
```
cartpole_RL                            : cartpole problem using reinforcement learning (RL)
 ├── setup.py                          : python dependency info
 ├── drivers                           : a folder contains RL steering scripts
 |   └── run_dqn.py                    : steering script to debug changes in the dqn agent using cartpole en
 ├── agents                            : a folder contains agent codes
 |   └── dqn.py                        : agent used for discrete action space 
 ├── cfg                               : a folder contains the agent and environment configuration
 ├── utils                             : a folder contains utilities

pendulum_SL                            : pendulum problem using supervised learning (SL) to build environment for RL
 ├ Deep_Deterministic_Policy_Gradients_notebooks
 │      ├ DDPG.ipynb
 │      ├ ddpg_agent.py
 │      └ model.py
 ├ Physics_Informed_Neural_Networks_notebooks
 │      ├ PINNs_pendulum_daniel.ipynb
 │      └ PINNs_pendulum_weishi.ipynb
 ├ pendulum_alexis
 │       ├ data				                : data sets for SL training
 │       ├ lbfgs.py			              : limited memory Broyden-Fletcher-Goldfarb-Shanno optimization algorithm 
 │       ├ logs				                : collected output reports from runs
 │       ├ pinn.py			              : Physics informed neural network (NN) code
 │       ├ plot.py			              : plotting loss and predictions
 │       ├ plots			                : collected plots from plot.py
 │       ├ prep_data.py			          : work-up data for use in NN
 │       └ run_pendulum.py		        : build environment for pendulum problem

docs
  ├ Gantt_Chart.pdf
  ├ Reinforcement Learning.pdf
  └ use_cases.ipynb
```
## Software Requirement
* Python 3.7 
* The cartpole environment framework and pendulum data set is built using [OpenAI Gym](https://gym.openai.com) 
* Additional python packages are defined in the setup.py  

## Installing 
```
git clone https://github.com/schr476/uw_capstone_2020.git

cd uw_capstone_2020

pip install -e . --user
```

## References
* Surma, G. ["Cartpole - Introduction to Reinforcement Learning (DQN - Deep Q-Learning)"](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288)

* Supervised learning code built from [PINNs-TF2.0](https://github.com/pierremtb/PINNs-TF2.0)
