# Building a Physics Informed Neural Network (PINN) Model

A PINN is used to create a model suitable for solving the pendulum problem. The model is developed using traditonal machine learning techniques in addition to known physics defined by partial differential equations.  

### Getting Started

Navigate to the *pendulum_SL* directory:

`$ cd pendulum_SL/`

To begin running, 
1. Define the neural network model type
        nn = standard neural network optimizes based off of MSE from NN model
        pinn = physics informed neural network optimizes based off of MSE from both NN and PINN model
2. Explicity define the number of epochs to run (must be an integer)
3. Select the dataset used for training
        50 = pendulum length of 50 (as defined by openAI)
        100 = pendulum length of 100 (as defined by openAI)

`$ python pendulum_SL.py <nn or pinn> <num epochs> <50 or 100>`


For example, the following line will run the physics informed neural network algorithm for 500 epochs on the dataset following the oscillation of a pendulum with a length of 50
`$ python pendulum_SL.py pinn 500 50` 
A series of plots will be printed to the *plots* directory showing results after every 50 epochs. 

To remove plots from the *plots* directory run:

`$ python clean_dir.py`


