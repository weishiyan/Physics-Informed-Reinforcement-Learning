# Building a Physics Informed Agent Using Supervised Learning
(include short description of what is happening here)
### Overview
(talk about lbfgs vs tf)
### Getting Started

Navigate to the *pendulum_SL* directory:

`$ cd pendulum_SL/`

To begin running, 
1. define the neural network type
        nn = standard neural network
        pinn = physics informed neural network
2. explicity define the number of epochs to run (must be an integer)
3. select the dataset 
        50 = pendulum length of 50 (as defined by openAI)
        100 = pendulum length of 100 (as defined by openAI)

`$ python pendulum_SL.py <nn or pinn> <num epochs> <50 or 100>`

Setting either of these values to zero will skip the method

For example:

`$ python pendulum_SL.py pinn 500 100` 

will run the physics informed neural network algorithm for 500 epochs on the dataset following the oscillation of a pendulum with a length of 100

To clean up the *plots* directory run

`$ python clean_dir.py`

****EDIT README (this is a rough draft)
1. Include description of plots that are saved in the plots directory (maybe show examples here)
2. give description of data sets in data directory

