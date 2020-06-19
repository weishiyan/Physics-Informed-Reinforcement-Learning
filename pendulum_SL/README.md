# Building a Physics Informed Agent Using Supervised Learning
(include short description of what is happening here)
### Overview
(talk about lbfgs vs tf)
### Getting Started

Navigate to the *pendulum_SL* directory:

`$ cd pendulum_SL/`

To begin running, explicity define the number of epochs to run

`$ python pendulum.py <tf_epochs> <nt_epochs>`

Setting either of these values to zero will skip the method
For example:

`$ python pendulum.py 0 100` 

will run only the lbfgs method for 100 epochs and ignore tensorflow

To clean up the *plots* directory run

`$ python clean_dir.py`

****EDIT README (this is a rough draft)
1. Include description of plots that are saved in the plots directory (maybe show examples here)
2. give description of data sets in data directory

