# deepl_p2
Project 2 - Mini Deep Learning Framework

Run test.py will run two models - 

1) Vanilla model that was outlined in the project
1) Model we built ourselves with cross entropy loss


**module.py** - contains every class inheriting from the module class. These modules are the core of our framework and include linear layers, non-linear activation functions, loss functions, and a sequential class for stacking the layers. 

**solvers.py** - contains one solver that can perform gradient descent, mini-batch gradient descent and stochastic gradient descent depending on the input parameters. 

**train.py** - contains functions that manage the training and testing of the deep learning models.

**test.py** - run file built to the project specifications. 

**utils.py** - utility file that contains functions to generate training and testing data, calculate errors and save models. 

**test** - contains a unittest file

**model** - where output of the models is saved
