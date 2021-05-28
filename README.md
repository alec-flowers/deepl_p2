# Project 2 - Mini Deep Learning Framework
The goal of this project was to implement a mini-deep learning framework using only pytorch's tensor operations and the standard math library.


Run test.py will run two models for 200 epochs, print the training loss along the way, and output the final train and test accuracy using the mini deep learning framework that we built. The two models are: 

1) Vanilla model that was outlined in the project
1) Model we built ourselves with cross entropy loss

### File Explanation
**module.py** - contains every class inheriting from the module class. These modules are the core of our framework and include linear layers, non-linear activation functions, loss functions, and a sequential class for stacking the layers. 

**solvers.py** - contains one solver that can perform gradient descent, mini-batch gradient descent and stochastic gradient descent depending on the input parameters. 

**train.py** - contains functions that manage the training and testing of the deep learning models.

**test.py** - run file built to the project specifications. 

**utils.py** - utility file that contains functions to generate training and testing data, calculate errors and save models. 

**test** - contains a unittest file

**model** - where output of the models is saved
