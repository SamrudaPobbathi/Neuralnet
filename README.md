**Illinois Institute of Technology**  
**CS554 Data-Intensive Computing (Fall 2018)**  
**uDNN Deep Learning Project**  
Project mentor: Alexandru Iulian Orhean (aorhean@hawk.iit.edu)

# Description

This repository contains the implementation of a simple two layer neural 
network that learns how to recognize the MNIST hand written digits. The neural
network and the utility functions are implemented in Python 3.6.5 and you can
feel free to use the code as the base of your CS554 project.

# Requirements

The MNIST data is assumed to reside in the *mnist-data* directory and can be
created and populated through a utility script. The implementation was tested 
on Ubuntu 16.04 and Ubunut 18.04, but it should work on any OS that has the 
following python libraries installed: python-numpy, python-matplotlib and 
python-mnist.

# How to build and run

All the command are assumed to be run from the base directory of the 
repository. Before running the neural network application install the required
python libraries:

$ sudo apt update  
$ sudo apt install python3 python3-pip python3-matplotlib python3-numpy  
$ pip3 install python-mnist

The populate the MNIST data in the *mnist-data* directory by running:

$ python3 src/data_loader.py

Now you can run the neural network application, that learns continuously with a
specific learning rate *m* and prints the accuracy of the model every *n* 
training iterations, where *n* can be passed as an argument for the program:

$ python3 src/train.py  
$ python3 src/train.py --learning_rate *m* --eval_every *n*

When you start implementing the low precision version of the neural network, 
you should consider analyzing not only the training speed but also low precision
has on the learning rate. The greater the learning rate ration the faster the
neuron trains, but the lower the ration the more accurate the training is.
