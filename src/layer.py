# Originaly created by Tudor Berariu in 2016 for the Machine Learning class
# Artificial Intelligence and Multi-Agent Systems Laboratory
# Faculty of Automatic Control and Computer Science
# University Politehnica of Bucharest

# Modified by Alexandru Iulian Orhean in 2018
# CS554 Data-Intensive Computing class
# Data-Intensive Distributed Systems Laboratory
# Illinois Institute of Technology

import numpy as np

class Layer:
    # Layer constructor, that receives as a parameter the number of inputs, the
    # number of outputs and a sigmoid transfer function
    def __init__(self, num_input, num_output, transfer_function):
        self.num_input = num_input
        self.num_output = num_output
        self.func = transfer_function
        
        # You can use the .astype function to change precision
        self.weights = np.random.normal(0, 
                np.sqrt(2.0 / float(self.num_output + self.num_input)),
                (self.num_output, self.num_input))

        self.biases = np.random.normal(0,
                np.sqrt(2.0 / float(self.num_output + self.num_input)),
                (self.num_output, 1))

        self.outputs = np.zeros((self.num_output, 1))
        self.g_weights = np.zeros((self.num_output, self.num_input))
        self.g_biases = np.zeros((self.num_output, 1))
        self.g_outputs = np.zeros((self.num_output, 1))

    # Calculate the output of a layer according to the given input and features
    def forward(self, inputs):
        self.outputs = self.func(np.dot(self.weights, inputs) + self.biases)
        return self.outputs

    # Calculate the gradient of the error for the layer features and for the
    # layer inputs
    def backward(self, inputs, errors):
        self.g_outputs = self.func(self.outputs, True)
        self.g_biases = errors * self.g_outputs
        self.g_weights = np.dot(errors * self.g_outputs, inputs.T)
        return np.dot(np.transpose(self.weights), self.g_biases)

    def to_string(self):
        return "(%s -> %s)" % (self.num_input, self.num_output)

