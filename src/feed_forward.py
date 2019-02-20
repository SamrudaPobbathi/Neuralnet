# Originaly created by Tudor Berariu in 2016 for the Machine Learning class
# Artificial Intelligence and Multi-Agent Systems Laboratory
# Faculty of Automatic Control and Computer Science
# University Politehnica of Bucharest

# Modified by Alexandru Iulian Orhean in 2018
# CS554 Data-Intensive Computing class
# Data-Intensive Distributed Systems Laboratory
# Illinois Institute of Technology

import numpy as np
from layer import Layer

class FeedForward:
    # FeedForward neural network constructor, that receives as a paramenter the
    # number of inputs in the neural network and a list of layer descriptions,
    # each element from the list containing the output size of the layer and a
    # sigmoid transfer function. The number of inputs of a layer is going to
    # match the number of outputs of the previous layer, or the neural network
    # inputs if the layer is the first layer
    def __init__(self, input_size, layers_info):
        self.layers = []
        last_size = input_size
        for layer_size, transfer_function in layers_info:
            self.layers.append(Layer(last_size, layer_size, transfer_function))
            last_size = layer_size

    def forward(self, inputs):
        last_input = inputs
        for layer in self.layers:
            last_input = layer.forward(last_input)
        return last_input

    def backward(self, inputs, output_error):
        crt_error = output_error
        for layer_no in range(len(self.layers)-1, 0, -1):
            crt_layer = self.layers[layer_no]
            prev_layer = self.layers[layer_no - 1]
            crt_error = crt_layer.backward(prev_layer.outputs, crt_error)
        self.layers[0].backward(inputs, crt_error)

    # Update the weights of the layer according to the learning rate and
    # the gradient of the calculated error
    def update_parameters(self, learning_rate):
        for i in range(len(self.layers)):
            g_weight = learning_rate * self.layers[i].g_weights
            self.layers[i].weights = self.layers[i].weights - g_weight

    def to_string(self):
        return " -> ".join(map(lambda l: l.to_string(), self.layers))
