# Originaly created by Tudor Berariu in 2016 for the Machine Learning class
# Artificial Intelligence and Multi-Agent Systems Laboratory
# Faculty of Automatic Control and Computer Science
# University Politehnica of Bucharest

# Modified by Alexandru Iulian Orhean in 2018
# CS554 Data-Intensive Computing class
# Data-Intensive Distributed Systems Laboratory
# Illinois Institute of Technology

import numpy as np
from argparse import ArgumentParser
from datetime import datetime

from data_loader import load_mnist
from feed_forward import FeedForward
from transfer_functions import identity, logistic, hyperbolic_tangent

# Evaluate the accuracy of a neural network for a given set of images
def eval_nn(nn, imgs, labels, maximum = 0):
    correct_no = 0
    how_many = imgs.shape[0] if maximum == 0 else maximum
    for i in range(imgs.shape[0])[:how_many]:
        y = np.argmax(nn.forward(imgs[i]))
        t = labels[i]
        if y == t:
            correct_no += 1

    return float(correct_no) / float(how_many)

# Train a neural network and print once in a while the accuracy of the model
def train_nn(nn, data, args):
    start = datetime.now()
    for i in np.random.permutation(data["train_no"]):
        nn_output = nn.forward(data["train_imgs"][i])
        expected_output = np.zeros(nn_output.shape)
        expected_output[data["train_labels"][i]] = 1
        error = nn_output - expected_output
        nn.backward(data["train_imgs"][i], error)
        nn.update_parameters(args.learning_rate)

        # Evaluate the network
        if i % args.eval_every == 0:
            test_acc = \
                eval_nn(nn, data["test_imgs"], data["test_labels"])
            train_acc = \
                eval_nn(nn, data["train_imgs"], data["train_labels"], 5000)
            current = datetime.now()
            timediff = current - start
            timemillisec = (timediff.seconds * 1000) \
                    + (timediff.microseconds // 1000)
            print("Train acc: %2.6f ; Test acc: %2.6f, Time elapsed: %d ms" \
                    % (train_acc, test_acc, timemillisec))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type = float, default = 0.001,
                        help="Learning rate")
    parser.add_argument("--eval_every", type = int, default = 200,
                        help="Evaluation rate")
    args = parser.parse_args()

    mnist = load_mnist()
    input_size = mnist["train_imgs"][0].size

    # Create a neural network with two layers
    nn = FeedForward(input_size, [(300, logistic), (10, identity)])
    # print(nn.to_string())

    train_nn(nn, mnist, args)
