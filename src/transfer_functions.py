# Originaly created by Tudor Berariu in 2016 for the Machine Learning class
# Artificial Intelligence and Multi-Agent Systems Laboratory
# Faculty of Automatic Control and Computer Science
# University Politehnica of Bucharest

# Modified by Alexandru Iulian Orhean in 2018
# CS554 Data-Intensive Computing class
# Data-Intensive Distributed Systems Laboratory
# Illinois Institute of Technology

import numpy as np

def identity(x, derivative = False):
    res = None

    if derivative:
        res = np.ones(x.shape)
    else:
        res = x
    
    return res

def logistic(x, derivative = False):
    res = None

    if derivative:
        res =  x * (1 - x)
    else:
        res =  1 / (1 + np.e ** (-x))

    return res

def hyperbolic_tangent(x, derivative = False):
    res = None

    if derivative:
        res =  1 - x ** 2
    else:
        res = (np.e ** x - np.e ** (-x)) / (np.e ** x + np.e ** (-x))

    return res

