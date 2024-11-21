# Line Generation Neural Network


import pandas as pd
import numpy as np

df = pd.read_csv('../data/historic_clean.csv')



'''
        NEURAL NETWORK SETUP
- Split (80/20) and shuffle data
- Define layer dimensions
- initialize parameters
'''
# Data Split

# Initialize params
def __init__(layers):
    np.random.seed(1)
    params = {}
    L = len(layers)
    # He initialization
    # formula: w = N(0, sqrt(2/n_l-1)
    # N = normal distribution
    # l = layer
    # n_l-1 = neuron count in prev layer
    for curr in range(1, L):
        params['W' + str(curr)] = np.random.randn(layers[curr], layers[curr - 1]) * np.sqrt(2 / layers[curr - 1])
        # initialize bias at 0
        params['b' + str(curr)] = np.zeros((layers[curr], 1))
    print(params)
    return params

'''
        FORWARD PROP
- Define input -> output algorithm
- compute cost function
'''
def forward(X, params):
    cache = {}
    L = len(params)
    cache["X"] = x



'''
        ACTIVATION FUNCTIONS
- Define linear and relu algorithms along w/ their derivatives
'''
def relu




'''
        BACK PROP
- Move back through network driven by cost value
- Update parameters
'''


'''
        TRAINING AND OPTIMIZATION
'''
