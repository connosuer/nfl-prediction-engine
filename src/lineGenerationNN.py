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

class NeuralNetwork:
    def __init__(self, layers, activation='relu', optimizer='momentum', learning=0.01, beta=0.9):
        self.layers = layers
        self.activation = activation
        self.optimizer = optimizer
        self.learning = learning
        self.beta = beta
        self.params = self._init_params()
        self.velo = self._init_velo()
        self.cache = {}
        self.gradient = {}

    def _activation_function(self, x):
        if x == 'relu':
            return self._relu, self._relu_derivative
        else:
            raise ValueError()

    def _init_params(self):
        np.random.seed(0)
        params = {}
        L = len(self.layers)
        for layer in range(1, L):
            params['W' + str(layer)] = \
                (np.random.randn(self.layers[layer], self.layers[layer - 1])
                 ) * np.sqrt(2 / self.layers(layer - 1))
            params['b' + str(layer)] = np.zeros((self.layers[layer], 1))
            return params


'''
        FORWARD PROP
- Define input -> output algorithm
- compute cost function
'''





'''
        ACTIVATION FUNCTIONS
- Define linear and relu algorithms along w/ their derivatives
'''



'''
        BACK PROP
- Move back through network driven by cost value
- Update parameters
'''


'''
        TRAINING AND OPTIMIZATION
'''
