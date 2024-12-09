import pandas as pd
import numpy as np

class NeuralNetwork:
    '''
        NEURAL NETWORK SETUP
    ---------------------------
    - Initialize parameters
    - Set activation functions
    - Choose optimizer and learning settings
    '''
    def __init__(self, layers, activation='relu', output_activation='linear', optimizer='momentum', learning=0.01, beta=0.9):
        """
        Initializes the neural network.
        Args:
            layers: List containing the number of neurons in each layer (input, hidden, output).
            activation: Activation function to use ('relu' is supported).
            optimizer: Optimization method ('momentum' is supported).
            learning: Learning rate for gradient descent.
            beta: Momentum coefficient for velocity calculation.
        """
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.learning = learning
        self.beta = beta
        self.params = self._init_params()  # Initialize weights and biases
        self.velo = self._init_velo()  # Initialize velocity for momentum
        self.cache = {}  # Store intermediate results
        self.gradient = {}  # Store computed gradients
        self.activation_func, self.activation_derivative = self._activation_function(activation)  # Activation and its derivative

    def _activation_function(self, x):
        """
        Returns the activation function and its derivative based on the given name.
        Args:
            x: Name of the activation function.
        Returns:
            Tuple containing the activation function and its derivative.
        """
        if x == 'relu':
            return self._relu, self._relu_derivative
        else:
            raise ValueError("Check activation function")
    def _output_function(self, x):
        if x == 'linear':
            return self._linear
        else:
            raise ValueError("Check Output function")

    def _init_params(self):
        """
        Initializes weights and biases using He initialization.
        Returns:
            Dictionary of parameters.
        """
        np.random.seed(0)
        params = {}
        L = len(self.layers)
        for layer in range(1, L):
            params['W' + str(layer)] = (
                np.random.randn(self.layers[layer], self.layers[layer - 1])
                * np.sqrt(2 / self.layers[layer - 1])
            )
            params['b' + str(layer)] = np.zeros((self.layers[layer], 1))
        return params

    def _init_velo(self):
        """
        Initializes velocity for momentum optimization.
        Returns:
            Dictionary of velocities.
        """
        velo = {}
        L = len(self.layers)
        for layer in range(1, L):
            velo['dW' + str(layer)] = np.zeros_like(self.params['W' + str(layer)])
            velo['db' + str(layer)] = np.zeros_like(self.params['b' + str(layer)])
        print(velo)
        return velo

    '''
        ACTIVATION FUNCTIONS
    - Define ReLU and its derivative
    - Define linear activation for output layer
    '''
    def _relu(self, Z):
        """
        Implements the ReLU activation function.
        """
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        """
        Computes the gradient of ReLU.
        """
        return Z > 0

    def _linear(self, Z):
        """
        Linear activation function for regression outputs.
        """
        return Z

    '''
        FORWARD PROPAGATION
    - Compute activations from input to output
    - Store intermediate results for backpropagation
    '''
    def forward_feed(self, X):
        """
        Performs forward propagation through the network.
        Args:
            X: Input data of shape (m, n).
        Returns:
            Final activation (output of the network).
        """
        A = X.T
        self.cache['A0'] = A
        L = len(self.layers) - 1

        # Hidden layers
        for layer in range(1, L):
            W = self.params['W' + str(layer)]
            b = self.params['b' + str(layer)]
            Z = np.dot(W, A) + b
            A = self.activation_func(Z)
            self.cache['A' + str(layer)] = A
            self.cache['Z' + str(layer)] = Z

        # Output layer
        W = self.params['W' + str(L)]
        b = self.params['b' + str(L)]
        Z = np.dot(W, A) + b
        A = self._linear(Z)
        self.cache['A' + str(L)] = A
        self.cache['Z' + str(L)] = Z
        return A

    def cost(self, y1, y2):
        """
        Computes the Mean Squared Error (MSE) cost function.
        Args:
            y1: Predicted values.
            y2: True values.
        Returns:
            Cost value.
        """
        m = y2.shape[1]
        C = np.sum((y1 - y2) ** 2) / (2 * m)
        return C

    '''
        BACK PROPAGATION
    - Compute gradients for weights and biases
    - Store gradients for updating parameters
    '''
    def backward_feed(self, Y):
        """
        Performs backward propagation to compute gradients.
        Args:
            Y: True values (ground truth).
        """
        L = len(self.layers) - 1
        m = Y.shape[1]

        # Gradients for output layer
        dZ = self.cache['A' + str(L)] - Y
        self.gradient['dW' + str(L)] = np.dot(dZ, self.cache['A' + str(L - 1)].T) / m
        self.gradient['db' + str(L)] = np.sum(dZ, axis=1, keepdims=True) / m

        # Gradients for hidden layers
        for layer in reversed(range(1, L)):
            dZ = np.dot(self.params['W' + str(layer + 1)].T, dZ) * self.activation_derivative(self.cache['Z' + str(layer)])
            self.gradient['dW' + str(layer)] = np.dot(dZ, self.cache['A' + str(layer - 1)].T) / m
            self.gradient['db' + str(layer)] = np.sum(dZ, axis=1, keepdims=True) / m

    def update_network(self):
        """
        Updates the network parameters using gradient descent or momentum.
        """
        L = len(self.layers) - 1
        for layer in range(1, L + 1):
            if self.optimizer == 'momentum':
                self.velo['dW' + str(layer)] = self.beta * self.velo['dW' + str(layer)] + (1 - self.beta) * self.gradient['dW' + str(layer)]
                self.velo['db' + str(layer)] = self.beta * self.velo['db' + str(layer)] + (1 - self.beta) * self.gradient['db' + str(layer)]
                self.params['W' + str(layer)] -= self.learning * self.velo['dW' + str(layer)]
                self.params['b' + str(layer)] -= self.learning * self.velo['db' + str(layer)]
            else:
                self.params['W' + str(layer)] -= self.learning * self.gradient['dW' + str(layer)]
                self.params['b' + str(layer)] -= self.learning * self.gradient['db' + str(layer)]

    '''
        TRAINING AND OPTIMIZATION
    - Train the network over epochs
    - Print cost at intervals
    '''
    def train(self, X, Y, epochs=1000, batchsize=64):
        """
        Trains the neural network.
        Args:
            X: Training input data.
            Y: Training labels.
            epochs: Number of training iterations.
            batchsize: Size of mini-batches.
        Returns:
            List of cost values for each epoch.
        """
        m = X.shape[0]
        batches = m // batchsize
        cost_vals = []

        for epoch in range(epochs):
            perm = np.random.permutation(m)
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]

            for batch in range(batches):
                start = batch * batchsize
                end = min(start + batchsize, m)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                Y_pred = self.forward_feed(X_batch)
                cost = self.cost(Y_pred, Y_batch.T)
                self.backward_feed(Y_batch.T)
                self.update_network()

            cost_vals.append(cost)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Cost: {cost:.4f}")

        return cost_vals

    def prediction(self, X_pred):
        """
        Makes predictions using the trained network.
        Args:
            X_pred: Input data for prediction.
        Returns:
            Predicted output values.
        """
        Y_pred = self.forward_feed(X_pred)
        return Y_pred.T