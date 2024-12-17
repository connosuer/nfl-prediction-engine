# NFL-AI

### Usage
After cloning, install the necessary packages using:
~~~
pip install -r requirements.txt
~~~

To test the neural network and feature engineering individually, run the 'nn_test.py'
and 'test_feature_engineering.py' files. 

To test the betting system as a whole, run the 'test_betting_system.py' file.

### Introduction
The purpose of this project was to determine whether a machine learning approach to sports-betting
would outperform popular alternative strategies.  Our project utilizes the predictive capabilities of 
neural networks to assess point spreads and find potential advantageous betting situations for a 
particular NFL game.


### Data
**Historic NFL Data** *(https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)*

- Kaggle dataset comprised of the majority of NFL football games dating back to 1966 with 25 unique features for each game.
- Games taking place prior to the 2010-2011 season were taken out.
- Irrelevant and/or noisy potential indicators were removed.

**Feature Engineering**

### Neural Network
**Neural Network Input Initialization**
- layers - Specifies the number of neurons in each layer and the configuration of neurons.
- activation - Activation function for hidden layers.
- output_activation - Activation function for output layer.
- learning - Network learning rate.
- beta - Momentum coefficient

**Neural Network Component Initialization**
- params - Initializes weights and biases using He Initialization.
- velo - Initializes velocity calculation for momentum optimization.

**Caching and Storage**
- params - Stores weights and biases.
- cache - Stores intermediate calculations.
- gradient - Stores computed gradients.

**Forward Propagation**
- Inputs 'X' are transposed onto a matrix as:
~~~
A = X.T
self.cache['A0'] = A
~~~
- Input and updated weight matrix's are multiplied and bias is added:
~~~
Z = np.dot(W, A) + b
~~~
- Output is ran through the chosen activation function:
~~~
A = self.activation_func(Z)
~~~
- Cache the new input matrix 'A' and dot product + bias result 'Z':
~~~
self.cache['A' + str(layer)] = A
self.cache['Z' + str(layer)] = Z
~~~
- Cost function is calculated using the Mean Squared Error (MSE) fromula.

**Backpropagation**

- Computes cost gradients with respect to weights and biases

*Output Layer*
- Cost function is computed with respect to Z (pre-activation):
~~~
dZ = self.cache['A' + str(L)] - Y
~~~
- Gradient for weights and biases:
~~~
self.gradient['dW' + str(L)] = np.dot(dZ, self.cache['A' + str(L - 1)].T) / m
self.gradient['db' + str(L)] = np.sum(dZ, axis=1, keepdims=True) / m
~~~
- dW = derivative of cost function with respect to weights
- db = derivative of cost function with respect to bias
- Gradients are averaged over batch size 'm'

*Hidden Layers*
- Moves gradients back through the hidden layers.
- Multiplies the gradient from the following layer with weight matrix.
- Multiplies result with the derivative of the activation function.
~~~
dZ = np.dot(self.params['W' + str(layer + 1)].T, dZ) * self.activation_derivative(self.cache['Z' + str(layer)])
~~~
- Computation of the gradients is performed as outlined in the output layer.
- 'dW' and 'db' are cached.

**Training & Optimization**
- Batch size and cost value storage are initialized.
- Data is shuffled at the start of each epoch (training round).
- For each batch, forward propagation is performed and cost calculated:
~~~
Y_pred = self.forward_feed(X_batch)
cost = self.cost(Y_pred, Y_batch.T)
~~~
- Back propagation is performed and network is updated:
~~~
self.backward_feed(Y_batch.T)
self.update_network()
~~~
- Cost is logged and the network makes predictions on test data:
~~~
Y_pred = self.forward_feed(X_pred)
~~~
- Transpose to original format:
~~~
return Y_pred.T
~~~

### Betting System

### Results

**Neural Network**

The neural network we engineered succeeded in accurately predicting NFL scores after
sufficient training.

Cost *(First Training Round)*: 0.52 \
Cost *(Last Training Round)*: 0.16 \
RMSE Score: 2.52

Randomly Selected Test Results: \
Predicted: -4.86, Actual: -4.50 \
Predicted: -2.04, Actual: -3.00 \
Predicted: -4.63, Actual: -2.00 \
Predicted: -7.35, Actual: -7.00

Cost Graphs Over Epochs:

<img src="Media/cost_10.png" alt="Cost Graph - 10 Epochs" width="45%"> <img src="Media/cost_100.png" alt="Cost Graph - 100 Epochs" width="45%">

*From nn_test.py*