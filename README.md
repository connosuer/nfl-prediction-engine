# NFL-AI

### High Level Workflow
![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcLU08nA7xFEh2n-G2vPRhgB7C-RNrSskxoLcIsmzaxXvvpyi97RkeW8JKXiHvtuUsrKfw0YGRd32pfzIEnstnyN1ctUw3dLtdvALLhKkQZOGvNVYQXQJa4HXm-2MkWmFJgWL4fSIbptZzcJpLHfzo1CLmU?key=AxnRBYyBp3KqZx20flgOOynW)

### Data
**Historic NFL Data** *(https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)*

- Kaggle dataset comprised of the majority of NFL football games dating back to 1966 with 25 unique features for each game.
- Games taking place prior to the 2010-2011 season were taken out.
- Irrelevant and/or noisy potential indicators were removed.

### Line Generation Neural Network
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


**Cost Graphs Over Epochs**

<img src="Media/cost_10.png" alt="Cost Graph - 10 Epochs" width="45%"> <img src="Media/cost_100.png" alt="Cost Graph - 100 Epochs" width="45%">