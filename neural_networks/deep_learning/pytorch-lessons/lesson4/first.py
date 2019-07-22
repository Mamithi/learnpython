import torch 
import numpy as np

def activation(x):
    return 1 / (1+torch.exp(-x))


### Generate some data
torch.manual_seed(7)

# fEatures are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1] # Number of input units must match number of input features

n_hidden = 2 # Number of hidden inputs

n_output = 1 # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)

# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)


# Bias terms for hidden and output layers
B1 = torch.randn(1, n_hidden)
B2 = torch.randn(1, n_output)


h1 = torch.mm(features, W1) + B1
h = activation(h1)

output = torch.mm(h, W2) + B2

y = activation(output)

print(output)
print(y)