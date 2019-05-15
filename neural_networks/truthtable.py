import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x)
    b1 = np.zeroes(n_h, 1)
    w2 = np.random.randn(n_y, n_h)
    b2 = np.zeros(n_y, 1)

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }

def forward_prop(X, parameters):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]

    Z1 = np.dot(w1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "A1" : A1,
        "A2" : A2
    }