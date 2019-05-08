import numpy as np

ts_inputs = np.array([[0,0,1,0],
                      [1,1,1,0],
                      [1,0,1,1],
                      [0,1,1,1],
                      [0,1,0,1],
                      [1,1,1,1],
                      [0,0,0,0]])

ts_outputs = np.array([[0,1,1,0,0,1,0]]).T

class Perceptron():
    def __init(self):
        self.syn_weights = np.random.rand(4, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return np.exp(-x)/((1 + np.exp(-x))**2)

    def train(self, inputs, real_outputs, its, lr):
        delta_weights = np.zeros((4, 7))
        for iteration in (range(its)):
            # Foward pass
            z = np.dot(inputs, self.syn_weights)
            activation = self.sigmoid(z)

            # backward Pass
            for i in range(7):
                cost = (activation[i] - real_outputs[i]**2)
                cost_prime = 2*(activation[i] - real_outputs[i])
                for n in range(4):
                    delta_weights[n][i] = cost_prime * inputs[i][n] * self.sigmoid_deriv(z[i])

            delt_avg = np.array([np.average(delta_weights, axis=1)]).T
            self.syn_weights = self.syn_weights - delt_avg*lr

    def results(self, inputs):
        return self.sigmoid(np.dot(inputs, self.syn_weights))
