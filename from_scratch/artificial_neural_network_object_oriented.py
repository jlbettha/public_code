"""Betthauser, J. 2023 - personal manual object oriented implementation for building
layer-wise prediction models.
"""

import numpy as np
import time

# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.datasets import load_iris, load_digits
from numba import njit
from modules.my_loss_functions import mean_squared_error, mse_derivative


class GenericLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def feedforward(self, input):
        pass

    def backprop(self, output_gradient, learning_rate):
        pass


class FullyConnectedLayer(GenericLayer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.uniform(-1, 1, size=(output_dim, input_dim))
        self.biases = np.random.uniform(-1, 1, size=(output_dim, 1))

    def feedforward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    @njit
    def backprop(self, gradient_outputs, learning_rate):
        gradient_weights = np.dot(gradient_outputs, self.input.T)
        gradient_inputs = np.dot(self.weights.T, gradient_outputs)
        self.weights = self.weights - learning_rate * gradient_weights
        self.biases = self.biases - learning_rate * gradient_outputs
        return gradient_inputs


class ActivationLayer(GenericLayer):
    def __init__(self, activation_func, derivative_activation_func):
        self.activation_func = activation_func
        self.derivative_activation_func = derivative_activation_func

    def feedforward(self, input):
        self.input = input
        return self.activation_func(self.input)

    def backprop(self, gradient_outputs, learning_rate):
        return np.multiply(
            gradient_outputs, self.derivative_activation_func(self.input)
        )


class Tanh(ActivationLayer):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        derivative_tanh = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, derivative_tanh)


class Sigmoid(ActivationLayer):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def derivative_sigmoid(x):
            sigx = sigmoid(x)
            return (1 - sigx) * sigx

        super().__init__(sigmoid, derivative_sigmoid)


class Relu(ActivationLayer):
    def __init__(self):
        relu = lambda x: np.maximum(x, 0)
        derivative_relu = lambda x: (x > 0).astype(float)
        super().__init__(relu, derivative_relu)


class Softmax(GenericLayer):
    def forward(self, input):
        exp_all = np.exp(input)
        self.output = exp_all / np.sum(exp_all)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


def encode_one_hot(y, num_classes):
    y_one_hot = np.zeros((y.shape[0], num_classes))
    for n in range(y.shape[0]):
        y_one_hot[n, y[n]] = 1
    return np.vstack(y_one_hot)


def main() -> None:
    # X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    # Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    X, Y = load_digits(return_X_y=True)
    num_classes = len(np.unique(Y))
    num_features = X.shape[1]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X = np.reshape(X, (X.shape[0], num_features, 1))
    Y = np.expand_dims(encode_one_hot(Y, num_classes), 2)

    mlp_neural_network = [
        FullyConnectedLayer(num_features, 6),
        Tanh(),
        FullyConnectedLayer(6, num_classes),
        Tanh(),
    ]

    num_epochs = 200
    learning_rate = 0.1

    for ep in range(num_epochs):
        error = 0
        for x, y in zip(X, Y):

            ## feed forward
            output = x
            for layer in mlp_neural_network:
                output = layer.feedforward(output)

            ## accumulate error
            error = error + np.sum(mean_squared_error(y, output))

            ## back propagation for gradient descent
            gradient = mse_derivative(y, output)
            for layer in reversed(mlp_neural_network):
                gradient = layer.backprop(gradient, learning_rate)

        error = error / y.shape[0]
        if ep % 10 == 0:
            print(f"Epoch {ep} of {num_epochs}: {error=:.5f}", flush=True)


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-tmain:.3f} seconds.")
