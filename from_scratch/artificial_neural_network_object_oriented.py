"""
Betthauser, J. 2023 - personal manual object oriented implementation for building
layer-wise prediction models.
"""

import time

import numpy as np
from modules.my_loss_functions import mean_squared_error, mse_derivative
from sklearn.datasets import load_digits

# from sklearn.metrics import *
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class GenericLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def feedforward(self, input):  # noqa: A002
        pass

    def backprop(self, output_gradient, learning_rate):
        pass


class FullyConnectedLayer(GenericLayer):
    rng = np.random.default_rng()  # Use a class-level generator

    def __init__(self, input_dim, output_dim):
        self.weights = self.rng.uniform(-1, 1, size=(output_dim, input_dim))
        self.biases = self.rng.uniform(-1, 1, size=(output_dim, 1))

    def feedforward(self, inputx):
        self.input = inputx
        return np.dot(self.weights, self.input) + self.biases

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

    def feedforward(self, inputx):
        self.input = inputx
        return self.activation_func(self.input)

    def backprop(self, gradient_outputs, learning_rate):  # noqa: ARG002
        return np.multiply(gradient_outputs, self.derivative_activation_func(self.input))


class Tanh(ActivationLayer):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def derivative_tanh(x):
            return 1 - np.tanh(x) ** 2

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
        def relu(x):
            return np.maximum(x, 0)

        def derivative_relu(x):
            return (x > 0).astype(float)

        super().__init__(relu, derivative_relu)


class Softmax(GenericLayer):
    def forward(self, inputx):
        exp_all = np.exp(inputx)
        self.output = exp_all / np.sum(exp_all)
        return self.output

    def backward(self, output_gradient, learning_rate):  # noqa: ARG002
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

    x, y = load_digits(return_X_y=True)
    num_classes = len(np.unique(y))
    num_features = x.shape[1]

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    x = np.reshape(x, (x.shape[0], num_features, 1))
    y = np.expand_dims(encode_one_hot(y, num_classes), 2)

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
        for xx, yy in zip(x, y, strict=False):
            ## feed forward
            output = xx
            for layer in mlp_neural_network:
                output = layer.feedforward(output)

            ## accumulate error
            error = error + np.sum(mean_squared_error(yy, output))

            ## back propagation for gradient descent
            gradient = mse_derivative(yy, output)
            for layer in reversed(mlp_neural_network):
                gradient = layer.backprop(gradient, learning_rate)

        error = error / y.shape[0]
        if ep % 10 == 0:
            print(f"Epoch {ep} of {num_epochs}: {error=:.5f}", flush=True)


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
