# # Basic artificial neural network from scratch (functional version)
# Currently, Digits dataset.

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import NDArray


def relu(wx_b):
    return np.maximum(wx_b, 0)


def d_relu_dz(z):
    return z > 0


def softmax(z):
    num_s = z.shape[1]
    A = np.array([np.exp(z[:, n]) / np.sum(np.exp(z[:, n])) for n in range(num_s)])
    return A.T


def encode_one_hot(y, num_classes):
    y_one_hot = np.zeros((y.shape[0], num_classes))
    for n in range(y.shape[0]):
        y_one_hot[n, y[n]] = 1
    return np.vstack(y_one_hot)


def forward(W1, b1, W2, b2, X):
    z1 = W1 @ X.T + b1
    A1 = relu(z1)
    z2 = W2 @ A1 + b2
    A2 = softmax(z2)
    return z1, A1, z2, A2


def back_prop(z1, A1, A2, W2, X, y):
    num_classes = A2.shape[0]
    N = y.shape[0]
    y_one_hot = encode_one_hot(y, num_classes)
    dZ2 = A2 - y_one_hot.T
    dW2 = (dZ2 @ A1.T) / N
    db2 = np.expand_dims(np.sum(dZ2, axis=1) / N, axis=1)
    dZ1 = (W2.T @ dZ2) * d_relu_dz(z1)
    dW1 = (dZ1 @ X) / N
    db1 = np.expand_dims(np.sum(dZ1, axis=1) / N, axis=1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, eta):
    W1 = W1 - eta * dW1
    b1 = b1 - eta * db1
    W2 = W2 - eta * dW2
    b2 = b2 - eta * db2

    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def accuracy(predictions, y):
    return np.sum(predictions == y) / y.shape[0]


def gradient_descent(X, y, eta, iterations, W1, b1, W2, b2):
    accs = []
    for i in range(iterations):
        z1, A1, z2, A2 = forward(W1, b1, W2, b2, X)

        dW1, db1, dW2, db2 = back_prop(z1, A1, A2, W2, X, y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, eta)
        if i % 100 == 0:
            predictions = get_predictions(A2)
            acc = accuracy(predictions, y)
            accs.append([i, acc])
            print(f"Iteration {i}: accuracy = {acc:.4f}", flush=True)

    accs = np.vstack(accs)
    return W1, b1, W2, b2, accs


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def main() -> None:
    """artificial neural network"""

    ## init vars
    iterations = 1500
    eta = 0.05

    # X, y = load_iris(return_X_y=True)
    X, y = load_digits(return_X_y=True)
    print(X.shape, y.shape)

    num_classes = len(np.unique(y))
    num_features = X.shape[1]
    num_nodes_1 = 6

    W1 = np.random.uniform(size=(num_nodes_1, num_features))
    b1 = np.random.uniform(size=(num_nodes_1, 1))
    W2 = np.random.uniform(size=(num_classes, num_nodes_1))
    b2 = np.random.uniform(size=(num_classes, 1))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    W1, b1, W2, b2, accs = gradient_descent(
        X_train, y_train, eta, iterations, W1, b1, W2, b2
    )

    print(f"Training accuracy = {100*accs[-1, 1]:.3f}%")
    # plt.figure()
    # plt.plot(accs[:, 0], accs[:, 1])
    # plt.xlabel("iterations")
    # plt.ylabel("accuracy")
    # plt.show()

    y_pred = make_predictions(X_test, W1, b1, W2, b2)
    acc_test = accuracy(y_pred, y_test)
    print(f"Test accuracy = {100*acc_test:.3f}%")

    index = np.random.randint(400)

    current_image = X_test[index, :]
    prediction = make_predictions(X_train[index, :], W1, b1, W2, b2)
    label = y_test[index]

    current_image = current_image.reshape((8, 8)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.title(f"truth: {label}, predict: {prediction}")
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
