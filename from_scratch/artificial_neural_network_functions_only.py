# # Basic artificial neural network from scratch (functional version)
# Currently, Digits dataset.

import time

import numpy as np
from matplotlib import pyplot as plt
from modules.my_activations import d_relu_dz, relu, softmax_jit
from numba import njit
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@njit
def encode_one_hot(y):
    num_classes = len(np.unique(y))
    y_one_hot = np.zeros((y.shape[0], num_classes))
    for n in range(y.shape[0]):
        y_one_hot[n, y[n]] = 1
    return y_one_hot


@njit
def forward(w1, b1, w2, b2, x):
    z1 = w1 @ x.T + b1
    a1 = relu(z1)
    z2 = w2 @ a1 + b2
    a2 = softmax_jit(z2)
    return z1, a1, z2, a2


@njit
def back_prop(z1, a1, a2, w2, x, y):
    n = y.shape[0]
    y_one_hot = encode_one_hot(y)
    dz2 = a2 - y_one_hot.T
    dw2 = (dz2 @ a1.T) / n
    db2 = np.expand_dims(np.sum(dz2, axis=1) / n, axis=1)
    dz1 = (w2.T @ dz2) * d_relu_dz(z1)
    dw1 = (dz1 @ x) / n
    db1 = np.expand_dims(np.sum(dz1, axis=1) / n, axis=1)
    return dw1, db1, dw2, db2


@njit
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, eta):
    w1 = w1 - eta * dw1
    b1 = b1 - eta * db1
    w2 = w2 - eta * dw2
    b2 = b2 - eta * db2
    return w1, b1, w2, b2


@njit
def get_predictions(a2):
    return np.argmax(a2, 0)


@njit
def accuracy(predictions, y):
    return np.sum(predictions == y) / y.shape[0]


# @njit
def gradient_descent(x, y, eta, iterations, w1, b1, w2, b2):
    accs = []
    for i in range(iterations):
        z1, a1, z2, a2 = forward(w1, b1, w2, b2, x)

        dw1, db1, dw2, db2 = back_prop(z1, a1, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, eta)

        if i % 200 == 0:
            predictions = get_predictions(a2)
            acc = accuracy(predictions, y)
            accs.append([i, acc])
            print(f"Iteration {i}: accuracy = {acc:.4f}", flush=True)

    return w1, b1, w2, b2


def make_predictions(x, w1, b1, w2, b2):
    _, _, _, a2 = forward(w1, b1, w2, b2, x)
    return get_predictions(a2)


def main() -> None:
    """Artificial neural network"""
    # Init vars
    iterations = 3000
    eta = 0.02

    x, y = load_digits(return_X_y=True)
    print(x.shape, y.shape)

    num_classes = len(np.unique(y))
    num_features = x.shape[1]
    num_nodes_1 = 6

    rng = np.random.default_rng()
    w1 = rng.uniform(low=0, high=1, size=(num_nodes_1, num_features))
    b1 = rng.uniform(low=0, high=1, size=(num_nodes_1, 1))
    w2 = rng.uniform(low=0, high=1, size=(num_classes, num_nodes_1))
    b2 = rng.uniform(low=0, high=1, size=(num_classes, 1))

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    w1, b1, w2, b2 = gradient_descent(x_train, y_train, eta, iterations, w1, b1, w2, b2)

    y_pred = make_predictions(x_test, w1, b1, w2, b2)
    acc_test = accuracy(y_pred, y_test)

    print(f"Test accuracy = {100 * acc_test:.3f}%")

    dim = 5
    idx = rng.permutation(len(y_test))

    plt.figure(figsize=(7, 7))
    for i in range(dim**2):
        img = np.squeeze(x_test[idx[i], ...]).reshape((8, 8)) * 255
        lbl = y_test[idx[i]]
        prd = y_pred[idx[i]]

        plt.subplot(dim, dim, i + 1)
        plt.imshow(img, cmap="gray", interpolation="none")
        plt.axis("off")
        clr = "red"
        if lbl == prd:
            clr = "green"
        plt.title(f"True: {lbl}, Pred: {prd}", color=clr, fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
