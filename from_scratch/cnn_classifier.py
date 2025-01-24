import numpy as np
from numpy.typing import NDArray


def my_batch_norm(Z: NDArray[np.float64]) -> NDArray[np.float64]:
    """batch normalization

    Args:
        Z (NDArray[np.float64]): data for batch normalization

    Returns:
        NDArray[np.float64]: batch normalized data
    """

    mu = np.mean(Z, axis=(0, 1, 2), keepdims=True)
    sigma2 = np.var(Z, axis=(0, 1, 2), keepdims=True)
    return (Z - mu) / np.sqrt(sigma2 + 1e-8)


def my_dropout(Z: NDArray[np.float64], rate: float = 0.25) -> NDArray[np.float64]:
    """dropout

    Args:
        Z (NDArray[np.float64]): data
        rate (float, optional): dropout rate. Defaults to 0.25.

    Returns:
        NDArray[np.float64]: data after dropout
    """
    D = np.random.rand(*Z.shape) < (1 - rate)
    return Z * D / (1 - rate)


def my_relu(Z: NDArray[np.float64]) -> NDArray[np.float64]:
    """rectified linear unit

    Args:
        Z (NDArray[np.float64]): input

    Returns:
        NDArray[np.float64]: relu activation
    """
    return np.maximum(0, Z)


# Convolution block with batch normalization and dropout.
def my_conv_block(
    X: NDArray[np.float64],
    W: NDArray[np.float64],
    b: NDArray[np.float64],
    stride: int = 1,
    padding: int = 0,
) -> NDArray[np.float64]:
    """conv_block with batch normalization and dropout for building cnn networks

    Args:
        X (NDArray[np.float64]): input data (batchsize x height_prev x width_prev x n_C_prev)
        W (NDArray[np.float64]): weights (filt_size x filt_size x n_C_prev x n_C)
        b (NDArray[np.float64]): biases (1 x 1 x 1 x n_C)
        stride (int, optional): stride/step-size. Defaults to 1. Stride >= 2 will downsample input dims
        padding (int, optional): zero-padding. Defaults to 0.

    Returns:
        NDArray[np.float64]: output after convolutions
    """
    (m, n_H_prev, n_W_prev, _) = X.shape
    (filt_size, filt_size, _, n_C) = W.shape

    n_H = int((n_H_prev - filt_size + 2 * padding) / stride) + 1
    n_W = int((n_W_prev - filt_size + 2 * padding) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    if padding > 0:
        X = np.pad(
            X,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode="constant",
            constant_values=(0, 0),
        )

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + filt_size
                    horiz_start = w * stride
                    horiz_end = horiz_start + filt_size

                    X_slice = X[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = np.sum(X_slice * W[:, :, :, c]) + b[:, :, :, c]

    return Z


def cnn_classifier():
    # TODO
    return NotImplementedError
