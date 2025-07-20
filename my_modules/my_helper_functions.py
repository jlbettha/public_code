from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray[int],
    y_pred: np.ndarray[int],
    classes=None,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
):
    """
    Prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        title = "Normalized confusion matrix" if normalize else "Confusion matrix, without normalization"

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if not classes:
        classes = [str(i) for i in np.unique(y_true)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # text of cm values in each square
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax


###
def majority_filter(seq: np.ndarray[int], width: int) -> int:
    """
    Majority vote filter for temporal classification streams/sequences
        to stabilize/improve accuracy (penalty is delay), input width=1 for no filter

    Args:
        seq (NDArray[np.float64]): raw prediction sequence
        width (int): filter length/width

    Returns:
        int: majority vote

    """
    offset = width // 2
    seq = [0] * offset + seq
    result = []
    for i in range(len(seq) - offset):
        a = seq[i : i + width]
        result.append(max(set(a), key=a.count))
    return np.squeeze(result)


def anybase2decimal(number: int, other_base: int) -> int:
    """
    Convert any base number to decimal

    Args:
        number (int): _description_
        other_base (int): _description_

    Returns:
        int: _description_

    """
    return np.sum([(int(v) * other_base**i) for i, v in enumerate(list(str(number))[::-1])])


@njit
def decimal2ternary(number: int) -> int:
    """
    Convert decimal to ternary

    Args:
        number (int): _description_

    Returns:
        int: _description_

    """
    arr = np.zeros(2)
    i = 2
    while number:
        number, arr[i] = np.divmod(number, 3)
        i = i - 1
    return arr


def smooth(x: np.ndarray[float], window_len: int) -> np.ndarray[float]:
    """
    Generic decent smoothing procedure (appropriate smoothing will always be data dependent)

    Args:
        x (np.ndarray[float]): raw signal
        window_len (int): smoothing window length

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray[float]: smoothed signal

    """
    if x.ndim != 1:
        msg = "smooth only accepts 1 dimension arrays."
        raise ValueError(msg)
    if x.size < window_len:
        msg = "Input vector needs to be bigger than window size."
        raise ValueError(msg)
    if window_len < 3:  # noqa: PLR2004
        return x
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    w = np.hanning(window_len)
    return np.convolve(w / w.sum(), s, mode="valid")


def flatten_lists(list_of_lists: list[Any]) -> list[Any]:
    """
    Betthauser 2025 - recursively flatten/unravel any depth of lists within lists

    Args:
        list_of_lists (Any list[list[list[...]]]): any depth lists within lists

    Returns:
        list[Any]: flattened list

    """
    if isinstance(list_of_lists, int):
        return list_of_lists
    try:
        flat_list = list(np.ravel(list_of_lists))
    except RuntimeError:
        flat_list = []
        for i in list_of_lists:
            flat_list.extend(flatten_lists(i))
    return flat_list


def main() -> None:
    print("my_helper_functions.py is a module")


if __name__ == "__main__":
    main()
