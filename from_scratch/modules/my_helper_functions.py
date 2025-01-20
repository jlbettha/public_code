import numpy as np
import math, sys
import matplotlib.pyplot as plt
from sklearn.metrics import *
from numpy.typing import NDArray


###################################################
### Function Definitions ##########################
###################################################
def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    classes = np.unique(y_true)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
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


### filter a prediction stream to stabilize/improve accuracy (penalty is delay), input width=1 for no filter
def majority_filter(seq, width):
    offset = width // 2
    seq = [0] * offset + seq
    result = []
    for i in range(len(seq) - offset):
        a = seq[i : i + width]
        result.append(max(set(a), key=a.count))
    return np.squeeze(result)


### stable softmax applied to vector x
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


### sigmoid applied to vector x
def sigmoid(x):
    return [1 / (1 + math.exp(-n)) for n in x]


def anybase2decimal(number, other_base):
    return sum(
        [(int(v) * other_base**i) for i, v in enumerate(list(str(number))[::-1])]
    )


def decimal2ternary(number):
    arr = [0, 0, 0]
    i = 2
    while number:
        number, arr[i] = divmod(number, 3)
        i = i - 1
    return arr


def smooth(x, window_len):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y
