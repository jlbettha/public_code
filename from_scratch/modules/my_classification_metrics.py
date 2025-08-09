import numpy as np
from numba import njit


@njit
def _accuracy(tn_fp_fn_tp: np.ndarray) -> float:
    """Calculate accuracy from confusion matrix components."""
    return (tn_fp_fn_tp[3] + tn_fp_fn_tp[0]) / np.sum(tn_fp_fn_tp)


@njit
def _sensitivity_tpr(tn_fp_fn_tp: np.ndarray) -> float:
    return tn_fp_fn_tp[3] / (tn_fp_fn_tp[3] + tn_fp_fn_tp[2])


@njit
def _fnr(tn_fp_fn_tp: np.ndarray) -> float:
    return tn_fp_fn_tp[2] / (tn_fp_fn_tp[3] + tn_fp_fn_tp[2])


@njit
def _specificity_tnr(tn_fp_fn_tp: np.ndarray) -> float:
    return tn_fp_fn_tp[0] / (tn_fp_fn_tp[0] + tn_fp_fn_tp[1])


@njit
def _fpr(tn_fp_fn_tp: np.ndarray) -> float:
    return tn_fp_fn_tp[1] / (tn_fp_fn_tp[0] + tn_fp_fn_tp[1])


@njit
def _ppv(tn_fp_fn_tp: np.ndarray) -> float:
    return tn_fp_fn_tp[3] / (tn_fp_fn_tp[3] + tn_fp_fn_tp[1])


@njit
def _npv(tn_fp_fn_tp: np.ndarray) -> float:
    return tn_fp_fn_tp[0] / (tn_fp_fn_tp[0] + tn_fp_fn_tp[2])


@njit
def _balanced_accuracy(tn_fp_fn_tp: np.ndarray) -> float:
    tprr = _sensitivity_tpr(tn_fp_fn_tp)
    tnrr = _specificity_tnr(tn_fp_fn_tp)
    return (tprr + tnrr) / 2


@njit
def _f1(tn_fp_fn_tp: np.ndarray) -> float:
    p = _ppv(tn_fp_fn_tp)
    r = _sensitivity_tpr(tn_fp_fn_tp)
    if p + r == 0:
        return 0.0
    return (2 * p * r) / (p + r)


@njit
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix.

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        np.ndarray: _description_

    """
    y_true.ravel()
    y_pred.ravel()
    classes = np.unique(y_true.astype(np.int64))
    n_label = len(classes)

    conf_mat = np.zeros((n_label, n_label)).astype(np.int64)

    for i in range(len(y_true)):
        conf_mat[y_true[i]][y_pred[i]] += 1

    return conf_mat


def classification_metrics(conf_mat: np.ndarray) -> dict:
    """
    Get classification metrics from confusion matrix.

    Args:
        conf_mat (np.ndarray): confusion matrix

    Returns:
        dict: classification metrics

    """
    tn_fp_fn_tp: np.ndarray = conf_mat.ravel()

    return {
        "tn": tn_fp_fn_tp[0],
        "fp": tn_fp_fn_tp[1],
        "fn": tn_fp_fn_tp[2],
        "tp": tn_fp_fn_tp[3],
        "total": np.sum(tn_fp_fn_tp),
        "accuracy": _accuracy(tn_fp_fn_tp),
        "tpr": _sensitivity_tpr(tn_fp_fn_tp),
        "fnr": _fnr(tn_fp_fn_tp),
        "tnr": _specificity_tnr(tn_fp_fn_tp),
        "fpr": _fpr(tn_fp_fn_tp),
        "ppv": _ppv(tn_fp_fn_tp),
        "npv": _npv(tn_fp_fn_tp),
        "balanced_accuracy": _balanced_accuracy(tn_fp_fn_tp),
        "f1": _f1(tn_fp_fn_tp),
    }


def main():
    # Example usage
    rng = np.random.default_rng(2)
    y_true = rng.choice([0, 1], size=64**3, p=[0.7, 0.3])
    y_pred = rng.choice([0, 1], size=64**3, p=[0.8, 0.2])

    conf_mat = confusion_matrix(y_true, y_pred)
    metrics = classification_metrics(conf_mat)

    print("Confusion Matrix:\n", conf_mat)
    print("Classification Metrics:")
    for k, v in metrics.items():
        print(f"   {k:<20}------ {v:<10.4f}")


if __name__ == "__main__":
    import time

    t0 = time.time()
    main()
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Run 1: {t1 - t0:.4f} seconds")
    print(f"Run 2: {t2 - t1:.4f} seconds")
    print(f"JIT speedup: {(t1 - t0) / (t2 - t1):.4f}x")
