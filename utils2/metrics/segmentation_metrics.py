import numpy as np

# import matplotlib.pyplot as plt
from numba import njit
from skimage.measure import regionprops, regionprops_table, shannon_entropy  # noqa: F401

EPS = 1e-5
ETA = 1e-5
THRESHOLD = 0.5


@njit
def mask_volume(mask: np.ndarray) -> float:
    """
    Calculate the pixel area/voxel volume of a binary mask.

    Args:
        mask (np.ndarray): binary mask.

    Returns:
        float: area/volume of the mask.

    """
    return np.sum(mask)


@njit
def dice_similarity_coefficient(pred: np.ndarray, target: np.ndarray, epsilon: float = EPS, eta: float = ETA) -> float:
    """
    Calculate the General Dice Similarity Coefficient (DSC) between two binary masks. If epsilon and eta are zero,
    this is equivalent to ordinary DSC.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.
        epsilon (float): Numerator smoothing parameter.
        eta (float): Denominator smoothing parameter.

    Returns:
        float: Dice Similarity Coefficient.

    """
    intersection = np.sum(pred * target) + epsilon
    return 2.0 * (intersection + eta) / (np.sum(pred) + np.sum(target) + eta)


@njit
def generalized_jaccard_index(pred: np.ndarray, target: np.ndarray, epsilon: float = EPS, eta: float = ETA) -> float:
    """
    Calculate the Generalized Jaccard Index (GJI) between two binary masks. If epsilon and eta are zero,
    this is equivalent to ordinary Jaccard Index.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.
        epsilon (float): Numerator smoothing parameter.
        eta (float): Denominator smoothing parameter.

    Returns:
        float: Generalized Jaccard Index.

    """
    intersection = np.sum(pred * target) + epsilon
    union = np.sum(pred) + np.sum(target) - intersection + eta
    return intersection / union


@njit
def criterion1(pred: np.ndarray, target: np.ndarray) -> bool:
    """
    Get criterion 1 from https://arxiv.org/html/2311.09614v2#S3.SS5.SSS2.

    Args:
        pred (np.ndarray): _description_
        target (np.ndarray): _description_

    Returns:
        bool: _description_

    """
    return np.sum(pred * target) > 0


@njit
def criterion2(pred: np.ndarray, target: np.ndarray) -> bool:
    """
    Get criterion 2 from https://arxiv.org/html/2311.09614v2#S3.SS5.SSS2.

    Args:
        pred (np.ndarray): _description_
        target (np.ndarray): _description_

    Returns:
        bool: _description_

    """
    return generalized_jaccard_index(pred, target) > THRESHOLD


@njit
def get_label_min_pt(img: np.ndarray, label: np.ndarray) -> tuple[int]:
    """
    Get the index of the maximum value in the image where the label is non-zero.

    Args:
        img (np.ndarray): _description_
        label (np.ndarray): _description_

    Returns:
        tuple[int]: _description_

    """
    masked_img = img * label
    return np.argmin(masked_img, axis=None)


# @njit
def segment_center_of_mass(label_mask: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass of a label mask.

    Args:
        label_mask (np.ndarray): The label mask to calculate the center of mass from.

    Returns:
        tuple[int, int, int]: The coordinates of the center of mass.

    """
    # if not isinstance(label_mask, np.ndarray):
    #     raise TypeError("label_mask must be a numpy array")

    # if label_mask.ndim not in {2, 3}:
    #     raise ValueError("label_mask must be a 2D or 3D array")

    # if np.sum(label_mask) == 0:
    #     return (0.5 * np.array(label_mask.shape)).astype(int)

    return np.array(np.where(label_mask)).mean(axis=1).astype(np.uint8)


@njit
def criterion3(pred: np.ndarray, label_max_pt: tuple[int]) -> bool:
    """
    Get criterion 3 from https://arxiv.org/html/2311.09614v2#S3.SS5.SSS2.

    Args:
        pred (np.ndarray): _description_
        label_max_pt (tuple[int]): _description_

    Returns:
        bool: _description_

    """
    return pred[label_max_pt] != 0


# @njit
def _hist1d(
    vals: np.ndarray[float],
    bins: int,
    val_range: tuple[float],
    norm: bool = False,
) -> np.ndarray[np.int64]:
    """
    JIT compute 1D histogram of values (fastest current method).

    Args:
        vals (np.ndarray[float]): _description_
        bins (int): _description_
        val_range (tuple[float]): _description_
        norm (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray[np.int64]: _description_

    """
    hist = np.histogram(vals, bins, val_range)[0]
    if norm:
        hist = hist / np.sum(hist)
    return hist


# @njit
def haziness_abs_norm(label: np.ndarray, img: np.ndarray) -> float:
    """
    Calculate the normalized histogram for background (B) and foreground (F). The metric is defined as the absolute
    value of the B histogram minus  F histogram, divided by their sum (B_hist + F_hist).


    Args:
        label (np.ndarray): _description_
        img (np.ndarray): _description_

    Returns:
        float: _description_

    """
    background_vals = img[np.where(label < THRESHOLD)]
    foreground_vals = img[np.where(label >= THRESHOLD)]

    mean_f = np.mean(foreground_vals) if foreground_vals.size > 0 else 0.0
    mean_b = np.mean(background_vals)
    imax = np.max(img)
    imin = np.min(img)
    background_hist = _hist1d(vals=background_vals, bins=32, val_range=(imin, imax), norm=True)
    foreground_hist = _hist1d(vals=foreground_vals, bins=32, val_range=(imin, imax), norm=True)

    haze = np.sum(np.abs(background_hist - foreground_hist)) / np.sum(background_hist + foreground_hist)
    haze_jlb = (mean_f - mean_b) / (mean_f + mean_b + EPS)
    weber = (mean_f - mean_b) / mean_b
    # michel = imax - imin / (imax + imin + EPS)
    return haze, haze_jlb, weber  # , michel


def get_segmentation_metrics(true: np.ndarray, pred: np.ndarray, img: np.ndarray):
    """
    Compute segmentation metrics for the given true and predicted masks.

    Args:
        true (np.ndarray): Ground truth binary mask.
        pred (np.ndarray): Predicted binary mask.
        img (np.ndarray): Image data for additional metrics.

    Returns:
        dict: Dictionary containing segmentation metrics.

    """
    metrics = {}
    metrics["intensity_max"] = np.max(img) if img is not None else None
    metrics["haziness"], metrics["haziness2"], metrics["weber_contrast"] = haziness_abs_norm(true, img)
    metrics["vol_true"] = mask_volume(true)
    metrics["vol_pred"] = mask_volume(pred)
    metrics["dice"] = dice_similarity_coefficient(pred, true)
    metrics["jaccard"] = generalized_jaccard_index(pred, true)

    max_pt = tuple(segment_center_of_mass(true))
    # print(f"Max point: {max_pt}")
    metrics["criterion1"] = criterion1(pred, true)
    metrics["criterion2"] = criterion2(pred, true)
    metrics["criterion3"] = criterion3(pred, max_pt)
    # print(metrics["criterion3"])
    return metrics


def main():
    pass


if __name__ == "__main__":
    main()
