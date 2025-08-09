import numpy as np

# import matplotlib.pyplot as plt
from numba import njit

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
    return 2.0 * intersection / (np.sum(pred) + np.sum(target) + eta)


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
def criterion3(pred: np.ndarray, g_suv_max_pt: tuple[int]) -> bool:
    """
    Get criterion 3 from https://arxiv.org/html/2311.09614v2#S3.SS5.SSS2.

    Args:
        pred (np.ndarray): _description_
        g_suv_max_pt (tuple[int]): _description_

    Returns:
        bool: _description_

    """
    return pred[g_suv_max_pt] == 1


@njit
def false_negative_volume(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate the volume of false negatives in a binary segmentation.
    FNV from https://arxiv.org/html/2311.09614v2#S3.SS5.SSS2.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.

    Returns:
        float: Volume of false negatives.

    """
    vp = mask_volume(pred)
    vg = mask_volume(target)
    np.sum(vp - vg)
    return NotImplementedError("This function is not implemented yet.")


@njit
def false_positive_volume(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate the volume of false positives in a binary segmentation.
    FPV from https://arxiv.org/html/2311.09614v2#S3.SS5.SSS2.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.

    Returns:
        float: Volume of false positives.

    """
    np.sum(pred * (1 - target))
    return NotImplementedError("This function is not implemented yet.")
