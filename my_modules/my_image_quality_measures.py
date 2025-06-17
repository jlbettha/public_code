"""
Created on Wed Mar 20 00:33:40 2024

@author: jlbetthauser
"""

import time
import numpy as np
from scipy import ndimage as ndi
import cv2
from scipy.signal import convolve2d
from skimage.filters import threshold_otsu, sobel
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_laplace
from numpy.typing import NDArray
from numba import njit
from my_distance_metrics import minmax_scaling, bhattacharyya_dist

# from skimage.restoration import estimate_sigma
# from skimage.measure import blur_effect
# from statsmodels.stats.outliers_influence import variance_inflation_factor


def blurriness2(
    image: NDArray[np.float64],
    h_size: int = 11,
) -> float:
    """metric that indicates the strength of blur in an image (0 for no blur, 1 for maximal blur).
            [1] Frederique Crete, et al. "The blur effect: perception and estimation with a new
            no-reference perceptual blur metric" Proc. SPIE 6492 (2007)
            https://hal.archives-ouvertes.fr/hal-00232709:DOI:'10.1117/12.702790'

    Args:
        image (NDArray): image
        h_size (int, optional): Size of the re-blurring filter. Defaults to 11.

    Returns:
        float: Blur metric in [0,1]: by default, the maximum (JLB changed to mean) of blur metrics along all axes.
    """

    B = np.zeros(2)
    slices = tuple([slice(2, s - 1) for s in image.shape])
    for ax in range(image.ndim):
        filt_im = ndi.uniform_filter1d(image, h_size, axis=ax)
        im_sharp = np.abs(sobel(image, axis=ax))
        im_blur = np.abs(sobel(filt_im, axis=ax))
        T = np.maximum(0, im_sharp - im_blur)
        M1 = np.sum(im_sharp[slices])
        M2 = np.sum(T[slices])
        B[ax] = np.abs(M1 - M2) / M1

    return np.max(B)


def otsu_threshold(img: NDArray[np.float64]) -> float:
    """Calculate otsu's threshold

    Args:
        img (NDArray[np.float64]): input image

    Returns:
        float: otsu's threshold
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thr = threshold_otsu(blur)
    # _,thr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thr


def otsu_interclass_distance(img: NDArray[np.float64]) -> float:
    """_summary_

    Args:
        img (NDArray[np.float64]): input image

    Returns:
        float: bhattacharya distance between image data above otsu threshold and below threshold
    """
    threshold = otsu_threshold(img)
    img = img.ravel()
    img = img[np.nonzero(img)]
    img_hi = img[np.where(img >= threshold)]
    img_lo = img[np.where(img < threshold)]
    mu1, v1 = np.mean(img_hi), np.var(img_hi)
    mu2, v2 = np.mean(img_lo), np.var(img_lo)
    if v1 <= 0 or v2 <= 0:
        return 0.0
    if mu1 <= 0 or mu2 <= 0:
        return 0.0
    return bhattacharyya_dist(mu1, v1, mu2, v2)


@njit
def estimate_variance(img: NDArray[np.float64]) -> float:
    img = img.ravel()
    img = img[np.nonzero(img)]
    return np.var(img)


def estimate_noise(img: NDArray[np.float64]) -> float:
    """_summary_

    Args:
        img (NDArray[np.float64]): image

    Returns:
        float: noise estimate
    """
    h, w = img.shape
    mfilter = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])

    sum_t = np.sum(np.absolute(convolve2d(img, mfilter)))
    sigma = sum_t * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    return sigma


## snr
@njit
def signal_to_noise(img: NDArray[np.float64]) -> float:
    """A rough analogue to signal-to-noise ratio of the input data.
        Returns the snr of img, here defined as the mean
        divided by the standard deviation.

    Args:
        img (NDArray[np.float64]): image

    Returns:
        float: snr
    """
    img = img.ravel()
    anz = img[np.nonzero(img)]
    m = np.mean(anz)
    sdv = np.sqrt(np.var(anz))
    if sdv == 0.0:
        return 0.0
    return m / sdv


def laplacian_edge_strength(img: NDArray[np.float64]) -> float:
    """_summary_

    Args:
        img (NDArray[np.float64]): _description_

    Returns:
        float: _description_
    """
    # lap = cv2.convertScaleAbs(cv2.Laplacian(img, 5))
    lap = np.abs(gaussian_laplace(img, sigma=3))
    return np.mean(lap[np.nonzero(lap)])


def jlb_iqa(img: NDArray[np.float64]) -> tuple[float]:
    """Compute all 6 no-reference measures

    Args:
        img (_type_): gray-scale (2D) image

    Returns:
        tuple[float]: tuple of no-reference measures
    """
    snr = signal_to_noise(img)
    est_noise = estimate_noise(img)
    blur2 = blurriness2(img)
    lap_edge_str = laplacian_edge_strength(img)
    est_var = estimate_variance(img)
    otsu = otsu_interclass_distance(img)
    return snr, est_noise, blur2, lap_edge_str, est_var, otsu


def main() -> None:
    img = cv2.imread("einstein.jpg")
    img = rgb2gray(img)
    img = minmax_scaling(img) / 255.0

    snr, est_noise, blur2, lap_edge_str, est_var, otsu = jlb_iqa(img)
    print(f"{snr=}, {est_noise=}, {blur2=}, {lap_edge_str=}, {est_var=}, {otsu=}")


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    tfirst = time.perf_counter() - tmain
    print(f"Program took {tfirst:.3f} seconds.")

    tmain = time.perf_counter()
    main()
    tlast = time.perf_counter() - tmain
    print(f"Program took {tlast:.3f} seconds.")

    print(f"jit speed-up: {tfirst/tlast:.3f}x")
