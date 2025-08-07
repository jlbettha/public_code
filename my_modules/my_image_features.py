import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from my_distance_metrics import minmax_scaling
from numba import njit
from scipy import ndimage as ndi
from skimage import feature

EPS = 1e-12


@njit
def normalize_ndarray(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Normalize a 1D or 2D numpy array to the range [0, 1]."""
    min_val = np.min(arr)
    return scale * (arr - min_val) / (np.max(arr) - min_val)


def image_gaussian_smooth(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    _summary_

    Args:
        image (np.ndarray[float]): _description_
        sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        np.ndarray[float]: _description_

    """
    image = ndi.gaussian_filter(image, sigma=sigma)
    return minmax_scaling(image, max_val=1)


def image_edge_potential(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    _summary_

    Args:
        image (np.ndarray[float]): _description_
        sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        np.ndarray[float]: _description_

    """
    edge_pot = ndi.gaussian_gradient_magnitude(image, sigma=sigma)  # gradient
    edge_pot = 1 / np.abs(1 + edge_pot)  # P = 1/ |1+ grad(I)|
    return 1 - edge_pot


def canny_edge_mask(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    _summary_

    Args:
        image (np.ndarray[float]): _description_
        sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        np.ndarray[float]: _description_

    """
    img_smoothed = image_gaussian_smooth(image, sigma=sigma)
    return feature.canny(img_smoothed, sigma=sigma).astype(float)


@njit
def image_color_histogram(image: np.ndarray, bins: int = 64) -> np.ndarray:
    """
    Generate color histogram of images. # num_colors = 64

    Args:
        image (np.ndarray): _description_
        bins (int, optional): _description_. Defaults to 64.

    Returns:
        np.ndarray: _description_

    """
    bins_per_color = int(np.round(bins ** (1 / 3)))
    image = np.floor(normalize_ndarray(image, scale=bins_per_color - EPS)).astype(np.uint64)

    rgb_histogram = np.zeros((bins_per_color, bins_per_color, bins_per_color), dtype=np.uint64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x, y, z = image[i, j, :]
            rgb_histogram[x, y, z] += 1

    return rgb_histogram


def image_edge_density(image: np.ndarray) -> np.ndarray:
    """
    Edge density. We define the edge density at pixel (j, k) to be the ratio of edges to
        pixels in a small neighborhood surrounding the pixel. The edge representation of the
        image is computed with a standard method [12]. # num_edges = 4

    Args:
        image (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_edge_density (m x n)

    """
    # num_densities = 4
    img_ranks = np.zeros((image.shape[0], image.shape[1])).astype(np.uint64)
    for i in np.arange(1, image.shape[0] - 1, 1):
        for j in np.arange(1, image.shape[1] - 1, 1):
            pxl_val = image[i, j]
            # local_region = image[i - 1 : i + 1, j - 1 : j + 1]
            local_region = [image[i - 1 : i + 1, j], image[i, j - 1 : j + 1]]
            img_ranks[i, j] = np.sum(local_region < pxl_val)
    return NotImplementedError("image_edge_density not implemented yet")


def image_texturedness(image: np.ndarray) -> np.ndarray:
    """
    Texturedness. We define the texturedness at pixel (j, k) to be the number of neighboring
        pixels whose intensities differ by more than a fixed value. This definition is similar to
        the texturedness feature used by Engelson [3] for place recognition. # num_textures = 4

    Args:
        image (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_texturedness (m x n)

    """
    tol = 0.1
    image = minmax_scaling(image, max_val=1)
    img_textures = np.zeros((image.shape[0], image.shape[1]))
    for i in np.arange(1, image.shape[0] - 1, 1):
        for j in np.arange(1, image.shape[1] - 1, 1):
            pxl_val = image[i, j]
            # local_region = image[i - 1 : i + 1, j - 1 : j + 1]
            local_region = [image[i - 1 : i + 1, j], image[i, j - 1 : j + 1]]
            diff = np.abs(local_region - pxl_val)
            img_textures[i, j] = np.sum(diff > tol)

    return img_textures


def image_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Gradient magnitude. Gradient magnitude is a measure of how rapidly intensity is
        changing in the direction of greatest change. The gradient magnitude at a pixel (j, k)
        is computed using standard methods [8].     # num_grads = 5

    Args:
        image (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_gradient_magnitude (m x n)

    """
    # gx = np.gradient(image[:,:,0])
    # print(gx[0])
    # return gx
    image = image_gaussian_smooth(image, sigma=1.0)
    return np.abs(cv2.Laplacian(image, cv2.CV_64F))


# @njit
def image_rank(image: np.ndarray) -> np.ndarray:
    """
    Rank. The rank of pixel (j, k) is defined as the number of pixels in the local 4c(+) neighborhood
        whose intensity is less than the intensity at (j, k). This feature can be used to compute
        optical flow.    # num_ranks = 4

    Args:
        image (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_rank (m x n)

    """
    img_ranks = np.zeros((image.shape[0], image.shape[1])).astype(np.uint64)
    for i in np.arange(1, image.shape[0] - 1, 1):
        for j in np.arange(1, image.shape[1] - 1, 1):
            pxl_val = image[i, j]
            # local_region = image[i - 1 : i + 1, j - 1 : j + 1]
            local_region = [image[i - 1 : i + 1, j], image[i, j - 1 : j + 1]]
            img_ranks[i, j] = np.sum(local_region < pxl_val)

    return img_ranks


def build_jointhist_n(img_colors, img_edges, img_textures, img_grads, img_ranks, n=5) -> np.ndarray:  # noqa: ARG001
    # img_colors
    # img_edges
    # img_textures
    # img_grads
    # img_ranks
    # n
    return NotImplementedError("build_jointhist_n not implemented yet")


def jh5_image_features(image: np.ndarray) -> np.ndarray:
    """
    Betthauser (implementation) 2023 -- Calculate JH4 image features (for search)
        JointHistograms5: color, edge density, texturedness, gradient magnitude, rank
        G. Pass and R. Zabih, 'Comparing Images Using Joint Histograms,'
                Computer Science Department, Cornell University, NY.

    "__We allowed for 64 possible discrete colors in the image, 4 possible values of edge density,
    4 possible values of texturedness, 5 possible values of gradient magnitude, and 4 possible
    ranks. Color histograms were also implemented with 64 colors.1 Both color histograms and
    joint histograms were compared using the L1 distance.__"

    Args:
        image (np.ndarray[float]): input image

    Returns:
        tuple[np.ndarray[float]]: joint_hist_5 image features for comparison

    """
    # num_colors = 64
    # num_edges = 4
    # num_textures = 4
    # num_grads = 5  # ?
    # num_ranks = 4

    img_colors = image_color_histogram(image)

    img_edges = image_edge_density(image)
    img_textures = image_texturedness(image)

    img_grads = image_gradient_magnitude(image)
    img_ranks = image_rank(image)

    return build_jointhist_n(img_colors, img_edges, img_textures, img_grads, img_ranks, n=5)


def main() -> None:
    # rng = np.random.default_rng()
    # image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint64)  # Random image
    image = cv2.imread("lena-headey.jpg", cv2.IMREAD_COLOR)  # Read the image
    # image_norm = minmax_scaling(image[:, :, 0], max_val=1)
    # image_gauss = image_gaussian_smooth(image[:, :, 0], sigma=1.0)
    # img_colors = image_color_histogram(image)
    # img_edge_potential = minmax_scaling(image_edge_potential(image_gauss))
    # img_canny = canny_edge_mask(image[:, :, 0])
    # img_edges = image_edge_potential(image)  # Use edge potential instead of canny edges
    # img_edges = image_edge_density(image)  # Use edge density instead of canny edges
    # img_edges = cv2.Canny(image, 100, 200)  # Canny edge detection

    # img_edges = image_edge_density(image)
    # img_textures = image_texturedness(image)

    # img_grads = minmax_scaling(image_gradient_magnitude(image[:, :, 0]))
    # img_ranks = image_rank(image)

    # STAR:
    star = cv2.xfeatures2d.StarDetector_create()

    # FAST:
    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)

    # BRIEF:
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16, use_orientation=False)

    # BRISK:
    brisk = cv2.BRISK_create(thresh=30, octaves=0, patternScale=1.0)

    # AKAZE:
    akaze = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size=0,
        descriptor_channels=3,
        threshold=0.001,
        nOctaves=4,
        nOctaveLayers=4,
        diffusivity=cv2.KAZE_DIFF_PM_G2,
    )

    # FREAK:
    freak = cv2.xfeatures2d.FREAK_create(
        orientationNormalized=True, scaleNormalized=True, patternScale=22.0, nOctaves=4
    )

    keypts = star.detect(image, None)
    keypts, descriptors = brief.compute(image, keypts)

    # Draw only 50 keypoints on input image
    kp_image = cv2.drawKeypoints(
        image=image,
        keypoints=keypts,
        outImage=None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    print("Keypoints:", keypts, "\n")

    print("Descriptors:", descriptors, "\n")

    plt.imshow(kp_image)
    plt.show()

    # sys.exit()
    # plt.figure(figsize=(18, 10))
    # plt.subplot(2, 4, 1)
    # plt.imshow(image)
    # plt.title("Original Image")
    # plt.axis("off")
    # plt.subplot(2, 4, 2)
    # plt.imshow(img_textures)
    # plt.title("Texturedness")
    # plt.axis("off")
    # plt.subplot(2, 4, 3)
    # plt.imshow(img_grads)
    # plt.title("Gradient Magnitude")
    # plt.axis("off")
    # plt.subplot(2, 4, 4)
    # plt.imshow(img_ranks)
    # plt.title("Image Ranks")
    # plt.axis("off")

    # plt.subplot(2, 4, 5)
    # plt.imshow(img_edge_potential)
    # plt.title("Edge Potential")
    # plt.axis("off")
    # plt.subplot(2, 4, 6)
    # plt.imshow(img_canny)
    # plt.title("Canny Edge Mask")
    # plt.axis("off")
    # plt.subplot(2, 4, 7)
    # plt.imshow(image_norm, cmap="gray")
    # plt.title("Normalized Image")
    # plt.axis("off")
    # plt.subplot(2, 4, 8)
    # plt.imshow(image_gauss, cmap="gray")
    # plt.title("Gaussian Smoothed Image")
    # plt.axis("off")

    # plt.show()


if __name__ == "__main__":
    main()
