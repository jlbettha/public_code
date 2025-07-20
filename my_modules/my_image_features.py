import cv2
import numpy as np
from my_distance_metrics import minmax_scaling
from scipy import ndimage as ndi
from skimage import feature

# import matplotlib.pyplot as plt


def image_gaussian_smooth(image: np.ndarray[float], sigma: float = 1.0) -> np.ndarray[float]:
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


def image_edge_potential(image: np.ndarray[float], sigma: float = 1.0) -> np.ndarray[float]:
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


def canny_edge_mask(image: np.ndarray[float], sigma=1.0) -> np.ndarray[float]:
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


def image_color(image: np.ndarray[float]) -> np.ndarray[float]:  # noqa: ARG001
    """
    Color. We use the standard RGB colorspace. Note that any improvements to color
        histograms (such as better colorspaces) can also be applied to joint histograms.

    Args:
        image (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_color (m x n)

    """
    # num_colors = 64
    return NotImplementedError("image_color not implemented yet")


def image_edge_density(image: np.ndarray[float]) -> np.ndarray[float]:  # noqa: ARG001
    """
    Edge density. We define the edge density at pixel (j, k) to be the ratio of edges to
        pixels in a small neighborhood surrounding the pixel. The edge representation of the
        image is computed with a standard method [12].

    Args:
        image (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_edge_density (m x n)

    """
    # num_densities = 4

    return NotImplementedError("image_edge_density not implemented yet")


def image_texturedness(image: np.ndarray[float]) -> np.ndarray[float]:
    """
    Texturedness. We define the texturedness at pixel (j, k) to be the number of neighboring
        pixels whose intensities differ by more than a fixed value. This definition is similar to
        the texturedness feature used by Engelson [3] for place recognition.

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


def image_gradient_magnitude(image: np.ndarray[float]) -> np.ndarray[float]:
    """
    Gradient magnitude. Gradient magnitude is a measure of how rapidly intensity is
        changing in the direction of greatest change. The gradient magnitude at a pixel (j, k)
        is computed using standard methods [8]

    Args:
        image (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_gradient_magnitude (m x n)

    """
    image = image_gaussian_smooth(image, sigma=1.0)
    return np.abs(cv2.Laplacian(image, cv2.CV_64F))


def image_rank(image: np.ndarray[float]) -> np.ndarray[float]:
    """
    Rank. The rank of pixel (j, k) is defined as the number of pixels in the local 4c(+) neighborhood
        whose intensity is less than the intensity at (j, k). This feature can be used to compute
        optical flow.

    Args:
        image (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_rank (m x n)

    """
    img_ranks = np.zeros((image.shape[0], image.shape[1]))
    for i in np.arange(1, image.shape[0] - 1, 1):
        for j in np.arange(1, image.shape[1] - 1, 1):
            pxl_val = image[i, j]
            # local_region = image[i - 1 : i + 1, j - 1 : j + 1]
            local_region = [image[i - 1 : i + 1, j], image[i, j - 1 : j + 1]]
            img_ranks[i, j] = np.sum(local_region < pxl_val)

    return img_ranks


def build_jointhist_n(img_colors, img_edges, img_textures, img_grads, img_ranks, n=5) -> np.ndarray[float]:  # noqa: ARG001
    # img_colors
    # img_edges
    # img_textures
    # img_grads
    # img_ranks
    # n
    return NotImplementedError("build_jointhist_n not implemented yet")


def jh5_image_features(image: np.ndarray[float]) -> np.ndarray[float]:
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

    img_colors = image_color(image)

    img_edges = image_edge_density(image)
    img_textures = image_texturedness(image)

    img_grads = image_gradient_magnitude(image)
    img_ranks = image_rank(image)

    return build_jointhist_n(img_colors, img_edges, img_textures, img_grads, img_ranks, n=5)


def main() -> None:
    print("my_image_features.py is a module")


if __name__ == "__main__":
    main()
