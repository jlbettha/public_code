import time
import numpy as np
import cv2
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage import feature
from my_distance_metrics import minmax_scaling

# import matplotlib.pyplot as plt


def image_gaussian_smooth(
    image: NDArray[np.float64], sigma: float = 1.0
) -> NDArray[np.float64]:
    """_summary_

    Args:
        image (NDArray[np.float64]): _description_
        sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        NDArray[np.float64]: _description_
    """
    image = ndi.gaussian_filter(image, sigma=sigma)
    return minmax_scaling(image, max_val=1)


def image_edge_potential(
    image: NDArray[np.float64], sigma: float = 1.0
) -> NDArray[np.float64]:
    """_summary_

    Args:
        image (NDArray[np.float64]): _description_
        sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        NDArray[np.float64]: _description_
    """
    edge_pot = ndi.gaussian_gradient_magnitude(image, sigma=1.0)  # gradient
    edge_pot = 1 / np.abs(1 + edge_pot)  # P = 1/ |1+ grad(I)|
    return 1 - edge_pot


def canny_edge_mask(image, sigma=1.0) -> NDArray[np.float64]:
    img_smoothed = image_gaussian_smooth(image, sigma=sigma)
    return feature.canny(img_smoothed, sigma=sigma).astype(float)


def image_color(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Color. We use the standard RGB colorspace. Note that any improvements to color
        histograms (such as better colorspaces) can also be applied to joint histograms.

    Args:
        image (NDArray[np.float64]): input image (m x n)

    Returns:
        NDArray[np.float64]: image_color (m x n)
    """
    num_colors = 64
    return 0.0


def image_edge_density(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Edge density. We define the edge density at pixel (j, k) to be the ratio of edges to
        pixels in a small neighborhood surrounding the pixel. The edge representation of the
        image is computed with a standard method [12].

    Args:
        image (NDArray[np.float64]): input image (m x n)

    Returns:
        NDArray[np.float64]: image_edge_density (m x n)
    """
    num_densities = 4

    return 0.0


def image_texturedness(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Texturedness. We define the texturedness at pixel (j, k) to be the number of neighboring
        pixels whose intensities differ by more than a fixed value. This definition is similar to
        the texturedness feature used by Engelson [3] for place recognition.

    Args:
        image (NDArray[np.float64]): input image (m x n)

    Returns:
        NDArray[np.float64]: image_texturedness (m x n)
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


def image_gradient_magnitude(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gradient magnitude. Gradient magnitude is a measure of how rapidly intensity is
        changing in the direction of greatest change. The gradient magnitude at a pixel (j, k)
        is computed using standard methods [8]

    Args:
        image (NDArray[np.float64]): input image (m x n)

    Returns:
        NDArray[np.float64]: image_gradient_magnitude (m x n)
    """
    image = image_gaussian_smooth(image, sigma=1.0)
    img_grads = np.abs(cv2.Laplacian(image, cv2.CV_64F))
    return img_grads


def image_rank(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rank. The rank of pixel (j, k) is defined as the number of pixels in the local 4c(+) neighborhood
        whose intensity is less than the intensity at (j, k). This feature can be used to compute
        optical flow.

    Args:
        image (NDArray[np.float64]): input image (m x n)

    Returns:
        NDArray[np.float64]: image_rank (m x n)
    """

    img_ranks = np.zeros((image.shape[0], image.shape[1]))
    for i in np.arange(1, image.shape[0] - 1, 1):
        for j in np.arange(1, image.shape[1] - 1, 1):
            pxl_val = image[i, j]
            # local_region = image[i - 1 : i + 1, j - 1 : j + 1]
            local_region = [image[i - 1 : i + 1, j], image[i, j - 1 : j + 1]]
            img_ranks[i, j] = np.sum(local_region < pxl_val)

    return img_ranks


def build_jointhist_n(
    img_colors, img_edges, img_textures, img_grads, img_ranks, n=5
) -> NDArray[np.float64]:
    img_colors
    img_edges
    img_textures
    img_grads
    img_ranks
    n
    return None


def jh5_image_features(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Betthauser (implementation) 2023 -- Calculate JH4 image features (for search)
        JointHistograms5: color, edge density, texturedness, gradient magnitude, rank
        G. Pass and R. Zabih, 'Comparing Images Using Joint Histograms,'
                Computer Science Department, Cornell University, NY.

    "__We allowed for 64 possible discrete colors in the image, 4 possible values of edge density,
    4 possible values of texturedness, 5 possible values of gradient magnitude, and 4 possible
    ranks. Color histograms were also implemented with 64 colors.1 Both color histograms and
    joint histograms were compared using the L1 distance.__"

    Args:
        image (NDArray[np.float64]): input image

    Returns:
        tuple[NDArray[np.float64]]: joint_hist_5 image features for comparison
    """
    num_colors = 64
    num_edges = 4
    num_textures = 4
    num_grads = 5  # ?
    num_ranks = 4

    img_colors = image_color(image)

    img_edges = image_edge_density(image)
    img_textures = image_texturedness(image)

    img_grads = image_gradient_magnitude(image)
    img_ranks = image_rank(image)

    joint_hist_5 = build_jointhist_n(
        img_colors, img_edges, img_textures, img_grads, img_ranks, n=5
    )
    return joint_hist_5


def main() -> None:
    """_summary_"""
    pass


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time() - t0:.3f} seconds")
