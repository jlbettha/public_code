import cv2
# import matplotlib

# matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from numba import njit
from scipy import ndimage as ndi
from skimage import feature

EPS = 1e-12
plt.rcParams["image.cmap"] = "plasma"  # Set default colormap to plasma


@njit
def normalize_ndarray(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Normalize a 1D or 2D numpy array to the range [0, 1]."""
    min_val = np.min(arr)
    return scale * (arr - min_val) / (np.max(arr) - min_val)


def image_gaussian_smooth(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    _summary_

    Args:
        img (np.ndarray[float]): _description_
        sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        np.ndarray[float]: _description_

    """
    img = ndi.gaussian_filter(img, sigma=sigma)
    return normalize_ndarray(img, scale=1)


def image_gradient(ix: np.ndarray, iy: np.ndarray) -> tuple[np.ndarray]:
    """Compute image gradient an partial differentials [Ix, Iy] of image Ir.

    Args:
        ix (np.ndarray): _description_
        iy (np.ndarray): _description_

    Returns:
        tuple[np.ndarray]: _description_
    """
    grad_mag = np.sqrt(ix**2 + iy**2)
    grad_angle = np.atan2(iy, ix)
    return grad_mag, grad_angle


def image_LoG(img: np.ndarray, hsize: int, sigma: float) -> np.ndarray:
    """Compute laplacian of gaussian.

    Args:
        img (np.ndarray): _description_
        hsize (int): _description_
        sigma (float): _description_

    Returns:
        np.ndarray: _description_
    """
    if hsize is None:
        size = int(6 * sigma + 1) if sigma >= 1 else 7

    if hsize % 2 == 0:
        hsize += 1

    x, y = np.meshgrid(
        np.arange(-size // 2 + 1, size // 2 + 1),
        np.arange(-size // 2 + 1, size // 2 + 1),
    )
    h = -(1 / (np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    h = h / np.sum(np.abs(h))
    imf = convolve2d(img, h, mode="same", boundary="symm")
    return imf


def image_partials_sobel_conv(img: np.ndarray) -> np.ndarray:
    """Convolve sobel x,y with image (edges)

    Args:
        img (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ix = convolve2d(img, sobelx, mode="same", boundary="symm")
    sobely = sobelx.T
    Iy = convolve2d(img, sobely, mode="same", boundary="symm")
    return Ix, Iy


def image_edge_potential(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Edge potential, e.g., for GVF and active contours.

    Args:
        img (np.ndarray[float]): _description_
        sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        np.ndarray[float]: _description_

    """
    edge_pot = ndi.gaussian_gradient_magnitude(img, sigma=sigma)  # gradient
    edge_pot = 1 / np.abs(1 + edge_pot)  # P = 1/ |1+ grad(I)|
    return 1 - edge_pot


def canny_edge_mask(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Highlight edges with canny edge detector.

    Args:
        img (np.ndarray[float]): _description_
        sigma (float, optional): _description_. Defaults to 1.0.

    Returns:
        np.ndarray[float]: _description_

    """
    img_smoothed = image_gaussian_smooth(img, sigma=sigma)
    return feature.canny(img_smoothed, sigma=sigma).astype(float)


def image_edge_density(img: np.ndarray) -> np.ndarray:
    """
    Edge density. We define the edge density at pixel (j, k) to be the ratio of edges to
        pixels in a small neighborhood surrounding the pixel. The edge representation of the
        image is computed with a standard method [12]. # num_edges = 4

    Args:
        img (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_edge_density (m x n)

    """
    # num_densities = 4

    return NotImplementedError("image_edge_density not implemented yet")


def image_texturedness(img: np.ndarray, hsize: int = 3) -> np.ndarray:
    """
    Texturedness. We define the texturedness at pixel (j, k) to be the number of neighboring
        pixels whose intensities differ by more than a fixed value. This definition is similar to
        the texturedness feature used by Engelson [3] for place recognition. # num_textures = 4

    Args:
        img (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_texturedness (m x n)

    """
    tol = 0.1
    hhalf = hsize // 2
    img = normalize_ndarray(img, scale=1)
    img_textures = np.zeros((img.shape[0], img.shape[1]))
    for i in np.arange(hhalf, img.shape[0] - hhalf + 1, 1):
        for j in np.arange(hhalf, img.shape[1] - hhalf + 1, 1):
            pxl_val = img[i, j]
            local_region = img[i - hhalf : i + hhalf, j - hhalf : j + hhalf]
            diff = np.abs(local_region - pxl_val)
            img_textures[i, j] = np.sum(diff[diff > tol])

    return img_textures


def image_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Gradient magnitude. Gradient magnitude is a measure of how rapidly intensity is
        changing in the direction of greatest change. The gradient magnitude at a pixel (j, k)
        is computed using standard methods [8].     # num_grads = 5

    Args:
        img (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_gradient_magnitude (m x n)

    """
    img = image_gaussian_smooth(img, sigma=1.0)
    return np.abs(cv2.Laplacian(img, cv2.CV_64F))


# @njit
def image_rank(img: np.ndarray, hsize: int = 3) -> np.ndarray:
    """
    Rank. The rank of pixel (j, k) is defined as the number of pixels in the local 4c(+) neighborhood
        whose intensity is less than the intensity at (j, k). This feature can be used to compute
        optical flow.    # num_ranks = 4

    Args:
        img (np.ndarray[float]): input image (m x n)

    Returns:
        np.ndarray[float]: image_rank (m x n)

    """
    hhalf = hsize // 2
    img_ranks = np.zeros((img.shape[0], img.shape[1])).astype(np.uint64)
    for i in np.arange(hhalf, img.shape[0] - hhalf + 1, 1):
        for j in np.arange(1, img.shape[1] - hhalf + 1, 1):
            pxl_val = img[i, j]
            local_region = img[i - hhalf : i + hhalf, j - hhalf : j + hhalf]
            img_ranks[i, j] = np.sum(local_region < pxl_val)

    return img_ranks


@njit
def image_color_histogram(image: np.ndarray, bins: int = 64) -> np.ndarray:
    """
    Generate color histogram of images.

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


def main():
    # rng = np.random.default_rng()
    # image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint64)  # Random image
    image = cv2.imread("my_modules/einstein.jpg", cv2.IMREAD_COLOR)  # Read the image
    image_norm = normalize_ndarray(image[:, :, 0])
    image_gauss = image_gaussian_smooth(image[:, :, 0], sigma=2.0)
    img_edge_potential = normalize_ndarray(image_edge_potential(image_gauss))
    img_canny = canny_edge_mask(image[:, :, 0])
    # img_edges = image_edge_density(image)  # Use edge density instead of canny edges

    img_textures = image_texturedness(image[:, :, 0], hsize=7)

    img_grads = normalize_ndarray(image_gradient_magnitude(image[:, :, 0]))
    img_ranks = image_rank(image[:, :, 0], hsize=7)
    print(f"Unique values in image ranks: {len(np.unique(img_ranks))}")
    print(f"Unique values in image textures: {len(np.unique(img_textures))}")

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(2, 4, 2)
    plt.imshow(img_textures)
    plt.title("Texturedness")
    # plt.colorbar()
    plt.axis("off")
    plt.subplot(2, 4, 3)
    plt.imshow(img_grads)
    plt.title("Gradient Magnitude")
    # plt.colorbar()
    plt.axis("off")
    plt.subplot(2, 4, 4)
    plt.imshow(img_ranks)
    plt.title("Image Ranks")
    # plt.colorbar()
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(img_edge_potential)
    plt.title("Edge Potential")
    # plt.colorbar()
    plt.axis("off")
    plt.subplot(2, 4, 6)
    plt.imshow(img_canny)
    plt.title("Canny Edge Mask")
    plt.axis("off")
    plt.subplot(2, 4, 7)
    plt.imshow(image_norm, cmap="gray")
    plt.title("Normalized Image")
    plt.axis("off")
    plt.subplot(2, 4, 8)
    plt.imshow(image_gauss, cmap="gray")
    plt.title("Gaussian Smoothed Image")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
