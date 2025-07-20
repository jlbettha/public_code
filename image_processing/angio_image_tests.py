import copy

import matplotlib

matplotlib.rcParams["image.cmap"] = "gray"
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import (
    frangi,
    meijering,
    sato,
    sobel,
)


def image_norm(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def identity(image):
    """Return the original image, ignoring any kwargs."""
    return image


def main() -> None:  # noqa: PLR0915
    dicom_file = "./data/42211"
    dicom_data = pydicom.dcmread(dicom_file)
    print(dicom_data.pixel_array.shape)

    ## Scale image to desired size 'dim'
    dim = 256
    sigma = 1.0

    orig_image = dicom_data.pixel_array[40, :, :]
    scale = dim / orig_image.shape[0]
    orig_image = ndi.zoom(orig_image, zoom=scale)

    ## Smoothe image, if desired
    img_smoothed = copy.deepcopy(orig_image)
    img_smoothed = ndi.gaussian_filter(img_smoothed, sigma=sigma)
    img_smoothed = image_norm(img_smoothed)

    ## Edge Potential
    img_edgepot = copy.deepcopy(img_smoothed)
    img_edgepot = ndi.gaussian_gradient_magnitude(img_edgepot, sigma=sigma)  # gradient
    # img_edgepot = image_norm(img_edgepot)  # normalize
    img_edgepot = 1 / np.abs(1 + img_edgepot)  # P = 1/ |1+ grad(I)|
    img_edgepot = 1 - img_edgepot

    ## Edge detection, types: canny, roberts, sobel, scharr
    struct = ndi.generate_binary_structure(2, 2)  # noqa: F841

    img_canny_edges = feature.canny(img_smoothed, sigma=sigma).astype(float)
    # img_canny_edges = ndi.binary_opening(img_canny_edges, structure=struct, iterations=5).astype(float)
    # img_canny_edges = ndi.binary_closing(img_canny_edges, structure=None, iterations=2).astype(
    #     float
    # )
    # img_canny_edges = ndi.binary_erosion(img_canny_edges, structure=None, iterations=1).astype(
    #     float
    # )
    # img_canny_edges = ndi.binary_dilation(img_canny_edges, structure=None, iterations=1).astype(
    #     float
    # )

    ## Sobel edges
    img_sobel = sobel(img_smoothed)
    img_sobel = np.abs(img_sobel)
    img_sobel = image_norm(img_sobel)

    ## Composite image
    img_composite = image_norm((1 - img_smoothed) * img_edgepot)

    plt.figure(figsize=(9, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(orig_image, aspect="auto")
    plt.axis("off")
    plt.title("Original Image")

    plt.subplot(2, 3, 2)
    plt.imshow(img_edgepot, aspect="auto")
    plt.axis("off")
    plt.title("Edge Potential")

    plt.subplot(2, 3, 3)
    plt.imshow(img_canny_edges, aspect="auto")
    plt.axis("off")
    plt.title("Canny Edges")

    plt.subplot(2, 3, 4)
    plt.imshow(img_sobel, aspect="auto")
    plt.axis("off")
    plt.title("Sobel Edges")

    plt.subplot(2, 3, 5)
    plt.imshow(img_composite, aspect="auto")
    plt.axis("off")
    plt.title("Composite Image")

    plt.subplot(2, 3, 6)
    kwargs = {}
    kwargs["sigmas"] = [sigma]
    kwargs["black_ridges"] = 0
    image = img_smoothed
    result = frangi(1 - image, **kwargs)
    print(result.shape)
    result = image_norm(result[4:-4, 4:-4])
    plt.imshow(result, aspect="auto")
    plt.title('Frangi "Vesselness"')
    plt.axis("off")
    plt.tight_layout()
    # plt.plot(xs, ys, c="r", linewidth=1)
    plt.pause(0.001)

    kwargs = {}
    kwargs["sigmas"] = [sigma]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, black_ridges in enumerate([1, 0]):
        for j, func in enumerate([identity, sato, frangi]):
            kwargs["black_ridges"] = black_ridges
            result = func(image, **kwargs)
            if func in (meijering, frangi):
                # Crop by 4 pixels for rendering purpose.
                result = result[4:-4, 4:-4]
            result = image_norm(result)
            axes[i, j].imshow(result, aspect="auto")
            if i == 0:
                axes[i, j].set_title(["Smoothed\nImage", 'Sato\n"Tubeness"', 'Frangi\n"Vesselness"'][j])
            if j == 0:
                axes[i, j].set_ylabel(f"{black_ridges=}")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
