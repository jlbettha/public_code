import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def minmax_scaling(arr: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """Normalize a 1D or 2D numpy array to the range [0, max_val]."""
    min_val = np.min(arr)
    return max_val * (arr - min_val) / (np.max(arr) - min_val)


def autocorrelate_image_scipy(image: np.ndarray, subtract_mean: bool = True):
    if image is None:
        msg = "Invalid image provided."
        raise ValueError(msg)

    if subtract_mean:
        image = image - np.mean(image)

    # Perform 2D autocorrelation
    result = signal.correlate2d(image, image, mode="same")

    # Normalize result for visualization
    return result / np.max(result)


def autocorrelate_image_fft(image: np.ndarray, subtract_mean: bool = True, full_fft: bool = True) -> np.ndarray:
    if image is None:
        msg = "Invalid image provided."
        raise ValueError(msg)

    if subtract_mean:
        image = image - np.mean(image)

    # Perform 2D autocorrelation using FFT
    if full_fft:
        m, n = image.shape
        f1 = np.fft.fft2(image, s=(m * 2 - 1, n * 2 - 1))
        result = np.fft.ifft2(f1 * np.conj(f1)).real
        # result = np.fft.fftshift(result)[M // 2 : -M // 2 + 1, N // 2 : -N // 2 + 1]
        result = np.roll(result, m // 2 - 1, axis=0)[:m, :]
        result = np.roll(result, n // 2 - 1, axis=1)[:, :n]
    else:
        f1 = np.fft.fft2(image)
        complexp = np.fft.ifft2(f1 * np.conj(f1))
        result = np.fft.fftshift(complexp.real)

    # Normalize result for visualization
    return result / np.max(result)


def main():
    pwd = pathlib.Path(__file__).parent.resolve()  ## Get the directory of the current script
    print(f"Present working directory: {pwd}")
    image_path = pwd / "data" / "einstein.jpg"

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = minmax_scaling(img, max_val=1.0)

    result_norm = autocorrelate_image_scipy(img, subtract_mean=True)
    result_norm_fft = autocorrelate_image_fft(img, subtract_mean=True)

    abs_diff = np.abs(result_norm_fft - result_norm)
    print(f"Max difference (FFT vs SciPy): {np.max(abs_diff)}")
    print(f"Total difference (FFT vs SciPy): {np.sum(abs_diff)}")
    if np.allclose(result_norm, result_norm_fft, atol=1e-7):
        print("Results from FFT and SciPy are a very close match! FFT is ~100x faster.")

    # Save or display result(s)
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="plasma")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(result_norm, cmap="plasma")
    plt.colorbar()
    plt.title("Autocorrelation")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(abs_diff, cmap="plasma")
    plt.colorbar()
    plt.title("| diff |")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(result_norm_fft, cmap="plasma")
    plt.colorbar()
    plt.title("Autocorrelation (FFT)")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
