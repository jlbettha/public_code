"""
Created on Wed Dec  1 14:07:16 2021

@author: jlb235
"""

import os
import time

import cv2
import numpy as np
import pytesseract
from PIL import Image


def image_to_textfile(image_path: str, output_textfile_path: str) -> None:
    imgf = image_path.split("/")[-1]

    with Image.open(image_path) as image:
        img = np.array(image)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        scale_factor = 2048 / img.shape[1]
        img = cv2.resize(
            img,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

        img = cv2.GaussianBlur(img, (3, 3), 0)

        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # ensure white background
        fraction_white_pixels = np.count_nonzero(img) / np.prod(img.shape)
        if fraction_white_pixels < 0.5:  # noqa: PLR2004
            img = 255 - img

        # img = cv2.medianBlur(img, kernel_size)

        text = pytesseract.image_to_string(img)

    txtf = imgf.replace(os.path.splitext(imgf)[-1], ".txt")

    with open(os.path.join(output_textfile_path, txtf), "w") as txt_file:
        txt_file.write(text)


def image_directory_to_textfiles(image_path: str, output_textfile_path: str) -> None:
    f_exts = [".png", ".jpg", ".jpeg"]

    ### desired images
    image_files = [
        f
        for f in os.listdir(image_path)
        if os.path.isfile(image_path + f) and os.path.splitext(f)[-1].lower() in f_exts
    ]

    for imgf in image_files:
        image_to_textfile(os.path.join(image_path, imgf), output_textfile_path)


def main() -> None:
    ### Set folders and paths
    image_path = "C:/Users/jlbetthauser/OneDrive/Images/"
    text_path = "C:/Users/jlbetthauser/Documents/Code/Python/temp_code_playground/image2text_outputs/"

    #### images to text
    t0 = time.perf_counter()

    image_directory_to_textfiles(image_path, text_path)

    print(f"Image conversion took {time.perf_counter() - t0:.6f} seconds")


if __name__ == "__main__":
    main()
