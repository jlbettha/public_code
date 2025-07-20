"""
Created on Wed Dec  1 14:07:16 2021

@author: jlb235
"""

import os
import time

from pypdf import PdfReader


def pdf_to_textfile(pdf_path: str, output_textfile_path: str) -> None:
    with open(pdf_path, "rb") as pdf_file:
        pdf = PdfReader(pdf_file)
        page = pdf.pages[0]

        num_pages = pdf.get_num_pages()
        text = ""
        for page_num in range(num_pages):
            page = pdf.get_page(page_num)
            text = page.extract_text(
                extraction_mode="layout",
                layout_mode_space_vertically=False,
                layout_mode_strip_rotated=False,
            )
            print(text)
            with open(output_textfile_path, "w") as txt_file2:
                txt_file2.write(text)


def pdf_directory_to_textfiles(pdf_root: str, text_root: str) -> None:
    f_exts = [".pdf"]

    pdf_files = [
        f for f in os.listdir(pdf_root) if os.path.isfile(pdf_root + f) and os.path.splitext(f)[-1].lower() in f_exts
    ]

    for pf in pdf_files:
        txt_file = pf.replace(os.path.splitext(pf)[-1], ".txt")
        pdf_to_textfile(pdf_root + pf, text_root + txt_file)


def main() -> None:
    t0 = time.perf_counter()

    pdf_root = "C:/Users/jlbetthauser/Documents/Code/Python/temp_code_playground/image2text_outputs/"
    pdf_directory_to_textfiles(pdf_root, pdf_root)

    print(f"PDF reads took {time.perf_counter() - t0:.6f} seconds")


if __name__ == "__main__":
    main()
