"""
Created on Wed Dec  1 14:07:16 2021

@author: jlb235
"""

import os, time
from pypdf import PdfReader


def main() -> None:

    ### Set folders and paths
    input_root = "./data/"
    print(input_root)

    ### Load desired file(s)
    pdf_files = [
        f
        for f in os.listdir(input_root)
        if os.path.isfile(input_root + f) and (f.endswith(".pdf") or f.endswith(".PDF"))
    ]
    print(pdf_files)

    #### PDF to text
    t0 = time.time()
    pdf = PdfReader(input_root + "nsf-reference-writers-guide.pdf")
    num_pages = pdf.get_num_pages()
    # info = pdf.getDocumentInfo()
    # print(info)

    for i in range(num_pages):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("PDF Page:", i + 1)
        page = pdf.get_page(i)
        text = page.extract_text()
        print(text)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("PDF Read took {:.6f} seconds".format(time.time() - t0))


if __name__ == "__main__":
    main()
