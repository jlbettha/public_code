from PIL import Image
import pillow_heif
import os

src_dir = "c:\\dir1\\"
dest_dir = "c:\\dir2\\"

for f in os.listdir(src_dir):
    in_filename = src_dir + f
    out_filename = dest_dir + f + ".jpeg"

    heif_file = pillow_heif.read_heif(in_filename)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
    )

    image.save(out_filename, format("jpeg"))
