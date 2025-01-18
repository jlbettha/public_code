from pathlib import Path
import os

p = Path(".")

files = [x for x in p.iterdir() if x.is_file()]

for root, dir_names, file_names in os.walk(p):
    for f in file_names:
        if (".txt" in f or ".py" in f) and not ".pyc" in f:
            with open(os.path.join(root, f), mode="r") as f2:
                for line in f2:
                    if "skel" in line:
                        print(f2)
