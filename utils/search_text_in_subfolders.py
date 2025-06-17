from pathlib import Path
import os


def main() -> None:
    search_target = "edge pot".lower()

    p = Path(".")
    desired_file_extensions = [".txt", ".md", ".py"]

    for root, dir_names, file_names in os.walk(p):
        for f in file_names:
            match_file_name = ""
            found = False
            file_ext = os.path.splitext(f)[-1].lower()
            if file_ext in desired_file_extensions:
                with open(os.path.join(root, f), mode="r") as text_file:
                    for line in text_file:
                        if search_target in line.lower():
                            found = True
                            match_file_name = os.path.join(root, f)
                            break
            if found:
                print(match_file_name)


if __name__ == "__main__":
    main()
