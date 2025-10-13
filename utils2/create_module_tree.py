import os

do_not_import = [
    ".ipynb_checkpoints",
    ".DS_Store",
    "feedforward",
    "backprop",
    "forward",
    "backward",
    "loss",
    "main",
    "decorator",
    "wrapper",
]


def create_init_file(path):
    init_file_path = os.path.join(path, "__init__.py")
    open(init_file_path, "w").close()
    print(f"Created {init_file_path}")
    return init_file_path


def get_file_function_names(file_path):
    function_names = []
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("def "):
                func_name = line.split("(")[0][4:]  # Extract function name
                if func_name.startswith("_") or func_name in do_not_import:
                    continue
                function_names.append(func_name)

            if line.startswith("class "):
                class_name = line.split("(")[0][6:]  # Extract class name
                if class_name.startswith("_") or class_name in do_not_import:
                    continue
                function_names.append(class_name)

    return function_names


def update_current_init_file(folder_path):
    if folder_path.startswith("_") or folder_path in do_not_import:
        return
    init_file_path = create_init_file(folder_path)

    for f in os.listdir(folder_path):
        if f.startswith("__") or not (f.endswith(".py") or os.path.isdir(os.path.join(folder_path, f))):
            continue

        if f.endswith(".py"):
            module_name = f[:-3]
            with open(init_file_path, "a") as file:
                f_str = f"from .{module_name} import {get_file_function_names(os.path.join(folder_path, f))}\n"
                f_str = f_str.replace("'", "").replace("[", "").replace("]", "").replace(":", "")
                file.write(f_str)
                print(f"Added import for {module_name} in {init_file_path}")
        else:
            module_name = f
            # If it's a folder, we need to create an __init__.py file inside it
            update_current_init_file(os.path.join(folder_path, module_name))
            with open(init_file_path, "a") as file:
                f_str = f"from .{module_name} import *\n"
                file.write(f_str)
                print(f"Added import for {module_name} in {init_file_path}")


def main():
    path_to_module_folder = "/home/jlbet/code/my_utils1/"
    create_init_file(path_to_module_folder)
    update_current_init_file(path_to_module_folder)

    files_and_folders = os.listdir(path_to_module_folder)

    for f in files_and_folders:
        if not (os.path.isdir(os.path.join(path_to_module_folder, f)) or f.endswith(".py")):
            continue

        print(f)


if __name__ == "__main__":
    main()
