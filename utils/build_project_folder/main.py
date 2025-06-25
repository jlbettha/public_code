import os
import time
import shutil

PROJECT_NAME = "my_project"
FOLDERS = [
    "config",
    "data",
    "data/raw",
    "data/processed",
    "data/splits",
    "models",
    "models/trained_models",
    "notebooks",
    "results",
    "results/images",
    "results/statistics",
    "src",
    "utils",
    "visualization",
]
FILES = [
    "src/__init__.py",
    "utils/__init__.py",
    "models/model_config.json",
    "data/__init__.py",
    "data/data_info.csv",
    "results/results.csv",
    "models/__init__.py",
    "visualization/__init__.py",
    "src/training_config.json",
    "notebooks/prototype_tests.ipynb",
    "config/config.json",
    "data/data_config.json",
]


def main():
    print("Hello from build-project-folder!")
    root_folder = "./" + PROJECT_NAME + "/"
    if os.path.exists(root_folder):
        shutil.rmtree(root_folder)
    os.makedirs(root_folder)

    time.sleep(1)

    for fol in FOLDERS:
        folder_path = root_folder + fol
        # print(folder_path)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    time.sleep(1)

    for file in FILES:
        file_path = root_folder + file
        # print(file_path)
        with open(file_path, "w") as fp:
            pass

    # COPYFILES = [".env", ".gitignore", "README.md"]

    shutil.copy("./config/.env", root_folder + ".env")
    shutil.copy("./config/readme.txt", root_folder + "README.md")
    shutil.copy("./config/gitignore.txt", root_folder + ".gitignore")
    shutil.copy("./config/docker.txt", root_folder + "Dockerfile")
    shutil.copy("./config/mainpy.txt", root_folder + "main.py")
    shutil.copy("./config/mainpy.txt", root_folder + "/data/load_data.py")
    shutil.copy("./config/mainpy.txt", root_folder + "/data/process_data.py")
    shutil.copy("./config/mainpy.txt", root_folder + "/models/models.py")
    shutil.copy("./config/mainpy.txt", root_folder + "/src/data_processing.py")
    shutil.copy("./config/mainpy.txt", root_folder + "/src/model_training.py")
    shutil.copy("./config/mainpy.txt", root_folder + "/src/results_testing.py")
    shutil.copy("./config/mainpy.txt", root_folder + "/utils/decorators.py")
    shutil.copy("./config/mainpy.txt", root_folder + "/utils/activations.py")
    shutil.copy("./config/mainpy.txt", root_folder + "/utils/loss_functions.py")
    shutil.copy("./config/mainpy.txt", root_folder + "visualization/visualize_model.py")
    shutil.copy("./config/mainpy.txt", root_folder + "visualization/visualize_results.py")


if __name__ == "__main__":
    main()
