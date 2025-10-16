import json
import pathlib

import deepdiff


def compare_model_files(file1_path, file2_path):
    """
    Compare two model files and return the differences.

    Args:
        file1_path (str): Path to the first model file.
        file2_path (str): Path to the second model file.

    Returns:
        dict: A dictionary containing the differences between the two files.

    """
    # Load the model files
    with open(file1_path) as file1:
        model1 = json.load(file1)

    with open(file2_path) as file2:
        model2 = json.load(file2)

    # Compare the models using deepdiff
    return deepdiff.DeepDiff(model1, model2, ignore_order=True)


def main():
    model1_runcode = "exo2tn8o"
    model2_runcode = "x6fqjo1u"

    # Define the paths to the model files
    model1_path = pathlib.Path(f"/home/joseph/workspace/model_checkpoints/{model1_runcode}/config.json")
    model2_path = pathlib.Path(f"/home/joseph/workspace/model_checkpoints/{model2_runcode}/config.json")

    # Compare the model files
    differences = compare_model_files(model1_path, model2_path)
    differences = deepdiff.DeepDiff("early_stopping.py", "activations.py")
    # Print the differences
    if differences:
        print("Differences found between the model files:")

        json_diff = differences.to_json(indent=4)
        print(json_diff)
    else:
        print("No differences found between the model files.")


if __name__ == "__main__":
    main()
