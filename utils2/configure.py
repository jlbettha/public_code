import json
import logging
import os

import yaml


# TODO: Implement the function to build config files in the workspace.
def build_config_files(workspace: str) -> None:
    """
    Configure the workspace by creating the necessary config files if they do not exist.

    Args:
        workspace (str): The path to the workspace directory.
    """
    raise NotImplementedError("This function is not implemented yet.")


def read_yaml_file(filename: str) -> dict:
    """
    filename: full path to a .yaml file.

    out: contents of the .yaml file as a python dictionary
    """
    with open(filename, encoding="utf-8") as f:
        out = yaml.safe_load(f)
    return out


def load_config_from_checkpoint_folder(checkpoint_path: str) -> dict | None:
    """Load the configuration file from a checkpoint folder if it is present."""
    if os.path.isfile(os.path.join(checkpoint_path, "config.json")):
        logging.info("Local config file located.")
        with open(os.path.join(checkpoint_path, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        return config
    return None


def convert_yaml_to_json(yaml_file_path, json_file_path):
    """
    Convert a YAML file to a JSON file.

    Args:
        yaml_file_path (str): The path to the input YAML file.
        json_file_path (str): The path to the output JSON file.
    """
    try:
        with open(yaml_file_path, encoding="utf-8") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
    except FileNotFoundError:
        print(f"Error: YAML file not found at {yaml_file_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return

    try:
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(yaml_data, json_file, indent=4)
    except RuntimeError as e:
        print(f"Error writing JSON file: {e}")
        return

    print("Successfully converted YAML to JSON")


def convert_json_to_yaml(json_file_path, yaml_file_path):
    """
    Convert a JSON file to a YAML file.

    Args:
        json_file_path (str): The path to the output JSON file.
        yaml_file_path (str): The path to the input YAML file.
    """
    try:
        with open(json_file_path, encoding="utf-8") as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return

    try:
        with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(json_data, yaml_file, sort_keys=False)
    except RuntimeError as e:
        print(f"Error writing YAML file: {e}")
        return

    print("Successfully converted JSON to YAML")


def main():
    pass


if __name__ == "__main__":
    main()
