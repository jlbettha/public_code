import yaml
import json


def convert_yaml_to_json(yaml_file_path, json_file_path):
    """
    Converts a YAML file to a JSON file.

    Args:
        yaml_file_path (str): The path to the input YAML file.
        json_file_path (str): The path to the output JSON file.
    """
    try:
        with open(yaml_file_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
    except FileNotFoundError:
        print(f"Error: YAML file not found at {yaml_file_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return

    try:
        with open(json_file_path, 'w') as json_file:
            json.dump(yaml_data, json_file, indent=4)
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        return
    
    print("Successfully converted YAML to JSON")


def convert_json_to_yaml(json_file_path, yaml_file_path):
    """
    Converts a JSON file to a YAML file.

    Args:
        json_file_path (str): The path to the output JSON file.
        yaml_file_path (str): The path to the input YAML file.    
    """
    try:
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return

    try:
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(json_data, yaml_file, sort_keys=False)
    except Exception as e:
        print(f"Error writing YAML file: {e}")
        return
    
    print("Successfully converted JSON to YAML")
    
    
def main():
    # Example usage:
    yaml_file_path = './config.yaml'
    json_file_path = './config.json'
    # Uncomment the line below to convert YAML to JSON
    # convert_yaml_to_json(yaml_file_path, json_file_path)
    
    # Uncomment the line below to convert JSON to YAML
    convert_json_to_yaml(json_file_path, yaml_file_path)
    
    pass

if __name__ == "__main__":
    main()