"""Revvity (2025) Utility functions for loading and managing MONAI models, loss functions, and optimizers."""

import os
import json
from types import NoneType
import torch
import numpy as np
import importlib
import inspect
from utils.configure import convert_json_to_yaml


def build_monai_model(model_name: str = "DynUNet", print_summary: bool = False) -> torch.nn.Module:
    """_summary_

    Args:
        model_name (str, optional): _description_. Defaults to "DynUNet".

    Returns:
        torch.nn.Module: _description_
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    MonaiModel = load_module_class("monai.networks.nets." + model_name)
    my_params = _load_model_params(model_name=model_name)
    model = MonaiModel(**my_params).to(device)

    if print_summary:
        from torchsummary import summary

        summary(model, input_size=(1, 32, 256, 128), device=device.type)

    return model


def set_optimizer(optimizer_name: str = "AdamW", opt_params: dict = None) -> torch.optim.Optimizer:
    optimizer_params = _load_optimizer_params(optimizer_name=optimizer_name)
    Optimizer = load_module_class("torch.optim." + optimizer_name)
    optimizer = Optimizer(params=opt_params, **optimizer_params)
    return optimizer


def set_loss_function(loss_name: str = "GeneralizedDiceLoss", loss_params: dict = None) -> torch.nn.modules.loss._Loss:
    """Set the loss function for the model.

    Args:
        loss_name (str, optional): _description_. Defaults to "GeneralizedDiceLoss".
        loss_kwargs (dict, optional): _description_. Defaults to None.
    """
    loss_params = _load_loss_params(loss_name=loss_name) if loss_params is None else loss_params
    LossFunction = load_module_class("monai.losses." + loss_name)
    loss_function = LossFunction(**loss_params)
    return loss_function


def load_module_class(full_class_string: str):
    """
    dynamically load a class from a string
    """

    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]

    module = importlib.import_module(module_path)
    # Finally, we retrieve the Class
    return getattr(module, class_str)


def load_model_checkpoint(torch_model: torch.nn.Module, model_path: str) -> None:
    if os.path.exists(model_path):
        print("Loading best model from previous training run...")
        torch_model.load_state_dict(torch.load(os.path.join(model_path), weights_only=True))
    else:
        print("No existing model found. Starting from scratch...")


def save_model_checkpoint(
    model: torch.nn.Module, savepath: str = "../../models/torch_model_checkpoints", test_split=None
):
    """
    Save model weights to disk at the given path.

    'model' - network
    'savepath' - path (.pt) to save model weights to
    """
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
    torch.save(model.state_dict(), savepath)
    if test_split is not None:
        np.save(os.path.splitext(savepath)[0] + ".testidx.npy", test_split)


def _create_model_param_config(
    model_names: list[str] = ["DynUNet", "UNETR", "VNet", "SegResNet", "SwinUNETR"], my_params: dict = None, **kwargs
) -> None:
    """Create a JSON (and YAML) file with the default parameters for MONAI models.

    Args:
        model_names (list[str], optional): _description_.
                Defaults to ["DynUNet", "UNETR", "VNet", "SegResNet", "SwinUNETR"].
        my_params (dict, optional): _description_. Defaults to None.
    """
    json_types = [int, float, str, bool, list, tuple, dict, NoneType()]

    model_dict = {}
    for model_name in model_names:
        MonaiModel = load_module_class("monai.networks.nets." + model_name)
        param_dict = {}

        for param in inspect.signature(MonaiModel).parameters.values():
            if type(param.default) not in json_types:
                param_val = str(param.default)
            else:
                param_val = param.default

            if my_params and param.name in my_params:
                param_val = my_params[param.name]

            if param.name in kwargs:
                param_val = kwargs[param.name]

            param_dict[param.name] = param_val

        model_dict[model_name] = param_dict

    with open("../../config/monai_model_params.json", "w") as f:
        json.dump(model_dict, f, indent=4)

    convert_json_to_yaml(
        json_file_path="../../config/monai_model_params.json",
        yaml_file_path="../../config/monai_model_params.yaml",
    )


def _create_optimizer_params_config(
    optimizer_names: list[str] = ["Adam", "AdamW", "SGD"], my_params: dict = None, **kwargs
) -> None:
    """Create a JSON (and YAML) file with the default parameters for optimizers.

    Args:
        model_names (list[str], optional): _description_.
                Defaults to ["Adam", "AdamW", "SGD"].
        my_params (dict, optional): _description_. Defaults to None.
    """
    json_types = [int, float, str, bool, list, tuple, dict, NoneType()]

    optimizer_dict = {}
    for optimizer_name in optimizer_names:
        Optimizer = load_module_class("torch.optim." + optimizer_name)
        param_dict = {}

        for param in inspect.signature(Optimizer).parameters.values():
            if param.name == "params":
                continue

            if type(param.default) not in json_types:
                param_val = str(param.default)
            else:
                param_val = param.default

            if my_params and param.name in my_params:
                param_val = my_params[param.name]

            if param.name in kwargs:
                param_val = kwargs[param.name]

            param_dict[param.name] = param_val

        optimizer_dict[optimizer_name] = param_dict

    with open("../../config/optimizer_params.json", "w") as f:
        json.dump(optimizer_dict, f, indent=4)

    convert_json_to_yaml(
        json_file_path="../../config/optimizer_params.json",
        yaml_file_path="../../config/optimizer_params.yaml",
    )


def _create_loss_params_config(
    loss_names: list[str] = [
        "GeneralizedDiceLoss",
        "DiceCELoss",
        "DiceFocalLoss",
        "GeneralizedDiceFocalLoss",
        "unified_focal_loss.AsymmetricFocalTverskyLoss",
        "FocalLoss",
        "TverskyLoss",
        "BarlowTwinsLoss",
        "ContrastiveLoss",
        "HausdorffDTLoss",
    ],
    my_params: dict = None,
    **kwargs,
) -> None:
    """Create a JSON (and YAML) file with the default parameters for loss functions.

    Args:
        loss_names (list[str], optional): _description_.
        my_params (dict, optional): _description_. Defaults to None.
    """
    json_types = [int, float, str, bool, list, tuple, dict, NoneType()]

    loss_dict = {}
    for loss_name in loss_names:

        LossFunction = load_module_class("monai.losses." + loss_name)
        param_dict = {}
        for param in inspect.signature(LossFunction).parameters.values():
            if type(param.default) not in json_types:
                param_val = str(param.default)
            else:
                param_val = param.default

            if my_params and param.name in my_params:
                param_val = my_params[param.name]

            if param.name in kwargs:
                param_val = kwargs[param.name]

            param_dict[param.name] = param_val

        loss_dict[loss_name] = param_dict

    with open("../../config/loss_params.json", "w") as f:
        json.dump(loss_dict, f, indent=4)

    convert_json_to_yaml(
        json_file_path="../../config/loss_params.json",
        yaml_file_path="../../config/loss_params.yaml",
    )


def _create_transform_params_config(
    transform_names: list[str],
    my_params: dict = None,
    **kwargs,
) -> None:
    """Create a JSON (and YAML) file with the default parameters for transforms.

    Args:
        loss_names (list[str]): _description_.
        my_params (dict, optional): _description_. Defaults to None.
    """
    json_types = [int, float, str, bool, list, tuple, dict, NoneType()]

    transform_dict = {}
    for transform_name in transform_names:

        TransformFunction = load_module_class("monai.transforms." + transform_name)
        param_dict = {}
        for param in inspect.signature(TransformFunction).parameters.values():
            if type(param.default) not in json_types:
                param_val = str(param.default)
            else:
                param_val = param.default

            if my_params and param.name in my_params:
                param_val = my_params[param.name]

            if param.name in kwargs:
                param_val = kwargs[param.name]

            param_dict[param.name] = param_val

        transform_dict[transform_name] = param_dict

    with open("../../config/transform_params.json", "w") as f:
        json.dump(transform_dict, f, indent=4)

    convert_json_to_yaml(
        json_file_path="../../config/transform_params.json",
        yaml_file_path="../../config/transform_params.yaml",
    )


def _load_model_params(model_name: str = "DynUNet", json_path: str = "../../config/monai_model_params.json") -> dict:
    """Load model parameters from a JSON file.

    Args:
        model_name (str, optional): _description_. Defaults to "DynUNet".
        json_path (str, optional): _description_. Defaults to "../../config/monai_model_params.json".

    Raises:
        FileNotFoundError: _description_

    Returns:
        dict: _description_
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    with open(json_path, "r") as f:
        all_params = json.load(f)

    for k, v in all_params.items():
        # print(k, v)
        if k.lower() == model_name.lower():
            model_params = v
            if "filters" in model_params and model_params["filters"] == "None":
                model_params["filters"] = None

    return model_params


def _load_optimizer_params(
    optimizer_name: str = "AdamW", json_path: str = "../../config/optimizer_params.json"
) -> dict:
    """Load optimizer parameters from a JSON file.

    Args:
        optimizer_name (str, optional): _description_. Defaults to "AdamW".
        json_path (str, optional): _description_. Defaults to "../../config/optimizer_params.json".

    Raises:
        FileNotFoundError: _description_

    Returns:
        dict: _description_
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    with open(json_path, "r") as f:
        all_params = json.load(f)

    for k, v in all_params.items():
        if k.lower() == optimizer_name.lower():
            optimizer_params = v
            for key, value in optimizer_params.items():
                if value == "None":
                    optimizer_params[key] = None

    return optimizer_params


def _load_loss_params(
    loss_name: str = "GeneralizedDiceLoss", json_path: str = "../../config/loss_params.json"
) -> dict:
    """Load loss parameters from a JSON file.

    Args:
        loss_name (str, optional): _description_. Defaults to "GeneralizedDiceLoss".
        json_path (str, optional): _description_. Defaults to "../../config/loss_params.json".

    Raises:
        FileNotFoundError: _description_

    Returns:
        dict: _description_
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    with open(json_path, "r") as f:
        all_params = json.load(f)

    for k, v in all_params.items():
        if k.lower() == loss_name.lower():
            loss_params = v
            for key, value in loss_params.items():
                if value == "None":
                    loss_params[key] = None

    return loss_params


def main():

    # transform_params = {
    #     "keys": ["image", "label"],
    #     "spatial_size": [32, 256, 128],
    #     "mode": ["bilinear", "nearest"],
    #     "prob": 0.1,
    #     "spatial_axis": [0, 1, 2],
    #     "gamma": [0.5, 1.5],
    #     "mean": 0.0,
    #     "std": [0.01, 0.1],
    #     "sigma_x": [0.5, 1.5],
    #     "sigma_y": [0.5, 1.5],
    #     "sigma_z": [0.5, 1.5],
    #     "shift_range": [-0.1, 0.1],
    #     "label_key": "label",
    #     "num_samples": 1,
    #     "allow_smaller:": False,
    #     "lazy": True,
    #     "num_threads": 4,
    #     "interpolation": "bicubic",
    #     "device": "cuda",
    #     "dtype": "float32",
    #     "roi_size": [32, 256, 128],
    #     "size_divisibility": compute_divisibility([2, 2, 2]),
    # }

    # with open("transform_names.txt", "r") as f:
    #     transforms = [line.rstrip() for line in f]

    # _create_transform_params_config(transform_names=transforms, my_params=transform_params)
    # # print(transforms)
    # exit()

    model = build_monai_model(model_name="DynUNet", print_summary=True)
    loss_fn = set_loss_function(loss_name="GeneralizedDiceLoss")
    optimizer = set_optimizer(optimizer_name="AdamW", opt_params=model.parameters())

    print(loss_fn)
    print(optimizer)


if __name__ == "__main__":
    main()
