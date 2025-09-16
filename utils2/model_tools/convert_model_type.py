import os
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort
import torch
from monai.networks.nets import DynUNet


@dataclass
class ModelArgs:
    model_version: str
    model_note: str
    model_target: str
    input_channels: int
    input_size: tuple[int]


def metadata_to_onnx_model(onnx_model_path: str, args: ModelArgs) -> None:
    """_summary_.

    Args:
        onnx_model_path (str): _description_
        args (ModelArgs): _description_
    """
    onnx_model = onnx.load(onnx_model_path)
    m1 = onnx_model.metadata_props.add()
    m1.key = "ModelVersion"
    m1.value = args.model_version

    m3 = onnx_model.metadata_props.add()
    m3.key = "ModelTarget"
    m3.value = args.model_target

    m4 = onnx_model.metadata_props.add()
    m4.key = "SpatialSize"
    m4.value = str(args.input_size)

    m5 = onnx_model.metadata_props.add()
    m5.key = "InputChannels"
    m5.value = str(args.input_channels)

    m5 = onnx_model.metadata_props.add()
    m5.key = "ModelNotes"
    m5.value = args.model_note

    onnx.save(onnx_model, onnx_model_path)


def validate_onnx_model(torch_model: torch.nn.Module, onnx_model_path: str, input_tensor: torch.torch.Tensor) -> None:
    """_summary_.

    Args:
        torch_model (torch.nn.Module): _description_
        onnx_model_path (str): _description_
        input_tensor (torch.torch.Tensor): _description_
    """
    torch_model.eval()  # Run inference with Pytorch model
    with torch.no_grad():
        pytorch_output = torch_model(input_tensor)

    ort_session = ort.InferenceSession(onnx_model_path)  # Run inference with ONNX model
    print(type(ort_session))
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare outputs
    np.testing.assert_allclose(pytorch_output.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("The outputs of the PyTorch model and the ONNX model match!")


def torch_to_onnx(torch_model: torch.nn.Module, onnx_model_path: str, input_tensor: torch.torch.Tensor) -> None:
    """Export a Pytorch model to ONNX format, optimizes it for inference, and saves it to the specified path.

    Args:
        torch_model (torch.nn.Module): _description_
        onnx_model_path (str): _description_
        input_tensor (torch.torch.Tensor): _description_
        args (OnnxModelArgs): _description_
    """
    torch_model.eval()  # Set the model to evaluation mode

    assert onnx_model_path.endswith(".onnx"), 'Filename must end with ".onnx"'

    try:
        onnx_program = torch.onnx.export(
            torch_model,
            input_tensor,
            onnx_model_path,
            export_params=True,
            do_constant_folding=True,
            # opset_version=10,
            input_names=["input"],
            output_names=["output"],
            # dynamic_axes = {'input': {0:'batch_size'}, 'output': {0:'batch_size'}},
            dynamo=True,
        )
        onnx_program.optimize()
        print(f"Saved model to {onnx_model_path}")
    except RuntimeError as e:
        print(f"Failed to export model to ONNX: {e}")


def onnx_to_torch(
    onnx_model: ort.InferenceSession = None,
    torch_model_path: str | None = None,
    input_tensor: torch.torch.Tensor = None,
) -> None:
    """_summary_.

    Args:
        onnx_model (ort.InferenceSession): _description_
        torch_model_path (str): _description_
        input_tensor (torch.torch.Tensor): _description_

    Raises:
        NotImplementedError: _description_
    """
    # TO-DO: Implement this function to convert ONNX model back to PyTorch
    raise NotImplementedError("onnx_to_torch() function is not implemented yet.")


def onnx_to_tflow(
    onnx_model: ort.InferenceSession = None,
    tflow_model_path: str | None = None,
    input_tensor=None,
) -> None:
    """_summary_.

    Args:
        onnx_model (ort.InferenceSession): _description_
        tflow_model_path (str): _description_
        input_tensor (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """
    # TO-DO: Implement this function to convert ONNX model back to Tensorflow/Keras
    raise NotImplementedError("onnx_to_tflow() function is not implemented yet.")


def tflow_to_onnx(tflow_model=None, onnx_model_path: str | None = None, input_tensor=None) -> None:
    """_summary_.

    Args:
        tflow_model (_type_): _description_
        onnx_model_path (str): _description_
        input_tensor (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """
    # TO-DO: Implement this function to convert Tensorflow/Keras model to ONNX
    raise NotImplementedError("tflow_to_onnx() function is not implemented yet.")


def predict_segmentation(tensor, onnx_model_path):
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    input_data = tensor.numpy()
    outputs = ort_session.run(None, {input_name: input_data})
    return outputs


def describe_onnx_model(onnx_model_path):
    ort_session = ort.InferenceSession(onnx_model_path)
    print("ONNX Model Summary:")
    print("Inputs:")
    for inp in ort_session.get_inputs():
        print(f"  Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
    print("Outputs:")
    for out in ort_session.get_outputs():
        print(f"  Name: {out.name}, Shape: {out.shape}, Type: {out.type}")


def main():
    model_folder = "../../models/"
    trained_models_folder = "../../models/trained_models_onnx/"

    args = ModelArgs(
        model_version="1.0",
        input_channels=1,
        input_size=(32, 256, 128),
    )

    torch_model = DynUNet(
        act_name="swish",
        in_channels=1,
        out_channels=2,
        spatial_dims=3,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="batch",
        deep_supervision=False,
        dropout=0.15,
        res_block=True,
    )  # .to(device)

    if os.path.exists(os.path.join(model_folder, "best_metric_model.pth")):
        print("Loading python model.")
        torch_model.load_state_dict(torch.load(os.path.join(model_folder, "best_metric_model.pth"), weights_only=True))

    model_type = torch_model.__class__.__name__
    onnx_model_filename = f"{args.modality}_{args.model_note}_{model_type}_{args.input_size}_model.onnx".replace(
        ", ",
        "-",
    )
    print(onnx_model_filename)
    onnx_model_path = os.path.join(trained_models_folder, onnx_model_filename)

    torch_model.eval()  # only exporting for inference

    # Define model inputs
    batch_size = 1  # random number
    rand_input = torch.randn(batch_size, args.input_channels, *args.input_size, requires_grad=False)

    torch_to_onnx(torch_model, onnx_model_path, rand_input)

    metadata_to_onnx_model(onnx_model_path, args)

    validate_onnx_model(torch_model, onnx_model_path, rand_input)


if __name__ == "__main__":
    main()
