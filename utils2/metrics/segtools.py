import cv2
import numpy as np
import plotly.graph_objects as go
import scipy.ndimage as ndi
from skimage import measure
from skimage.draw import polygon
from torch import Tensor
from torch.nn import Upsample


def get_outline_points_from_mask(mask):
    contours = measure.find_contours(mask, 0.5)
    outline_points = []
    for contour in contours:
        outline_points.append(contour)
    return np.squeeze(np.array(outline_points))


def get_mask_from_outline_points(outline, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    rr, cc = polygon(outline[:, 0], outline[:, 1], mask.shape)
    mask[rr, cc] = 1
    return ndi.binary_fill_holes(mask).astype(np.uint8)


def apply_clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=16, tileGridSize=(7, 7))
    img_clahe = clahe.apply(img_gray)
    # img_clahe = cv2.equalizeHist(img_gray)
    return img_clahe.astype(np.float32)


def resample_volume(
    volume: np.ndarray,
    output_shape: tuple[int] = (128, 128, 128),
    mode="area",
) -> np.ndarray:
    """
    Resample a 3D volume to a new shape.

    Args:
        volume (np.ndarray): The input 3D volume to resample.
        output_shape (tuple, optional): The desired output shape. Defaults to (128,128,128).
        mode (str, optional): The resampling mode. Defaults to "area".

    Returns:
        np.ndarray: The resampled 3D volume.

    """
    volume = np.squeeze(volume)
    get_dtype = volume.dtype
    data_dowmsample = Upsample(size=output_shape, mode=mode)

    # Rescale image
    volume = np.expand_dims(volume, axis=[0, 1])
    return np.squeeze(data_dowmsample(Tensor(volume)).numpy()).astype(get_dtype)


def plot_3d_volume(
    image_vol: np.ndarray,
    label_vol: np.ndarray,
    title: str = "3D Volume",
    modality: str = "Vega US",
    label_name: str = "Tumor",
) -> None:
    """Plot 3D volume using Plotly."""
    xx, yy, zz = np.indices(image_vol.shape)

    fig = go.Figure()

    fig.add_volume(
        x=xx.flatten(),
        y=yy.max() - yy.flatten(),
        z=zz.flatten(),
        value=image_vol.flatten(),
        isomin=0.05,
        isomax=0.95,
        colorscale="Greys",
        showscale=False,
        # slices_z=dict(show=True, locations=[8]),
        opacity=0.2,  # needs to be small to see through all surfaces
        name=modality,
        surface_count=12,  # needs to be a large number for good volume rendering
    )

    fig.add_volume(
        x=xx.flatten(),
        y=yy.max() - yy.flatten(),
        z=zz.flatten(),
        value=label_vol.flatten(),
        isomin=0.5,
        isomax=1.0,
        colorscale="Electric",
        showscale=False,
        # slices_z=dict(show=True, locations=[8]),
        opacity=0.2,  # needs to be small to see through all surfaces
        name=label_name,
        surface_count=10,
    )

    camera = {
        "up": {"x": 0.0, "y": 1, "z": 0.0},
        # eye={"x": 1.25, "y": 1.25, "z": 1.25},
        "eye": {"x": 1.0, "y": 1, "z": 2},
    }

    fig.update_layout(
        title=title,
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z", "aspectmode": "data"},
        showlegend=True,
        width=350,
        height=400,
        margin={"l": 0, "r": 0, "b": 0, "t": 30},
        scene_camera=camera,
        scene_xaxis_showticklabels=False,
        scene_yaxis_showticklabels=False,
        scene_zaxis_showticklabels=False,
    )

    fig.show()


def segment_center_of_mass(label_mask: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass of a label mask.

    Args:
        label_mask (np.ndarray): The label mask to calculate the center of mass from.

    Returns:
        tuple[int, int, int]: The coordinates of the center of mass.

    """
    if not isinstance(label_mask, np.ndarray):
        msg = f"label_mask must be a numpy array, got {type(label_mask)}"
        raise TypeError(msg)

    if label_mask.ndim not in {2, 3}:
        msg = f"label_mask must be 2D or 3D, got {label_mask.ndim}D"
        raise ValueError(msg)

    if np.sum(label_mask) == 0:
        return (0.5 * np.array(label_mask.shape)).astype(int)

    return np.array(np.where(label_mask)).mean(axis=1).astype(int)
