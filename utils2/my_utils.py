# import os
# import sys
import logging
from collections.abc import Callable

import numpy as np

# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponent,
    # RemoveSmallObjectsd,
    # DivisiblePadd,
    # CropForegroundd,
    # MedianSmoothd,
    # Orientationd,
    # Spacingd,
    # RandCropByPosNegLabeld,
    LoadImaged,
    RandAdjustContrastd,
    RandAffined,
    RandAxisFlipd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandKSpaceSpikeNoised,
    RandScaleIntensityd,
)
from torch import Tensor

# import gc
from torch.nn import Upsample
from torch.optim.optimizer import Optimizer

logger = logging.getLogger("FAdam")


class FAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        betas: tuple[float] = (0.9, 0.999),
        clip: float = 1.0,
        p: float = 0.5,
        eps: float = 1e-8,
        momentum_dtype: torch.dtype = torch.float32,
        fim_dtype: torch.dtype = torch.float32,
        maximize: bool = False,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
            "momentum_dtype": momentum_dtype,
            "fim_dtype": fim_dtype,
            "clip": clip,
            "p": p,
            "maximize": maximize,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        """
        Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                # to fix linter, we do not keep the returned loss for use atm.
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            clip = group["clip"]
            pval = group["p"]
            momentum_dtype = group["momentum_dtype"]
            fim_dtype = group["fim_dtype"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    msg = "FAdam does not support sparse gradients"
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state.setdefault("step", torch.tensor(0.0))
                    state.setdefault(
                        "momentum",
                        torch.zeros_like(p, dtype=momentum_dtype),
                    )
                    state.setdefault("fim", torch.ones_like(p, dtype=fim_dtype))

                # main processing -------------------------

                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                momentum = state["momentum"]
                fim = state["fim"]
                grad = p.grad

                # Apply maximize by flipping gradient sign
                if maximize:
                    grad = -grad

                # begin FAdam algo -------------------------
                # 6 - beta2 bias correction per Section 3.4.4
                curr_beta2 = beta2 * (1 - beta2 ** (step - 1)) / (1 - beta2**step)

                # 7 - update fim
                fim.mul_(curr_beta2).add_(grad * grad, alpha=1 - curr_beta2)

                # 8 - adaptive epsilon
                rms_grad = torch.sqrt(torch.mean(grad * grad))
                curr_eps = eps * max(1, rms_grad)

                # 9 - compute natural gradient
                fim_base = fim**pval + curr_eps  # **(2*pval)

                grad_nat = grad / fim_base

                # 10 - clip the natural gradient
                rms = torch.sqrt(torch.mean(grad_nat**2))
                divisor = max(1, rms)
                divisor = divisor / clip
                grad_nat = grad_nat / divisor

                # 11 - update momentum
                momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

                # 12 - weight decay
                grad_weights = p / fim_base

                # 13 - clip weight decay
                rms = torch.sqrt(torch.mean(grad_weights**2))
                divisor = max(1, rms)
                divisor /= clip
                grad_weights = grad_weights / divisor

                # 14 - compute update
                full_step = momentum + (weight_decay * grad_weights)
                lr_step = lr * full_step

                # 15 - update weights
                p.sub_(lr_step)

        return loss


# @njit
def normalize_image(image):
    """Normalize the image to the range [0, 1]."""
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def plot_3d_volume(
    image_vol: np.ndarray,
    label_vol: np.ndarray,
    title: str = "3D Volume",
    label_name: str = "UNK",
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
        # eye=dict(x=1.25, y=1.25, z=1.25),
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


# @njit
def resample_volume(
    volume: np.ndarray,
    factor: int = 2,
    mode="area",
) -> np.ndarray:
    """
    _summary_

    Args:
        volume (np.ndarray): _description_
        factor (int, optional): _description_. Defaults to 2.
        mode (str, optional): _description_. Defaults to "area".

    Returns:
        np.ndarray: _description_

    """
    volume = np.squeeze(volume)
    output_shape = (
        volume.shape[0] // factor,
        volume.shape[1] // factor,
        volume.shape[2] // factor,
    )
    get_dtype = volume.dtype
    data_dowmsample = Upsample(size=output_shape, mode=mode)

    # Rescale image
    volume = np.expand_dims(volume, axis=[0, 1])
    return np.squeeze(data_dowmsample(Tensor(volume)).numpy()).astype(get_dtype)


def get_transforms(data_mode="clean_test", downsample_factor: float = 1) -> Compose:
    """
    Get data transforms for training or inference.

    Args:
        data_mode (str, optional): Mode of data transformation. Options are 'random_augment_patch', 'random_augment',
        downsample_factor (float, optional): Factor by which to downsample the input images. Defaults to 1.

    Returns:
        Compose: A composition of data transforms.

    """
    match data_mode:
        case "random_augment_patch":
            pass  # TODO: Implement patch-based random augmentations
        case "random_augment":
            return Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    # DivisiblePadd(["image", "label"], 16),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
                    RandScaleIntensityd(keys=["image"], factors=(0.6, 1.1), prob=0.5),
                    RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.5, 2.0)),
                    RandHistogramShiftd(keys=["image"], prob=0.1),
                    RandCoarseDropoutd(
                        keys=["image"],
                        holes=2,
                        spatial_size=np.max([1, 2 // downsample_factor]),
                        dropout_holes=True,
                        max_holes=8,
                        max_spatial_size=10 // downsample_factor,
                        prob=0.2,
                    ),
                    RandCoarseShuffled(
                        keys=["image"],
                        holes=2,
                        spatial_size=np.max([1, 2 // downsample_factor]),
                        max_holes=8,
                        max_spatial_size=10 // downsample_factor,
                        prob=0.2,
                    ),
                    RandGaussianSmoothd(
                        keys=["image"],
                        sigma_x=(0.0, 1.5 / downsample_factor),
                        sigma_y=(0.0, 1.5 / downsample_factor),
                        sigma_z=(0.0, 1.5 / downsample_factor),
                        prob=0.5,
                    ),
                    RandGaussianSharpend(keys=["image"], prob=0.1),
                    RandGibbsNoised(keys=["image"], prob=0.1, alpha=(0.0, 1.0)),
                    RandKSpaceSpikeNoised(keys=["image"], prob=0.1),
                    RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=[0, 1, 2]),
                    RandAxisFlipd(keys=["image", "label"], prob=0.2),
                    RandAffined(
                        keys=["image", "label"],
                        prob=0.8,
                        rotate_range=(np.pi / 15, np.pi / 15, np.pi / 15),
                        scale_range=(0.2, 0.2, 0.2),
                        translate_range=(
                            8 // downsample_factor,
                            16 // downsample_factor,
                            8 // downsample_factor,
                        ),
                        shear_range=(0.05, 0.05, 0.05),
                        mode=("bilinear", "nearest"),
                    ),
                ],
                lazy=True,
            )

        case "clean_test":
            return Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    # DivisiblePadd(["image", "label"], 16),
                    EnsureChannelFirstd(keys=["image", "label"]),
                ],
                lazy=True,
            )

        case "prediction_onehot":
            return Compose([AsDiscrete(keys=["pred"], argmax=True, to_onehot=2)])

        case "label_onehot":
            return Compose([AsDiscrete(to_onehot=2)])

        case "prediction_keep_largest":
            return Compose(
                [
                    AsDiscrete(argmax=True, to_onehot=2),
                    KeepLargestConnectedComponent(is_onehot=True),
                ],
                lazy=True,
            )
        case _:
            msg = f"Unknown data mode: {data_mode}"
            raise ValueError(msg)
    return None


def segment_center_of_mass(label_mask: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass of a label mask.

    Args:
        label_mask (np.ndarray): The label mask to calculate the center of mass from.

    Returns:
        tuple[int, int, int]: The coordinates of the center of mass.

    """
    if not isinstance(label_mask, np.ndarray):
        msg = "label_mask must be a numpy array"
        raise TypeError(msg)

    if label_mask.ndim not in {2, 3}:
        msg = "label_mask must be a 2D or 3D array"
        raise ValueError(msg)

    if np.sum(label_mask) == 0:
        return (0.5 * np.array(label_mask.shape)).astype(int)

    return np.array(np.where(label_mask)).mean(axis=1).astype(int)


def ez_dice_score(ypred, ytrue, threshold=0.5):
    """
    Calculate the Dice score for binary segmentation.

    Args:
        ypred (np.ndarray): Predicted binary mask.
        ytrue (np.ndarray): Ground truth binary mask.
        threshold (float, optional): Threshold to apply to predictions. Defaults to 0.5.

    Returns:
        float: Dice score.

    """
    ypred = ypred.numpy() >= threshold
    ytrue = ytrue.numpy() >= threshold

    intersection = np.sum(ypred * ytrue)
    union = np.sum(ypred) + np.sum(ytrue)

    if union == 0:
        return 1.0

    return 2 * intersection / union


def main():
    """_summary_."""
    rng = np.random.default_rng()
    volume = rng.random((10, 10, 10))
    resampled_volume = resample_volume(volume, factor=2, mode="area")
    print("Original shape:", volume.shape)
    print("Resampled shape:", resampled_volume.shape)

    resampled_volume = normalize_image(resampled_volume)

    # plot_3d_volume(
    #     resampled_volume,
    #     resampled_volume,
    #     title="Resampled Volume",
    #     modality="Test Modality",
    #     label_name="Test Label",
    # )


if __name__ == "__main__":
    main()
