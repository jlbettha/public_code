import torch
from monai.losses import DiceCELoss
from monai.losses.unified_focal_loss import AsymmetricFocalTverskyLoss
from monai.losses.ssim_loss import SSIMLoss
from torch.nn.modules.loss import _Loss
import warnings
from monai.networks import one_hot
from typing import Callable


class U3PLoss(_Loss):
    def __init__(
        self,
        loss_weights=[0.6, 0.1, 0.3],
        class_weights=torch.Tensor([0.005, 0.995]),
        lambda_iou=1 / 3,
        softmax: bool = False,
        other_act: Callable | None = None,
        to_onehot_y: bool = False,
        include_background: bool = True,
    ):
        super().__init__(),
        self.softmax = softmax
        self.other_act = other_act
        self.to_onehot_y = to_onehot_y
        self.include_background = include_background
        self.loss_weights = loss_weights
        self.class_weights = class_weights
        self.lambda_iou = lambda_iou
        self.lambda_wce = 1 - lambda_iou
        self.ftv_loss = AsymmetricFocalTverskyLoss(
            to_onehot_y=True, reduction="mean", gamma=0.7, delta=0.75, epsilon=1e-7
        )
        self.ssim_loss = SSIMLoss(
            spatial_dims=3,
            data_range=1.0,
            kernel_type="gaussian",
            win_size=7,
            reduction="mean",
            kernel_sigma=1.5,
            k1=0.01,
            k2=0.03,
        )
        self.iou_wce_loss = DiceCELoss(
            reduction="mean",
            jaccard=True,
            weight=class_weights,
            lambda_dice=lambda_iou,
            lambda_ce=1 - lambda_iou,
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for U3PLoss.

        Args:
            input (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed U3PLoss.
        """
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        # if target.shape != input.shape:
        #     raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        # Calculate individual losses
        ftv = self.ftv_loss(input, target)
        ssim = self.ssim_loss(input, one_hot(target, num_classes=n_pred_ch))
        iou_wce = self.iou_wce_loss(input, target)

        # Combine losses with weights
        loss_weights = self.loss_weights
        if len(loss_weights) != 3:
            raise ValueError(
                "loss_weights must be a list of three elements: [ftv_weight, ssim_weight, iou_wce_weight]"
            )

        loss: torch.Tensor = loss_weights[0] * ftv + loss_weights[1] * ssim + loss_weights[2] * iou_wce
        return loss


def main():
    import numpy as np
    from monai.networks.nets import UNet

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=2,
        act="PRELU",
        norm="INSTANCE",
        dropout=0.0,
    )

    # Example inputs and targets

    inputs = torch.Tensor(np.random.choice([0, 1], size=(1, 1, 64, 64, 64), p=[9 / 10, 1 / 10]))
    target = torch.Tensor(np.random.choice([0, 1], size=(1, 1, 64, 64, 64), p=[9 / 10, 1 / 10]))

    # Forward pass
    outputs = model(inputs)
    print(f"{outputs.shape=}, {target.shape=}")
    # exit()

    # Combine losses with weights
    total_loss = U3PLoss()
    loss = total_loss(outputs, target)
    loss.backward()

    # Check gradients to ensure backpropagation worked
    print("Gradients for model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.grad)


if __name__ == "__main__":
    main()
