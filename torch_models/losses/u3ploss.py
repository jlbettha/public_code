<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
from .focalloss import FocalLoss
from .ms_ssimloss import MS_SSIMLoss, SSIMLoss
from .piqa_ssimloss import SSIM
from .iouloss import IoULoss


class U3PLloss(nn.Module):

    def __init__(self, loss_type="focal", aux_weight=0.4, process_input=True):
        super().__init__()
        self.aux_weight = aux_weight
        self.focal_loss = FocalLoss(ignore_index=255, size_average=True)
        if loss_type == "u3p":
            self.iou_loss = IoULoss(process_input=not process_input)
            self.ms_ssim_loss = MS_SSIMLoss(process_input=not process_input)
            # self.ms_ssim_loss = SSIMLoss(process_input=not process_input)
            # self.ms_ssim_loss = SSIM()
        elif loss_type != "focal":
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss_type = loss_type
        self.process_input = process_input

    def forward(self, preds, targets):
        if not isinstance(preds, dict):
            preds = {"final_pred": preds}
        if self.loss_type == "focal":
            return self._forward_focal(preds, targets)
        elif self.loss_type == "u3p":
            return self._forward_u3p(preds, targets)

    def _forward_focal(self, preds, targets):
        loss_dict = {}
        loss = self.focal_loss(preds["final_pred"], targets)
        loss_dict["head_focal_loss"] = loss.detach().item()  # for logging
        num_aux, aux_loss = 0, 0.0

        for key in preds:
            if "aux" in key:
                num_aux += 1
                aux_loss += self.focal_loss(preds[key], targets)
        if num_aux > 0:
            aux_loss = aux_loss / num_aux * self.aux_weight
            loss_dict["aux_focal_loss"] = aux_loss.detach().item()
            loss += aux_loss
            loss_dict["total_loss"] = loss.detach().item()

        return loss, loss_dict

    def onehot_softmax(self, pred, target: torch.Tensor, process_target=True):
        _, num_classes, h, w = pred.shape
        pred = F.softmax(pred, dim=1)

        if process_target:
            target = torch.clamp(target, 0, num_classes)
            target = (
                F.one_hot(target, num_classes=num_classes + 1)[..., :num_classes]
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(pred.dtype)
            )
        return pred, target

    def _forward_u3p(self, preds, targets):
        r"""Full-scale Deep Supervision"""

        loss, loss_dict = self._forward_focal(preds, targets)
        if self.process_input:
            final_pred, targets = self.onehot_softmax(preds["final_pred"], targets)
        iou_loss = self.iou_loss(final_pred, targets)
        msssim_loss = self.ms_ssim_loss(final_pred, targets)
        loss = loss + iou_loss + msssim_loss
        loss_dict["head_iou_loss"] = iou_loss.detach().item()
        loss_dict["head_msssim_loss"] = msssim_loss.detach().item()

        num_aux, aux_iou_loss, aux_msssim_loss = 0, 0.0, 0.0
        for key in preds:
            if "aux" in key:
                num_aux += 1
                if self.process_input:
                    preds[key], targets = self.onehot_softmax(preds[key], targets, process_target=False)
                aux_iou_loss += self.iou_loss(preds[key], targets)
                aux_msssim_loss += self.ms_ssim_loss(preds[key], targets)
        if num_aux > 0:
            aux_iou_loss /= num_aux
            aux_msssim_loss /= num_aux
            loss_dict["aux_iou_loss"] = aux_iou_loss.detach().item()
            loss_dict["aux_msssim_loss"] = aux_msssim_loss.detach().item()
            loss += (aux_iou_loss + aux_msssim_loss) * self.aux_weight
            loss_dict["total_loss"] = loss.detach().item()

        return loss, loss_dict


def build_u3p_loss(
    loss_type="focal",
    aux_weight=0.4,
) -> U3PLloss:
    return U3PLloss(loss_type, aux_weight)
=======
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
>>>>>>> 87794b25f5624ab526d51cdff2e92d4b239142aa
