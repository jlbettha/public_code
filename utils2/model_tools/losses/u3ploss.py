import torch
import torch.nn.functional as F
from torch import nn

from .focalloss import FocalLoss
from .iouloss import IoULoss


def rollwindow(tensor, dim=2, size=25, step=None):
    """Use tensor.unfold to return a tensor containing sliding windows over the source windows don't overlap by default, use the step parameter to change it."""
    if step is None:
        step = size
    result = tensor.unfold(dim, size, step)
    return result.unfold(dim + 1, size, step)


def batch_im_cov(i1, i2):
    assert i1.nelement() == i2.nelement(), "tensors need to have the same size"

    i1flat, i2flat = i1.flatten(start_dim=1).float(), i2.flatten(start_dim=1).float()
    meani1 = torch.mean(i1flat, dim=1)
    meani2 = torch.mean(i2flat, dim=1)

    return torch.mean((i1flat - meani1.unsqueeze(-1)) * (i2flat - meani2.unsqueeze(-1)), dim=1)


def ssim(image_1, image_2, beta=1, gamma=1, **rollwin_kwargs):
    """Compute mean single-scale Structural SIMilarity index between two image-like tensors (NCHW) using a sliding window."""
    # constants to avoid dividing by 0, given in MSSIM paper
    c1 = 1e-4
    c2 = 9e-4

    windows_1 = rollwindow(image_1, **rollwin_kwargs)
    windows_2 = rollwindow(image_2, **rollwin_kwargs)

    total_ssim = torch.zeros(image_1.shape[0], device=image_1.get_device())

    for i in range(windows_1.shape[2]):
        for j in range(windows_1.shape[3]):
            m1, m2 = torch.mean(windows_1[:, :, i, j], dim=(1, 2, 3)), torch.mean(windows_2[:, :, i, j], dim=(1, 2, 3))
            s1, s2 = torch.std(windows_1[:, :, i, j], dim=(1, 2, 3)), torch.std(windows_2[:, :, i, j], dim=(1, 2, 3))

            c = (2 * m1 * m2 + c1) / (m1 * m1 + m2 * m2 + c1)
            s = (2 * batch_im_cov(windows_1[:, :, i, j], windows_2[:, :, i, j]) + c2) / (s1 * s1 + s2 * s2 + c2)

            window_sim = c**beta * s**gamma
            if not torch.any(window_sim.isnan()):
                total_ssim += window_sim

    return total_ssim / (windows_1.shape[2] * windows_2.shape[3] + 0.0001)


def iou_loss(pred: torch.Tensor, targ: torch.Tensor):
    """Intersection over Union loss."""
    pred_flat, targ_flat = pred.flatten(start_dim=1), targ.flatten(start_dim=1)
    intersection = torch.sum(pred_flat * targ_flat)
    union = pred_flat.sum() + targ_flat.sum() - intersection
    return 1 - (intersection + 0.1) / (union + 0.1)


def ms_ssim_loss(pred_scales, target, betas=(1), gammas=(1)):
    """
    Multi-scale similarity loss computed from the product of losses at similar scales and a final target image.

    Typically used to compare the intermediary outputs of the decoder branch and the ground truth mask.
    """
    ssim_product = 1
    for i in range(len(pred_scales)):
        pred = pred_scales[i][:, 1:, ...]
        _beta = betas[min(i, len(betas) - 1)]
        _gamma = gammas[min(i, len(gammas) - 1)]
        ssim_product *= ssim(pred, target[:, 1:, ...], _beta, _gamma)

    return 1 - ssim_product


class U3PLloss(nn.Module):
    def __init__(self, loss_type="focal", aux_weight=0.4, process_input=True):
        super().__init__()
        self.aux_weight = aux_weight
        self.focal_loss = FocalLoss(ignore_index=255, size_average=True)
        if loss_type == "u3p":
            self.iou_loss = IoULoss(process_input=not process_input)
            self.ms_ssim_loss = ms_ssim_loss(process_input=not process_input)
            # self.ms_ssim_loss = SSIMLoss(process_input=not process_input)
            # self.ms_ssim_loss = SSIM()
        elif loss_type != "focal":
            msg = f"Unknown loss type: {loss_type}"
            raise ValueError(msg)
        self.loss_type = loss_type
        self.process_input = process_input

    def forward(self, preds, targets):
        if not isinstance(preds, dict):
            preds = {"final_pred": preds}
        if self.loss_type == "focal":
            return self._forward_focal(preds, targets)
        if self.loss_type == "u3p":
            return self._forward_u3p(preds, targets)
        return None

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
