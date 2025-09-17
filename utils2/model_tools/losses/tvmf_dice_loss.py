"""summary: tvMF Dice loss and Adaptive tvMF Dice loss."""

# import numpy as np
import torch
import torch.nn.functional as F

# from scipy.ndimage import zoom
from torch.nn.modules.loss import _Loss


class TvmfDiceLoss(_Loss):
    def __init__(
        self, n_classes: int, kappa: int | None = None, softmax: bool = True, include_background: bool = False
    ):
        super().__init__()
        self.n_classes = n_classes
        kappa_tensor = kappa * torch.ones(n_classes) if kappa is not None else torch.ones(n_classes)
        self.register_buffer("kappa", kappa_tensor)
        self.softmax = softmax
        self.include_background = include_background

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)  # .unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        # smooth = 1.0

        score = F.normalize(score, p=2, dim=[0, 1, 2])
        target = F.normalize(target, p=2, dim=[0, 1, 2])
        cosine = torch.sum(score * target)
        intersect = (1.0 + cosine).div(1.0 + (1.0 - cosine).mul(kappa)) - 1.0
        return (1 - intersect) ** 2.0

    def forward(self, inputs, target):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), f"predict {inputs.size()} & target {target.size()} shape do not match."
        loss = 0.0

        for i in range(self.n_classes):
            if not self.include_background and i == 0:
                continue
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], self.kappa[i])
            loss += tvmf_dice
        if self.include_background:
            return loss / self.n_classes
        return loss / (self.n_classes - 1)  # Exclude background class from averaging if not included


class AdaptiveTvmfDiceLoss(_Loss):
    def __init__(
        self, n_classes: int, kappa: int | None = None, softmax: bool = True, include_background: bool = False
    ):
        super().__init__()
        self.n_classes = n_classes
        kappa_tensor = kappa * torch.ones(n_classes) if kappa is not None else torch.ones(n_classes)
        self.register_buffer("kappa", kappa_tensor)
        self.softmax = softmax
        self.include_background = include_background

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)  # .unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        # smooth = 1.0

        score = F.normalize(score, p=2, dim=[0, 1, 2])
        target = F.normalize(target, p=2, dim=[0, 1, 2])
        cosine = torch.sum(score * target)
        intersect = (1.0 + cosine).div(1.0 + (1.0 - cosine).mul(kappa)) - 1.0
        return (1 - intersect) ** 2.0

    def forward(self, inputs, target):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), f"predict {inputs.size()} & target {target.size()} shape do not match."
        loss = 0.0

        for i in range(self.n_classes):
            if not self.include_background and i == 0:
                continue
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], self.kappa[i])
            loss += tvmf_dice
        if self.include_background:
            return loss / self.n_classes
        return loss / (self.n_classes - 1)  # Exclude background class from averaging if not included
