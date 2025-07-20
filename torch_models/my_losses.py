import torch
from torch import nn


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# TO-DO:
# class FocalLoss(nn.Module):
#     def forward(self, inputs, targets, smooth=1):
#         inputs = torch.sigmoid(inputs)
#         intersection = (inputs * targets).sum()
#         dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         return 1 - dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        return 1 - tversky


# TO-DO:
# class FocalTverskyLoss(nn.Module):
#     def __init__(self, alpha=0.5, beta=0.5, gamma=2):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#     def forward(self, inputs, targets, smooth=1):
#         inputs = torch.sigmoid(inputs)
#         intersection = (inputs * targets).sum()
#         dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         return 1 - dice


# TO-DO:
# class U3-Custom(nn.Module):
#     def forward(self, inputs, targets, smooth=1):
#         inputs = torch.sigmoid(inputs)
#         intersection = (inputs * targets).sum()
#         dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         return 1 - dice
