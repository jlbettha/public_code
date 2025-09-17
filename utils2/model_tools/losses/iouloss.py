import torch
import torch.nn.functional as F
from torch import nn


def binary_iou_loss(pred, target):
    iand = torch.sum(pred * target, dim=1)
    ior = torch.sum(pred, dim=1) + torch.sum(target, dim=1) - iand
    iou = 1 - iand.sum() / ior.sum()
    return iou.sum()


class IoULoss(nn.Module):
    def __init__(self, process_input=True) -> None:
        super().__init__()
        self.process_input = process_input

    def forward(self, pred, target):
        num_classes = pred.shape[1]

        if self.process_input:
            pred = F.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        total_loss = 0
        for i in range(num_classes):
            loss = binary_iou_loss(pred[:, i], target[:, i])
            total_loss += loss
        return total_loss / num_classes


if __name__ == "__main__":
    pred = torch.randn((6, 21, 50, 50))
    target = torch.randint(0, 21, (6, 50, 50))
    iou = IoULoss()
    rst = iou(pred, target)
    print(rst)
