"""Library for defining losses."""

import monai
import torch


class SoftDiceCeLoss(torch.nn.Module):
    # https://arxiv.org/pdf/2103.10504 Section 3.2
    def __init__(self):
        super().__init__()

    def forward(self, y, g):
        j = y.shape[1]
        ix = torch.numel(g)
        y = torch.softmax(y, 1)
        g = monai.networks.one_hot(g, j)

        # terms 2, and 3 from eqn. (7) in the paper

        # t2
        numerator = torch.dot(g.flatten(), y.flatten())
        denominator = torch.dot(g.flatten(), g.flatten()) + torch.dot(y.flatten(), y.flatten())

        t2 = (2 * numerator) / (j * denominator)

        # t3
        t3 = torch.dot(g.flatten(), torch.log(y.flatten())) / ix

        return 1.0 - t2 - t3
