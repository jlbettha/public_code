import torch
import torch.nn.functional as F


def iou(y_true, y_pred, smooth=1):
    """
    Calculate intersection over union (IoU) between images.
    Input shape should be Batch x #Classes x Height x Width (BxNxHxW).
    Using Mean as reduction type for batch values.
    """
    intersection = torch.sum(torch.abs(y_true * y_pred), dim=(1, 2, 3, 4))
    union = torch.sum(y_true, dim=(1, 2, 3, 4)) + torch.sum(y_pred, dim=(1, 2, 3, 4)) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth))
    return iou


def iou_loss(y_true, y_pred):
    """
    Jaccard / IoU loss
    """
    return 1 - iou(y_true, y_pred)


def focal_loss(y_true, y_pred):
    """
    Focal loss
    """
    gamma = 2.0
    alpha = 4.0
    epsilon = 1.0e-9

    y_true = y_true.float()
    y_pred = y_pred.float()

    model_out = y_pred + epsilon
    ce = y_true * -torch.log(model_out)
    weight = y_true * torch.pow(1 - model_out, gamma)
    fl = alpha * weight * ce
    reduced_fl = torch.max(fl, dim=1)[0]
    return torch.mean(reduced_fl)


def gaussian_kernel(size, sigma):
    """
    Generate a 2D Gaussian kernel
    """
    coords = torch.arange(size).float() - size // 2
    coords = coords.unsqueeze(0).repeat(size, 1)
    gaussian = torch.exp(-(coords**2 + coords.t() ** 2) / (2 * sigma**2))
    return gaussian / gaussian.sum()


def gaussian_kernel_3d(size, sigma):
    """
    Generate a 3D Gaussian kernel
    """
    coords = torch.arange(size).float() - size // 2
    coords = coords.unsqueeze(0).unsqueeze(0).repeat(size, size, 1)
    gaussian = torch.exp(-(coords**2 + coords.permute(1, 0, 2) ** 2 + coords.permute(2, 0, 1) ** 2) / (2 * sigma**2))
    return gaussian / gaussian.sum()


class SSIM(torch.nn.Module):
    def __init__(self, window_size=5, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = gaussian_kernel_3d(window_size, 1.5).unsqueeze(0).unsqueeze(0)

    def forward(self, img1, img2):
        img1 = img1[:, 1, :, :, :].unsqueeze(1)
        (_, channel, _, _, _) = img1.size()
        # print(f"SSIM: img1 shape: {img1.shape}, img2 shape: {img2.shape}, channel: {channel}")
        if channel == self.channel and self.window.data.type() == img1.data.type():
            # exit()
            window = self.window
        else:
            window = gaussian_kernel_3d(self.window_size, 1.5).unsqueeze(0).unsqueeze(0)
        # window = window.repeat(channel, 1, 1, 1, 1).to(img1.device)
        window = window.to(img1.device).type_as(img1)
        self.window = window
        self.channel = channel

        mu1 = F.conv3d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv3d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index loss.
    Input shape should be Batch x #Classes x Height x Width (BxCxHxW).
    """
    ssim = SSIM()
    return ssim(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1.0e-9):
    """
    Calculate dice coefficient.
    Input shape should be Batch x #Classes x Height x Width (BxNxHxW).
    Using Mean as reduction type for batch values.
    """
    intersection = torch.sum(y_true * y_pred, dim=(2, 3))
    union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3))
    return torch.mean((2.0 * intersection + smooth) / (union + smooth))
