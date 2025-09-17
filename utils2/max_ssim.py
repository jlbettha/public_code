import cv2
import numpy as np
import torch
from metrics import SSIM, SSIM3D, ssim, ssim3d
from torch import optim
from torch.autograd import Variable

SSIM_WINDOW_SIZE_2D = 11  # Must be odd.
SSIM_WINDOW_SIZE_3D = 7  # Must be odd.
SSIM_SIGMA_2D = 1.5
SSIM_SIGMA_3D = 1.5
GOOD_SSIM = 0.95


def example_2d():
    np_img1 = cv2.imread("/home/joseph/workspace/jlb_dev/utils/einstein.jpg")

    img1 = torch.from_numpy(np.rollaxis(np_img1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.rand(img1.size())
    print("img1 shape:", img1.shape)

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)

    ssim_value = ssim(img1, img2).data  # [0]
    print("Initial ssim:", ssim_value)

    ssim_loss = SSIM()

    optimizer = optim.AdamW([img2])

    while ssim_value < GOOD_SSIM:
        optimizer.zero_grad()
        ssim_out = -ssim_loss(img1, img2)
        ssim_value = -ssim_out.data  # [0]
        print(ssim_value)
        ssim_out.backward()
        optimizer.step()


def example_3d():
    img_shape = (64, 64, 64)

    img1 = torch.rand(img_shape).float().unsqueeze(0).unsqueeze(0)
    img2 = torch.rand(img_shape).float().unsqueeze(0).unsqueeze(0)
    print("img1 shape:", img1.shape)
    # sys.exit()

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)

    ssim_value = ssim3d(img1, img2).data  # [0]
    print("Initial ssim:", ssim_value)

    ssim_loss = SSIM3D()

    optimizer = optim.AdamW([img2])

    while ssim_value < GOOD_SSIM:
        optimizer.zero_grad()
        ssim_out = -ssim_loss(img1, img2)
        ssim_value = -ssim_out.data  # [0]
        print(ssim_value)
        ssim_out.backward()
        optimizer.step()


def main():
    # Example usage of the SSIM class
    example_2d()
    example_3d()


if __name__ == "__main__":
    main()
