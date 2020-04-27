import torch
from torch import nn
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import glob
import numpy as np
import math


def max_outputs(outputs):
    # outputs_i with prediction probability map, (2, 2, 1250, 1250) -> (batch, classes, dim, dim)
    # max select from classes logits
    # at most 3 predict maps from 3 models now.
    if len(outputs) == 3:
        probability_map = torch.max(torch.max(outputs[0], outputs[1]), outputs[2])
    else:
        probability_map = torch.max(outputs[0], outputs[1])
    return probability_map


def mean_outputs(outputs):
    if len(outputs) == 3:
        probability_map = (outputs[0] + outputs[1] + outputs[2]) / 3
    else:
        probability_map = (outputs[0] + outputs[1]) / 2

    return probability_map


def smooth_gaussian(probability_map, device='cuda'):
    batch = probability_map.shape[0]
    in_channels = probability_map.shape[1]
    out_channels = probability_map.shape[1]
    kernel_size = 5
    sigma = 1
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(batch, in_channels, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=2, bias=False)

    # gaussian_filter.weight = torch.nn.Parameter(gaussian_filter).
    #
    # gaussian_filter.weight = gaussian_filter.weight.to(device)
    gaussian_filter.weight.data = torch.nn.Parameter(gaussian_kernel).to(device)
    gaussian_filter.weight.requires_grad = False

    # filter = gaussian_filter()
    probability_map = gaussian_filter(probability_map)

    return probability_map


if __name__ == "__main__":
    # A full forward pass
    # im1 = torch.randn(1, 2, 4, 4)
    # im2 = torch.randn(1, 2, 4, 4)
    # im3 = torch.randn(1, 2, 4, 4)
    # prob = max_outputs(im1, im2, im3)
    # print(prob, prob.shape)
    # pred_class = torch.argmax(prob, dim=1).float()
    # print(pred_class, pred_class.shape)
    #
    # img_as_np = pred_class.numpy()
    # img_as_np[img_as_np >= 0.5] = 1
    # img_as_np[img_as_np < 0.5] = 0
    # img_as_np = (img_as_np) * 255
    # print(img_as_np)
    #
    # img_as_np = img_as_np.astype('uint8')
    # print(img_as_np)
    # img_cont = Image.fromarray(img_as_np)
    # img_cont.show()

    image_name = glob.glob("history1_3_5/UNET/result_images3/epoch_30/0.png")
    image = Image.open(image_name[0])
    image.show()
    img_as_np = np.asarray(image)
    img_as_np = torch.Tensor(img_as_np).unsqueeze(0).unsqueeze(0)
    print(img_as_np, img_as_np.shape)
    img_as_np = smooth_gaussian(img_as_np)
    img_as_np = img_as_np.numpy().astype(np.uint8)
    print(img_as_np)
    img1 = Image.fromarray(img_as_np.squeeze(0).squeeze(0))
    img1.show()
