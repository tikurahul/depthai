from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn


class Grayscale(nn.Module):
    """
    Converts a batch of RGB -> Gray Scale
    1, H, W, C
    """

    def __init__(self, shape: Tuple[int, int, int, int], dtype=torch.float):
        super(Grayscale, self).__init__()
        self.shape = shape
        self.dtype = dtype

    def forward(self, x):
        # A simplified version of GrayScale.
        # 1, H, W, C
        y_b = x[0, :, :, 0]
        y_g = x[0, :, :, 1]
        y_r = x[0, :, :, 2]
        g_ = 0.3 * y_r + 0.59 * y_g + 0.11 * y_b
        g_ = torch.unsqueeze(g_, dim=2)
        g_ = torch.unsqueeze(g_, dim=0)
        return g_


class GaussianBlur(nn.Module):
    def __init__(self, shape: Tuple[int, int, int, int], dtype=torch.float):
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        kernel = torch.tensor([
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1],

        ], dtype=torch.float)
        self.kernel = kernel / 273
        # Define Layers
        _, _, _, c = self.shape
        self.blur = nn.Conv2d(
            in_channels = 1,
            out_channels = 1,
            padding = 2,
            kernel_size=5,
            bias=False
        )
        self.blur.requires_grad_(False)
        # Initialize Weights
        self.blur.weight[:, :, :, :] = self.kernel

    def forward(self, x):
        b = x[0, :, :, 0]
        g = x[0, :, :, 1]
        r = x[0, :, :, 2]

        b_ = b.type(torch.float)
        b_ = torch.unsqueeze(b_, dim=0)
        b_ = torch.unsqueeze(b_, dim=0)

        g_ = g.type(torch.float)
        g_ = torch.unsqueeze(g_, dim=0)
        g_ = torch.unsqueeze(g_, dim=0)

        r_ = r.type(torch.float)
        r_ = torch.unsqueeze(r_, dim=0)
        r_ = torch.unsqueeze(r_, dim=0)

        b_ = self.blur(b_)[0][0]
        g_ = self.blur(g_)[0][0]
        r_ = self.blur(r_)[0][0]

        output = torch.stack([b_, g_, r_], dim=-1)
        output = torch.unsqueeze(output, dim=0)
        return output


class GrayscaleModel(nn.Module):
    def __init__(self, shape: Tuple[int, int, int, int], dtype=torch.float):
        super(GrayscaleModel, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.model = nn.Sequential(OrderedDict([
            ('grayscale', Grayscale(self.shape, self.dtype))
        ]))

    def forward(self, x):
        return self.model(x)
