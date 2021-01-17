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
