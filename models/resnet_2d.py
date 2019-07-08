import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
import math
from functools import partial

__all__ = ['ResidualCNN2D']

# ResidualBlock2D
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, stride=1):
        super(ResidualBlock2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if self.in_channels != self.out_channels:
            self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels:
            residual = residual.permute([0, 2, 3, 1])
            residual = self.linear(residual)
            residual = residual.permute([0, 3, 1, 2])

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)
        return out

class ResidualCNN2D(nn.Module):
    def __init__(self, input_dim, nb_blocks):
        super(ResidualCNN2D, self).__init__()

        self.residual_blocks = nn.ModuleList([])
        self.residual_blocks.append(ResidualBlock2D(input_dim, 60))
        for i in range(nb_blocks-1):
            self.residual_blocks.append(ResidualBlock2D(60, 60))

    def forward(self, x):
        # x has shape [batch_size, max_len, max_len, input_dim]
        x = x.permute([0, 3, 1, 2])

        for layer in self.residual_blocks:
            x = layer(x)
        outs = x.permute([0, 2, 3, 1])

        return outs
