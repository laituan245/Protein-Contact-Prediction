import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
import math
from functools import partial

__all__ = ['ResidualCNN1D']

# 1D Residual CNN
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, stride=1):
        super(ResidualBlock1D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if self.in_channels != self.out_channels:
            self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.in_channels != self.out_channels:
            residual = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        out += residual
        out = self.relu2(out)
        return out

class ResidualCNN1D(nn.Module):
    def __init__(self, input_dim):
        super(ResidualCNN1D, self).__init__()

        kernel_size = 17; padding = 8
        self.residual_blocks = nn.ModuleList([])
        self.residual_blocks.append(ResidualBlock1D(input_dim, 60, kernel_size, padding))
        self.residual_blocks.append(ResidualBlock1D(60, 60, kernel_size, padding))
        self.residual_blocks.append(ResidualBlock1D(60, 60, kernel_size, padding))

    def forward(self, x):
        # x has shape [batch_size, max_len, input_dim]
        x = x.permute([0, 2, 1])

        for layer in self.residual_blocks:
            x = layer(x)
        outs = x.permute([0, 2, 1])

        return outs
