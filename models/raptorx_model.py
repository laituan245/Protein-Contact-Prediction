import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
import math
from functools import partial
from models.resnet_1d import ResidualCNN1D
from models.resnet_2d import ResidualCNN2D

def _outer_concatenation(features_1d):
    max_len = features_1d.size()[1]
    outs = [[None for j in range(max_len)] for i in range(max_len)]
    for i in range(max_len):
        for j in range(max_len):
            mid = int((i + j) / 2.0)
            v_i, v_j, v_mid = features_1d[:,i,:], features_1d[:,j,:], features_1d[:,mid,:]
            outs[i][j] = torch.cat([v_i, v_j, v_mid], -1)
    stacked_outs = []
    for i in range(max_len):
        stacked_outs.append(torch.stack([outs[i][j] for j in range(max_len)], dim=1))
    outs = torch.stack(stacked_outs, dim=1)
    return outs

class RaptorXModel(nn.Module):
    def __init__(self, feature_1d_dim, feature_2d_dim):
        super(RaptorXModel, self).__init__()

        self.resnet_1d = ResidualCNN1D(feature_1d_dim)
        self.resnet_2d = ResidualCNN2D(feature_2d_dim + 3 * 60, nb_blocks = 10)
        self.fc_1 = nn.Linear(60, 100)
        self.fc_2 = nn.Linear(100, 2)

    def forward(self, _1d_features, _2d_feature):
        # _1d_features should have shape [batch_size, max_len, feature_1d_dim]
        # _2d_features should have shape [batch_size, max_len, max_len, feature_2d_dim]

        # 1D Residual Network
        feats_1d = self.resnet_1d(_1d_features)

        # Outer concatenation
        outs_1d = _outer_concatenation(feats_1d)

        # 2D Residual Network
        inputs_2d = torch.cat([_2d_feature, outs_1d], -1)
        outs_2d = self.resnet_2d(inputs_2d)

        # preds
        preds = self.fc_2(torch.relu(self.fc_1(outs_2d)))

        return preds
