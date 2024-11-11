# -*- coding: utf-8 -*-


import math
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F


class SpatialGCNLayer(nn.Module):
    """
    Spatial domain GCN, adj = D-1A
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SpatialGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  # in_features, out_features
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)  # (batch, out_features) = (batch, in_features) * (in_features, out_features)
        output = torch.spmm(adj, support)  # (batch, out_features) = (batch, batch) * (batch, out_features)
        if self.bias is not None:
            output = output + self.bias
        return output   # (batch, out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # Randomization parameters
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

