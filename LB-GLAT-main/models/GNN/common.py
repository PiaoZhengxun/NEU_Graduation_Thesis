# -*- coding: utf-8 -*-


import math
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import Tensor, LongTensor


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def edgeIndex2CooAdj(adj_dim, edge_index : LongTensor):
    """Converts edge_index into sparse matrix coo_matrix"""
    edge_index = edge_index.cpu().numpy()
    edge_num = edge_index.shape[1]
    coo_adj = sp.coo_matrix((np.ones(edge_num), (edge_index[0, :], edge_index[1, :])),
                        shape=(adj_dim, adj_dim),
                        dtype=np.float32)
    return coo_adj


def norm_adj(coo_adj):
    """Random walk normalized critical matrix D-1*A"""
    coo_adj = coo_adj + sp.diags(np.ones(coo_adj.shape[0]))  # Add self-loop
    degree = np.power(coo_adj.A.sum(axis=1), -1)  # 1 / Sum matrix rows
    degree[np.isinf(degree)] = 0.  # If inf, convert to 0
    D = sp.diags(degree)   # degree matrix
    return D.dot(coo_adj)  # Construct D-1*A, asymmetric mode


def mask_h_vector(coo_adj):
    """
    Masked sample vector(0101)
    :param coo_adj: forward graph adjacency matrix
    :return: mask1【!=1: 】 Follow-up the concat operation
             mask2【==1: 】 There is no outdegree, only self-loop, and only DGh is used in the follow-up
    """
    adj = coo_adj.todense()  # numpy.matrix
    mask1 = torch.IntTensor((adj.sum(axis=0) > 1)).permute(1, 0) & torch.IntTensor((adj.sum(axis=1) > 1))
    mask2 = ((torch.IntTensor((adj.sum(axis=0) <= 1)).permute(1, 0) & torch.IntTensor((adj.sum(axis=1) > 1))) |
             (torch.IntTensor((adj.sum(axis=0) <= 1)).permute(1, 0) & torch.IntTensor((adj.sum(axis=1) <= 1))))
    mask3 = torch.IntTensor((adj.sum(axis=0) > 1)).permute(1, 0) & torch.IntTensor((adj.sum(axis=1) <= 1))
    assert torch.equal(mask1 + mask2 + mask3, torch.Tensor(np.ones(mask1.shape)))
    return mask1, mask2, mask3  # (batch, 1)


def mask_h_inf(h, mask, mask1, mask2, mask3):
    """ Replace the rows with all 0 in them with negative infinity """
    mask4 = (torch.Tensor(np.ones(mask1.shape)) - (mask1 + mask2 + mask3)).squeeze(dim=1)[mask]  # 为1的是需要Mask的
    for i in range(h.shape[0]):
        if mask4[i] == 1:
            h[i, :] = torch.full_like(h[i, :], float('-inf'))
    return h


def mask_h_vector2(coo_adj):
    """
    Masked sample vector, required mask (False True False True)
    """
    adj = coo_adj.todense()  # numpy.matrix
    mask1 = torch.tensor((adj.sum(axis=0) != 1), dtype=torch.bool).permute(1, 0).squeeze(dim=1)
    mask2 = torch.tensor((adj.sum(axis=0) == 1), dtype=torch.bool).permute(1, 0).squeeze(dim=1)
    # assert torch.equal(mask1 + mask2, torch.Tensor(np.ones(mask1.shape)))
    return mask1, mask2  # (batch, 1)


class FCsBlock(nn.Module):
    """Linear part"""
    def __init__(self, layer_num, n_features, linears_hidden: Tensor, n_classes, do_bn, dropout, bias):
        super(FCsBlock, self).__init__()
        dims = [n_features] + linears_hidden.tolist()  # dims length = layer_num+1
        self.lins = nn.ModuleList([
            LinearBlock(dims[i], dims[i + 1], do_bn, dropout, bias) for i in range(layer_num)  # Build layer i+1 Linear
        ])
        self.out = nn.Linear(dims[-1], n_classes, bias=bias)  # (batch, n_classes)

    def forward(self, x):
        for lin in self.lins:
            x = lin(x)  # (batch, linears_hidden[i])
        x = self.out(x)  # (batch, n_classes)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, do_bn, dropout, bias=True):
        super(LinearBlock, self).__init__()
        self.lin = nn.Linear(in_features, out_features,  bias=bias)
        self.do_bn = do_bn
        if do_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.lin(x)
        if self.do_bn and x.shape[0] > 1:
            h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        return h


