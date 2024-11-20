# -*- coding: utf-8 -*-

import math
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import Tensor, LongTensor

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def edgeIndex2CooAdj(adj_dim, edge_index: LongTensor):
    edge_index = edge_index.cpu().numpy()
    edge_num = edge_index.shape[1]
    coo_adj = sp.coo_matrix((np.ones(edge_num), (edge_index[0, :], edge_index[1, :])),
                            shape=(adj_dim, adj_dim),
                            dtype=np.float32)
    return coo_adj

def norm_adj(coo_adj):
    coo_adj = coo_adj + sp.diags(np.ones(coo_adj.shape[0]))
    degree = np.power(coo_adj.A.sum(axis=1), -1)
    degree[np.isinf(degree)] = 0.
    D = sp.diags(degree)
    return D.dot(coo_adj)

def mask_h_vector(coo_adj):
    adj = coo_adj.todense()
    mask1 = torch.IntTensor((adj.sum(axis=0) > 1)).permute(1, 0) & torch.IntTensor((adj.sum(axis=1) > 1))
    mask2 = ((torch.IntTensor((adj.sum(axis=0) <= 1)).permute(1, 0) & torch.IntTensor((adj.sum(axis=1) > 1))) |
             (torch.IntTensor((adj.sum(axis=0) <= 1)).permute(1, 0) & torch.IntTensor((adj.sum(axis=1) <= 1))))
    mask3 = torch.IntTensor((adj.sum(axis=0) > 1)).permute(1, 0) & torch.IntTensor((adj.sum(axis=1) <= 1))
    assert torch.equal(mask1 + mask2 + mask3, torch.Tensor(np.ones(mask1.shape)))
    return mask1, mask2, mask3

def mask_h_inf(h, mask, mask1, mask2, mask3):
    mask4 = (torch.Tensor(np.ones(mask1.shape)) - (mask1 + mask2 + mask3)).squeeze(dim=1)[mask]
    for i in range(h.shape[0]):
        if mask4[i] == 1:
            h[i, :] = torch.full_like(h[i, :], float('-inf'))
    return h

def mask_h_vector2(coo_adj):
    adj = coo_adj.todense()
    mask1 = torch.tensor((adj.sum(axis=0) != 1), dtype=torch.bool).permute(1, 0).squeeze(dim=1)
    mask2 = torch.tensor((adj.sum(axis=0) == 1), dtype=torch.bool).permute(1, 0).squeeze(dim=1)
    return mask1, mask2

class FCsBlock(nn.Module):
    def __init__(self, layer_num, n_features, linears_hidden: Tensor, n_classes, do_bn, dropout, bias):
        super(FCsBlock, self).__init__()
        dims = [n_features] + linears_hidden.tolist()
        self.lins = nn.ModuleList([
            LinearBlock(dims[i], dims[i + 1], do_bn, dropout, bias) for i in range(layer_num)
        ])
        self.out = nn.Linear(dims[-1], n_classes, bias=bias)

    def forward(self, x):
        for lin in self.lins:
            x = lin(x)
        x = self.out(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, do_bn, dropout, bias=True):
        super(LinearBlock, self).__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
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