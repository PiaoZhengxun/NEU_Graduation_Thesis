# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import Tensor
from models.GNN.common import *
from models.GNN.GCN.layers import SpatialGCNLayer
from models.GNN.LB_GLAT.layers import LongTermLayerAttention


"""
3 models about GCN:
    1. GCN + Fully Connection Layers
    2. Bi-graph + GCN + Fully Connection Layers
    3. GCN + Long-term Layer Attention + Fully Connection Layers
"""


class GCNFCModel(nn.Module):
    """
    1. GCN + Fully Connection Layers
    """
    def __init__(self, gnn_forward_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, linears_hidden: Tensor, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, device, bias=True):
        super(GCNFCModel, self).__init__()
        self.device = device
        # GNN
        self.DGcn = GCNsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build the GCN of the forward graph

        # Fully Connection Layers
        self.fc = FCsBlock(layer_num=linear_layer_num, n_features=int(gnns_forward_hidden[-1].item()),
                           linears_hidden=linears_hidden, n_classes=n_classes,
                           do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 1【Forward, reverse】

    def forward(self, x, edge_index, mask):
        # Convert the edge_index to a COO sparse matrix and calculate D-1*A
        coo_adj = norm_adj(edgeIndex2CooAdj(x.shape[0], edge_index))
        coo_adj = sparse_mx_to_torch_sparse_tensor(coo_adj).to(self.device)
        # GCN
        h = self.DGcn(x, coo_adj, mask)
        # Fully Connection Layers
        h = self.fc(h[-1])
        return h
    
    
class BiGCNFCModel(nn.Module):
    """
    2. Bi-graph + GCN + Fully Connection Layers
    """
    def __init__(self, gnn_forward_layer_num, gnn_reverse_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, gnns_reverse_hidden: Tensor, linears_hidden: Tensor, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, device, bias=True):
        super(BiGCNFCModel, self).__init__()
        self.device = device
        # GNN
        self.DGcn = GCNsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build the GCN of the forward graph
        self.RDGcn = GCNsBlock(layer_num=gnn_reverse_layer_num, n_features=n_features, gnns_hidden=gnns_reverse_hidden,
                               do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build the GCN of the reverse graph

        # Fully Connection Layers
        self.fc_path1 = FCsBlock(layer_num=linear_layer_num,
                                 n_features=int(gnns_forward_hidden[-1].item()) + int(gnns_reverse_hidden[-1].item()),
                                 linears_hidden=linears_hidden, n_classes=n_classes,
                                 do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 1【Forward, reverse】
        self.fc_path2 = FCsBlock(layer_num=linear_layer_num, n_features=int(gnns_forward_hidden[-1].item()),
                                 linears_hidden=linears_hidden, n_classes=n_classes,
                                 do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 2【Forward】

    def forward(self, x, edge_index, mask):
        # Convert the edge_index to a coo sparse matrix and calculate the D-1*A
        # transpose matrix
        coo_adj = norm_adj(edgeIndex2CooAdj(x.shape[0], edge_index))
        coo_adj_T = sp.coo_matrix(coo_adj.A.T)
        # Get the vector that needs to be masked(1,0,0,0,1,...)
        mask_1, mask_2, mask_3 = mask_h_vector(coo_adj)  # (batch, 1)
        mask_1 = mask_1.to(self.device)[mask]
        mask_2 = mask_2.to(self.device)[mask]
        mask_3 = mask_3.to(self.device)[mask]

        coo_adj = sparse_mx_to_torch_sparse_tensor(coo_adj).to(self.device)
        coo_adj_T = sparse_mx_to_torch_sparse_tensor(coo_adj_T).to(self.device)

        # GCN
        DGh = self.DGcn(x, coo_adj, mask)  # gnn_layer_num * (batch, gnns_hidden[-1])
        RDGh = self.RDGcn(x, coo_adj_T, mask)  # gnn_layer_num * (batch, gnns_hidden[-1])

        # Fully Connection Layers
        # path1
        h_1 = torch.cat((DGh[-1], RDGh[-1]), dim=1)  # (batch, gnns_hidden[-1]*2)
        h_1 = self.fc_path1(h_1)  # (batch, n_classes): not softmax
        # path2
        h_2_1 = self.fc_path2(DGh[-1])  # (batch, n_classes): not softmax
        h_2_2 = self.fc_path2(RDGh[-1])  # (batch, n_classes): not softmax
        # Merge separate batches
        h_1 = torch.mul(mask_1.repeat(1, h_1.shape[1]), h_1)
        h_2_1 = torch.mul(mask_2.repeat(1, h_2_1.shape[1]), h_2_1)  
        h_2_2 = torch.mul(mask_3.repeat(1, h_2_2.shape[1]), h_2_2)  
        h = h_1 + h_2_1 + h_2_2  # (batch, n_classes): not softmax
        return h


class GCNLTLAFCModel(nn.Module):
    """
    3. GCN + Long-term Layer Attention + Fully Connection Layers
    """
    def __init__(self, gnn_forward_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, linears_hidden: Tensor, project_hidden, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, tsf_dim, tsf_mlp_hidden, tsf_depth, tsf_heads, tsf_head_dim, tsf_dropout,
                 vit_emb_dropout, vit_pool, device, bias=True):
        super(GCNLTLAFCModel, self).__init__()
        self.device = device
        # GNN
        self.DGcn = GCNsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build the GCN of the forward graph

        # Long-Term Layer Attention
        self.patches_forward_dim = [n_features] + gnns_forward_hidden.tolist()
        self.DGltla = LongTermLayerAttention(num_patches=gnn_forward_layer_num+1, patches_dim=self.patches_forward_dim,
                                             project_hidden=project_hidden, tsf_dim=tsf_dim,
                                             tsf_mlp_hidden=tsf_mlp_hidden, depth=tsf_depth,
                                             heads=tsf_heads, head_dim=tsf_head_dim, tsf_dropout=tsf_dropout,
                                             vit_emb_dropout=vit_emb_dropout, pool=vit_pool, bias=bias)  # Build LTLA for the forward graph

        # Fully Connection Layers
        self.fc = FCsBlock(layer_num=linear_layer_num, n_features=tsf_dim,
                           linears_hidden=linears_hidden, n_classes=n_classes,
                           do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 1【Forward, reverse】

    def forward(self, x, edge_index, mask):
        # Convert the edge_index to a coo sparse matrix and calculate the D-1*A
        coo_adj = norm_adj(edgeIndex2CooAdj(x.shape[0], edge_index))
        coo_adj = sparse_mx_to_torch_sparse_tensor(coo_adj).to(self.device)
        # GCN
        h = self.DGcn(x, coo_adj, mask)  # gnn_layer_num * (batch, gnns_hidden[-1])
        h.insert(0, x[mask])
        # Long-Term Layer Attention
        h = self.DGltla(h)  # (batch, project_hidden)
        # Fully Connection Layers
        h = self.fc(h)  # (batch, n_classes): not softmax
        return h


class GCNsBlock(nn.Module):
    """GCN part"""
    def __init__(self, layer_num, n_features, gnns_hidden: Tensor, do_bn, dropout, bias):
        super(GCNsBlock, self).__init__()
        dims = [n_features] + gnns_hidden.tolist()  # dims length = layer_num+1
        self.gcns = nn.ModuleList([
            GCNBlock(dims[i], dims[i+1], do_bn, dropout, bias) for i in range(layer_num)  # Build layer i+1 GCN
        ])

    def forward(self, x, coo_adj, mask):
        h = []
        for gcn in self.gcns:
            x = gcn(x, coo_adj)
            h.append(x[mask])
        return h


class GCNBlock(nn.Module):
    def __init__(self, in_features, out_features, do_bn, dropout, bias=True):
        super(GCNBlock, self).__init__()
        self.gcn = SpatialGCNLayer(in_features, out_features, bias=bias)
        self.do_bn = do_bn
        if do_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, coo_adj):
        h = self.gcn(x, coo_adj)
        if self.do_bn:
            h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        return h


# class GCN4Block(nn.Module):
#     """GCN part"""
#     def __init__(self, layer_num, n_features, gnns_hidden: Tensor, bias):
#         super(GCN4Block, self).__init__()
#         assert layer_num == 4
#         self.gcn1 = SpatialGCNLayer(n_features, int(gnns_hidden[0].item()), bias=bias)  # 构建第一层 GCN
#         self.gcn2 = SpatialGCNLayer(int(gnns_hidden[0].item()), int(gnns_hidden[1].item()), bias=bias)  # 构建第二层 GCN
#         self.gcn3 = SpatialGCNLayer(int(gnns_hidden[1].item()), int(gnns_hidden[2].item()), bias=bias)  # 构建第三层 GCN
#         self.gcn4 = SpatialGCNLayer(int(gnns_hidden[2].item()), int(gnns_hidden[3].item()), bias=bias)  # 构建第四层 GCN
#
#     def forward(self, x, coo_adj):
#         h1 = self.gcn1(x, coo_adj)
#         h1 = F.relu(h1)
#         # x = F.dropout(h, self.dropout, training=self.training)
#         h2 = self.gcn2(h1, coo_adj)
#         h2 = F.relu(h2)
#         h3 = self.gcn3(h2, coo_adj)
#         h3 = F.relu(h3)
#         h4 = self.gcn4(h3, coo_adj)
#         h4 = F.relu(h4)
#         return h1, h2, h3, h4
#
#
# class GCN5Block(nn.Module):
#     """GCN part"""
#     def __init__(self, layer_num, n_features, gnns_hidden: Tensor, bias):
#         super(GCN5Block, self).__init__()
#         assert layer_num == 5
#         self.gcn1 = SpatialGCNLayer(n_features, int(gnns_hidden[0].item()), bias=bias)  # 构建第一层 GCN
#         self.gcn2 = SpatialGCNLayer(int(gnns_hidden[0].item()), int(gnns_hidden[1].item()), bias=bias)  # 构建第二层 GCN
#         self.gcn3 = SpatialGCNLayer(int(gnns_hidden[1].item()), int(gnns_hidden[2].item()), bias=bias)  # 构建第三层 GCN
#         self.gcn4 = SpatialGCNLayer(int(gnns_hidden[2].item()), int(gnns_hidden[3].item()), bias=bias)  # 构建第四层 GCN
#         self.gcn5 = SpatialGCNLayer(int(gnns_hidden[3].item()), int(gnns_hidden[4].item()), bias=bias)  # 构建第五层 GCN
#
#     def forward(self, x, coo_adj):
#         h1 = self.gcn1(x, coo_adj)
#         h1 = F.relu(h1)
#         # x = F.dropout(h, self.dropout, training=self.training)
#         h2 = self.gcn2(h1, coo_adj)
#         h2 = F.relu(h2)
#         h3 = self.gcn3(h2, coo_adj)
#         h3 = F.relu(h3)
#         h4 = self.gcn4(h3, coo_adj)
#         h4 = F.relu(h4)
#         h5 = self.gcn5(h4, coo_adj)
#         h5 = F.relu(h5)
#         return h1, h2, h3, h4, h5
