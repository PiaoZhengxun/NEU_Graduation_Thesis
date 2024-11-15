# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import Tensor
from torch_geometric.nn import GATConv
from models.GNN.common import *
from models.GNN.LB_GLAT.layers import LongTermLayerAttention
from utils.common_utils import setup_seed

"""
3 models about GAT:
    1. GAT + Fully Connection Layers
    2. Bi-graph + GAT + Fully Connection Layers
    3. GAT + Long-term Layer Attention + Fully Connection Layers
    4. Bi-graph + GAT + Long-term Layer Attention + Fully Connection Layers
"""

class GATsBlock(nn.Module):
    """GAT part"""

    def __init__(self, layer_num, n_features, gnns_hidden: Tensor, do_bn, dropout, bias):
        super(GATsBlock, self).__init__()
        dims = [n_features] + gnns_hidden.tolist()  # dims length = layer_num+1
        self.gats = nn.ModuleList([
            GATBlock(dims[i], dims[i + 1], do_bn, dropout, bias) for i in range(layer_num)  # Build layer i+1 GAT
        ])

    def forward(self, x, edge_index, mask):
        # print("GATsBlock Debugging:")
        # print("Input Features Shape (x):", x.shape)
        # print("Edge Index Shape Before GAT:", edge_index.shape)


        # #######convert
        # if edge_index.shape[0] != 2:
        #     from torch_geometric.utils import dense_to_sparse
        #     print("dense edge_index를 sparse format으로 변경")
        #     edge_index, _ = dense_to_sparse(edge_index)

        h = []  # 记录每次Relu(GAT)后的结果
        for gat in self.gats:
            x = gat(x, edge_index)
            h.append(x[mask])
        return h

class GATBlock(nn.Module):
    def __init__(self, in_features, out_features, do_bn, dropout, bias=True):
        super(GATBlock, self).__init__()
        # setup_seed(seed)
        self.gat = GATConv(in_features, out_features, bias=bias)
        self.do_bn = do_bn
        if do_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # print("GATBlock Debugging:")
        # print("Input Features Shape:", x.shape)
        # print("Edge Index Shape Inside GATBlock:", edge_index.shape)

        h = self.gat(x, edge_index)
        if self.do_bn:
            h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        return h
class GATFCModel(nn.Module):
    """
    1. GAT + Fully Connection Layers
    """

    def __init__(self, gnn_forward_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, linears_hidden: Tensor, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, device, bias=True):
        super(GATFCModel, self).__init__()
        self.device = device
        # GNN
        self.DGat = GATsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build GAT for the forward graph

        # Fully Connection Layers
        self.fc = FCsBlock(layer_num=linear_layer_num, n_features=int(gnns_forward_hidden[-1].item()),
                           linears_hidden=linears_hidden, n_classes=n_classes,
                           do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 1【Forward, reverse】

    def forward(self, x, edge_index, mask):
        # GAT
        h = self.DGat(x, edge_index, mask)
        # Fully Connection Layers
        h = self.fc(h[-1])
        return h


class BiGATFCModel(nn.Module):
    """
    2. Bi-graph + GAT + Fully Connection Layers
    """

    def __init__(self, gnn_forward_layer_num, gnn_reverse_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, gnns_reverse_hidden: Tensor, linears_hidden: Tensor, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, device, bias=True):
        super(BiGATFCModel, self).__init__()
        self.device = device
        # GNN
        self.DGat = GATsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build GAT for the forward graph
        self.RDGat = GATsBlock(layer_num=gnn_reverse_layer_num, n_features=n_features, gnns_hidden=gnns_reverse_hidden,
                               do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build GAT for the reverse graph

        # Fully Connection Layers
        self.fc_path1 = FCsBlock(layer_num=linear_layer_num,
                                 n_features=int(gnns_forward_hidden[-1].item()) + int(gnns_reverse_hidden[-1].item()),
                                 linears_hidden=linears_hidden, n_classes=n_classes,
                                 do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 1【Forward, reverse】
        self.fc_path2 = FCsBlock(layer_num=linear_layer_num, n_features=int(gnns_forward_hidden[-1].item()),
                                 linears_hidden=linears_hidden, n_classes=n_classes,
                                 do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 2【Forward】

    def forward(self, x, edge_index, mask):
        # Get the vector that needs to be masked(1,0,0,0,1,...)
        mask_1, mask_2, mask_3 = mask_h_vector(norm_adj(edgeIndex2CooAdj(x.shape[0], edge_index)))  # (batch, 1)
        mask_1 = mask_1.to(self.device)[mask]
        mask_2 = mask_2.to(self.device)[mask]
        mask_3 = mask_3.to(self.device)[mask]

        # GAT
        DGh = self.DGat(x, edge_index, mask)  # gnn_layer_num * (batch, gnns_hidden[-1])
        RDGh = self.RDGat(x, edge_index, mask)  # gnn_layer_num * (batch, gnns_hidden[-1])

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


class GATLTLAFCModel(nn.Module):
    """
    3. GAT + Long-term Layer Attention + Fully Connection Layers
    """

    def __init__(self, gnn_forward_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, linears_hidden: Tensor, project_hidden, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, tsf_dim, tsf_mlp_hidden, tsf_depth, tsf_heads, tsf_head_dim, tsf_dropout,
                 vit_emb_dropout, vit_pool, device, bias=True):
        super(GATLTLAFCModel, self).__init__()
        self.device = device
        # GNN
        self.DGat = GATsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build GAT for the forward graph

        # Long-Term Layer Attention
        self.patches_forward_dim = [n_features] + gnns_forward_hidden.tolist()
        self.DGltla = LongTermLayerAttention(num_patches=gnn_forward_layer_num + 1, patches_dim=self.patches_forward_dim,
                                             project_hidden=project_hidden, tsf_dim=tsf_dim,
                                             tsf_mlp_hidden=tsf_mlp_hidden, depth=tsf_depth,
                                             heads=tsf_heads, head_dim=tsf_head_dim, tsf_dropout=tsf_dropout,
                                             vit_emb_dropout=vit_emb_dropout, pool=vit_pool, bias=bias)  # Build LTLA for the forward graph

        # Fully Connection Layers
        self.fc = FCsBlock(layer_num=linear_layer_num, n_features=tsf_dim,
                           linears_hidden=linears_hidden, n_classes=n_classes,
                           do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 1【Forward, reverse】

    def forward(self, x, edge_index, mask):
        # GAT
        h = self.DGat(x, edge_index, mask)  # gnn_layer_num * (batch, gnns_hidden[-1])
        h.insert(0, x[mask])
        # Long-Term Layer Attention
        h = self.DGltla(h)  # (batch, project_hidden)
        # Fully Connection Layers
        h = self.fc(h)  # (batch, n_classes): not softmax
        return h


class BiGATLTLAFCModel(nn.Module):
    """
    4. Bi-graph + GAT + Long-term Layer Attention + Fully Connection Layers
    """

    def __init__(self, gnn_forward_layer_num, gnn_reverse_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, gnns_reverse_hidden: Tensor, linears_hidden: Tensor, project_hidden, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, tsf_dim, tsf_mlp_hidden, tsf_depth, tsf_heads, tsf_head_dim, tsf_dropout,
                 vit_emb_dropout, vit_pool, device, bias=True):
        super(BiGATLTLAFCModel, self).__init__()
        self.device = device
        # GNN
        self.DGat = GATsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build GAT for the forward graph
        self.RDGat = GATsBlock(layer_num=gnn_reverse_layer_num, n_features=n_features, gnns_hidden=gnns_reverse_hidden,
                               do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build GAT for the reverse graph

        # Long-Term Layer Attention
        self.patches_forward_dim = [n_features] + gnns_forward_hidden.tolist()
        self.patches_reverse_dim = [n_features] + gnns_reverse_hidden.tolist()
        self.DGltla = LongTermLayerAttention(num_patches=gnn_forward_layer_num + 1, patches_dim=self.patches_forward_dim,
                                             project_hidden=project_hidden, tsf_dim=tsf_dim,
                                             tsf_mlp_hidden=tsf_mlp_hidden, depth=tsf_depth,
                                             heads=tsf_heads, head_dim=tsf_head_dim, tsf_dropout=tsf_dropout,
                                             vit_emb_dropout=vit_emb_dropout, pool=vit_pool, bias=bias)  # Build LTLA for the forward graph
        self.RDGltla = LongTermLayerAttention(num_patches=gnn_reverse_layer_num + 1, patches_dim=self.patches_reverse_dim,
                                              project_hidden=project_hidden, tsf_dim=tsf_dim,
                                              tsf_mlp_hidden=tsf_mlp_hidden, depth=tsf_depth,
                                              heads=tsf_heads, head_dim=tsf_head_dim, tsf_dropout=tsf_dropout,
                                              vit_emb_dropout=vit_emb_dropout, pool=vit_pool, bias=bias)  # Build LTLA for the reverse graph

        # Fully Connection Layers
        self.fc_path1 = FCsBlock(layer_num=linear_layer_num, n_features=tsf_dim * 2,
                                 linears_hidden=linears_hidden, n_classes=n_classes,
                                 do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 1【Forward, reverse】
        self.fc_path2 = FCsBlock(layer_num=linear_layer_num, n_features=tsf_dim,
                                 linears_hidden=linears_hidden, n_classes=n_classes,
                                 do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)  # path 2【Forward】

    def forward(self, x, edge_index, mask):
        # Get the vector that needs to be masked(1,0,0,0,1,...)
        mask_1, mask_2, mask_3 = mask_h_vector(norm_adj(edgeIndex2CooAdj(x.shape[0], edge_index)))  # (batch, 1)
        mask_1 = mask_1.to(self.device)[mask]
        mask_2 = mask_2.to(self.device)[mask]
        mask_3 = mask_3.to(self.device)[mask]

        # GAT
        DGh = self.DGat(x, edge_index, mask)  # gnn_layer_num * (batch, gnns_hidden[-1])
        RDGh = self.RDGat(x, edge_index, mask)  # gnn_layer_num * (batch, gnns_hidden[-1])
        DGh.insert(0, x[mask])  # gnn_layer_num+1 * (batch, gnns_hidden[-1])
        RDGh.insert(0, x[mask])  # gnn_layer_num+1 * (batch, gnns_hidden[-1])

        # Long-Term Layer Attention
        DGh = self.DGltla(DGh)  # (batch, project_hidden)
        RDGh = self.RDGltla(RDGh)  # (batch, project_hidden)

        # Fully Connection Layers
        # path1
        h_1 = torch.cat((DGh, RDGh), dim=1)  # (batch, gnns_hidden[-1]*2)
        h_1 = self.fc_path1(h_1)  # (batch, n_classes): not softmax
        # path2
        h_2_1 = self.fc_path2(DGh)  # (batch, n_classes): not softmax
        h_2_2 = self.fc_path2(RDGh)  # (batch, n_classes): not softmax
        # Merge separate batches
        h_1 = torch.mul(mask_1.repeat(1, h_1.shape[1]), h_1)
        h_2_1 = torch.mul(mask_2.repeat(1, h_2_1.shape[1]), h_2_1)
        h_2_2 = torch.mul(mask_3.repeat(1, h_2_2.shape[1]), h_2_2)
        h = h_1 + h_2_1 + h_2_2  # (batch, n_classes): not softmax
        return h