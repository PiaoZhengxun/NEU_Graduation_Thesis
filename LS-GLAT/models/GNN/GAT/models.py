import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import Tensor
from torch_geometric.nn import GATConv
from models.GNN.common import *
from models.GNN.LS_GLAT.layers import LongTermLayerAttention
from utils.common_utils import setup_seed

class GATsBlock(nn.Module):
    def __init__(self, layer_num, n_features, gnns_hidden: Tensor, do_bn, dropout, bias):
        super(GATsBlock, self).__init__()
        dims = [n_features] + gnns_hidden.tolist()
        self.gats = nn.ModuleList([
            GATBlock(dims[i], dims[i + 1], do_bn, dropout, bias) for i in range(layer_num)
        ])

    def forward(self, x, edge_index, mask):
        h = []
        for gat in self.gats:
            x = gat(x, edge_index)
            h.append(x[mask])
        return h

class GATBlock(nn.Module):
    def __init__(self, in_features, out_features, do_bn, dropout, bias=True):
        super(GATBlock, self).__init__()
        self.gat = GATConv(in_features, out_features, bias=bias)
        self.do_bn = do_bn
        if do_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        if self.do_bn:
            h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        return h

class GATFCModel(nn.Module):
    def __init__(self, gnn_forward_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, linears_hidden: Tensor, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, device, bias=True):
        super(GATFCModel, self).__init__()
        self.device = device
        self.DGat = GATsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)
        self.fc = FCsBlock(layer_num=linear_layer_num, n_features=int(gnns_forward_hidden[-1].item()),
                           linears_hidden=linears_hidden, n_classes=n_classes,
                           do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)

    def forward(self, x, edge_index, mask):
        h = self.DGat(x, edge_index, mask)
        h = self.fc(h[-1])
        return h

class GATLTLAFCModel(nn.Module):
    def __init__(self, gnn_forward_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, linears_hidden: Tensor, project_hidden, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, tsf_dim, tsf_mlp_hidden, tsf_depth, tsf_heads, tsf_head_dim, tsf_dropout,
                 vit_emb_dropout, vit_pool, device, bias=True):
        super(GATLTLAFCModel, self).__init__()
        self.device = device
        self.DGat = GATsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)
        self.patches_forward_dim = [n_features] + gnns_forward_hidden.tolist()
        self.DGltla = LongTermLayerAttention(num_patches=gnn_forward_layer_num + 1, patches_dim=self.patches_forward_dim,
                                             project_hidden=project_hidden, tsf_dim=tsf_dim,
                                             tsf_mlp_hidden=tsf_mlp_hidden, depth=tsf_depth,
                                             heads=tsf_heads, head_dim=tsf_head_dim, tsf_dropout=tsf_dropout,
                                             vit_emb_dropout=vit_emb_dropout, pool=vit_pool, bias=bias)
        self.fc = FCsBlock(layer_num=linear_layer_num, n_features=tsf_dim,
                           linears_hidden=linears_hidden, n_classes=n_classes,
                           do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)

    def forward(self, x, edge_index, mask):
        h = self.DGat(x, edge_index, mask)
        h.insert(0, x[mask])
        h = self.DGltla(h)
        h = self.fc(h)
        return h