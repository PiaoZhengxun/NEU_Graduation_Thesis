import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import SAGEConv
from models.GNN.common import *
from models.GNN.LS_GLAT.layers import LongTermLayerAttention

class SingleGraphSAGEModel(nn.Module):
    def __init__(self, gnn_layer_num, linear_layer_num, n_features, n_classes,
                gnns_hidden: Tensor, linears_hidden: Tensor, project_hidden, gnn_do_bn, linear_do_bn,
                gnn_dropout, linear_dropout, tsf_dim, tsf_mlp_hidden, tsf_depth, tsf_heads,
                tsf_head_dim, tsf_dropout, gt_emb_dropout, gt_pool, device, bias=True):
        super(SingleGraphSAGEModel, self).__init__()
        self.device = device
        self.Sage = SAGEsBlock(
            layer_num=gnn_layer_num,
            n_features=n_features,
            gnns_hidden=gnns_hidden,
            do_bn=gnn_do_bn,
            dropout=gnn_dropout,
            bias=bias
        )
        num_nodes = gnn_layer_num + 1
        node_dims = [n_features] + gnns_hidden.tolist()
        self.use_ltla = True
        if self.use_ltla:
            self.ltla = LongTermLayerAttention(
                num_nodes=num_nodes,
                node_dim=node_dims,
                project_hidden=project_hidden,
                tsf_dim=tsf_dim,
                tsf_mlp_hidden=tsf_mlp_hidden,
                depth=tsf_depth,
                heads=tsf_heads,
                head_dim=tsf_head_dim,
                tsf_dropout=tsf_dropout,
                gt_emb_dropout=gt_emb_dropout,
                gt_pool=gt_pool,
                bias=bias
            )
            fc_input_dim = tsf_dim
        else:
            fc_input_dim = int(gnns_hidden[-1].item())
        self.fc = FCsBlock(
            layer_num=linear_layer_num,
            n_features=fc_input_dim,
            linears_hidden=linears_hidden,
            n_classes=n_classes,
            do_bn=linear_do_bn,
            dropout=linear_dropout,
            bias=bias
        )
        self.fc = FCsBlock(layer_num=linear_layer_num, n_features=int(gnns_hidden[-1].item()),
                           linears_hidden=linears_hidden, n_classes=n_classes,
                           do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)

    def forward(self, x, edge_index, mask=None):
        h_list = self.Sage(x, edge_index)
        h_list.insert(0, x)
        if mask is not None:
            h_list = [h[mask] for h in h_list]
        if self.use_ltla:
            h = self.ltla(h_list, edge_index)
        else:
            h = h_list[-1]
        output = self.fc(h)
        return output

class SAGEsBlock(nn.Module):
    def __init__(self, layer_num, n_features, gnns_hidden, do_bn, dropout, bias=True):
        super(SAGEsBlock, self).__init__()
        dims = [n_features] + gnns_hidden.tolist()
        self.sages = nn.ModuleList([
            SAGEBlock(dims[i], dims[i + 1], do_bn, dropout, bias) for i in range(layer_num)
        ])

    def forward(self, x, edge_index, mask=None):
        h = []
        for sage in self.sages:
            x = sage(x, edge_index)
            if mask is not None:
                h.append(x[mask])
            else:
                h.append(x)
        return h

class SAGEBlock(nn.Module):
    def __init__(self, in_features, out_features, do_bn, dropout, bias=True):
        super(SAGEBlock, self).__init__()
        self.sage = SAGEConv(in_features, out_features, bias=bias)
        self.do_bn = do_bn
        if do_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.sage(x, edge_index)
        if self.do_bn:
            h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        return h