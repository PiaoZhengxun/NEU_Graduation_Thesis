import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import Tensor
from torch_geometric.nn import GATConv
from models.GNN.common import *
from models.GNN.LS_GLAT.layers import LongTermLayerAttention
from utils.common_utils import setup_seed


"""
1. Bidirectional Graph + GAT + Long Term Layer Attention (GATNoFCModel)
2. GAT + Full Connection Layers + Long Term Layer Attention (GATNoSingleGraphModel)
3. Bidirectional Graph + GAT + Full Connection Layers (GATNoLTLAFCModel)
4. Bidirectional Graph + GAT + Long Term Layer Attention + Full Connection Layers (LSGLATModel)
"""
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


#############################################################################################################################################

class GATNoFCModel(nn.Module):
    """
    1. Bidirectional Graph + GAT + Long Term Layer Attention
    problem: RuntimeError: weight tensor should be defined either for all 128 classes or no classes but got weight tensor of shape: [2]
    --> FCsBlock needs
    solution
    1. add a linear layer to match output // 这样的话还是会用fc 拉倒吧
    2. adjust the loss function
    """

    def __init__(self, gnn_forward_layer_num, n_features, gnns_forward_hidden, project_hidden, gnn_do_bn,
                    gnn_dropout, tsf_dim, tsf_mlp_hidden, tsf_depth, tsf_heads, tsf_head_dim, tsf_dropout,
                    gt_emb_dropout, gt_pool, device, bias=True):
        super(GATNoFCModel, self).__init__()
        self.device = device
        ## adjust loss function
        # t_gnns_hidden = gnns_forward_hidden.tolist()
        # t_gnns_hidden[-1] = tsf_dim

        #n_classes = 2
        # gnns_forward_hidden[-1] = 2

        #Graph Attention Layer ok
        self.DGat = GATsBlock(
            layer_num=gnn_forward_layer_num,
            n_features=n_features,
            gnns_hidden=gnns_forward_hidden,
            do_bn=gnn_do_bn,
            dropout=gnn_dropout,
            bias=bias
        )
        #ltlta
        self.patches_forward_dim = [n_features] + gnns_forward_hidden.tolist()
        self.DGltla = LongTermLayerAttention(
            num_nodes=gnn_forward_layer_num + 1,
            node_dim=self.patches_forward_dim,
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

        # linear layer
        self.classifier = nn.Linear(tsf_dim, 2)

    def forward(self, x, edge_index, mask):
        # h = self.DGat(x, edge_index, mask) #go through GAT
        # h.insert(0, x[mask]) #insert node feature for ltla
        # h = self.DGltla(h, edge_index) #apply ltla
        # l_classifier = self.classifier(h)
        # return l_classifier
        coo_adj = norm_adj(edgeIndex2CooAdj(x.shape[0], edge_index))
        coo_adj = sparse_mx_to_torch_sparse_tensor(coo_adj).to(self.device)

        DGh = self.DGat(x, coo_adj, mask)
        DGh.insert(0, x[mask])

        DGh = self.DGltla(DGh, edge_index)

        # h = self.fc(DGh)
        return self.classifier(DGh)

class GATNoBidirectionalGraphModel(nn.Module):
    """
    2. GAT + Full Connection Layers + Long Term Layer Attention
    --> No bidirectional Graph means Random Edges or No Edges
    """

    def __init__(self, gnn_forward_layer_num, linear_layer_num, n_features, n_classes,
                    gnns_forward_hidden: Tensor, linears_hidden: Tensor, project_hidden, gnn_do_bn, linear_do_bn,
                    gnn_dropout, linear_dropout, tsf_dim, tsf_mlp_hidden, tsf_depth, tsf_heads, tsf_head_dim, tsf_dropout,
                    gt_emb_dropout, gt_pool, device, bias=True):
        super(GATNoBidirectionalGraphModel, self).__init__()
        self.device = device
        self.DGat = GATsBlock(
            layer_num=gnn_forward_layer_num,
            n_features=n_features,
            gnns_hidden=gnns_forward_hidden,
            do_bn=gnn_do_bn,
            dropout=gnn_dropout,
            bias=bias
        )
        self.fc = FCsBlock(
            layer_num=linear_layer_num,
            n_features=int(gnns_forward_hidden[-1].item()),
            linears_hidden=linears_hidden,
            n_classes=n_classes,
            do_bn=linear_do_bn,
            dropout=linear_dropout,
            bias=bias
        )

    def forward(self, x, edge_index, mask):
        num_nodes = x.size(0)
        random_edge_index = torch.randint(0, num_nodes, edge_index.size(), device=edge_index.device)
        coo_adj = norm_adj(edgeIndex2CooAdj(x.shape[0], random_edge_index))
        coo_adj = sparse_mx_to_torch_sparse_tensor(coo_adj).to(self.device)
        h = self.DGat(x, coo_adj, mask)
        result = self.fc(h[-1])
        return result

class GATNoLTLAFCModel(nn.Module):
    """
    3. Bidirectional Graph + GAT + Full Connection Layers
    """
    def __init__(self, gnn_forward_layer_num, linear_layer_num, n_features, n_classes,
                    gnns_forward_hidden: Tensor, linears_hidden: Tensor, gnn_do_bn, linear_do_bn,
                    gnn_dropout, linear_dropout, device, bias=True):
        super(GATNoLTLAFCModel, self).__init__()
        self.device = device
        self.DGat = GATsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                                do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)
        self.fc = FCsBlock(layer_num=linear_layer_num, n_features=int(gnns_forward_hidden[-1].item()),
                            linears_hidden=linears_hidden, n_classes=n_classes,
                            do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)

    def forward(self, x, edge_index, mask):
        coo_adj = norm_adj(edgeIndex2CooAdj(x.shape[0], edge_index))
        coo_adj = sparse_mx_to_torch_sparse_tensor(coo_adj).to(self.device)

        h = self.DGat(x, coo_adj, mask)
        h = self.fc(h[-1])
        return h






