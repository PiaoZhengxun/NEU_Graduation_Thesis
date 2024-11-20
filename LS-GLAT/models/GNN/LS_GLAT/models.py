# -*- coding: utf-8 -*-

from models.GNN.GAT.models import GATsBlock
from models.GNN.LS_GLAT.layers import *
from models.GNN.common import *

class LSGLATModel(nn.Module):
    #Long-Term Single-Graph Layer Attention Convolutional Network
    def __init__(self, gnn_forward_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, linears_hidden: Tensor, project_hidden, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, tsf_dim, tsf_mlp_hidden, tsf_depth, tsf_heads, tsf_head_dim, tsf_dropout,
                 gt_emb_dropout, gt_pool, device, bias=True):
        super(LSGLATModel, self).__init__()
        self.device = device
        self.DGcn = GATsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                                do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)

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
            emb_dropout=gt_emb_dropout,
            pool=gt_pool,
            bias=bias
        )
        self.fc_path2 = FCsBlock(layer_num=linear_layer_num, n_features=tsf_dim,
                                    linears_hidden=linears_hidden, n_classes=n_classes,
                                    do_bn=linear_do_bn, dropout=linear_dropout, bias=bias)

    def forward(self, x, edge_index, mask):
        coo_adj = norm_adj(edgeIndex2CooAdj(x.shape[0], edge_index))
        coo_adj = sparse_mx_to_torch_sparse_tensor(coo_adj).to(self.device)

        DGh = self.DGcn(x, coo_adj, mask)
        DGh.insert(0, x[mask])

        DGh = self.DGltla(DGh, edge_index)

        h = self.fc_path2(DGh)

        return h