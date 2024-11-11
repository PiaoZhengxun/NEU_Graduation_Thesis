# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import Tensor
from models.GNN.LB_GLAT.layers import *
from models.GNN.common import *
from models.GNN.GCN.models import GCNsBlock


class LBGLATModel(nn.Module):
    def __init__(self, gnn_forward_layer_num, gnn_reverse_layer_num, linear_layer_num, n_features, n_classes,
                 gnns_forward_hidden: Tensor, gnns_reverse_hidden: Tensor,
                 linears_hidden: Tensor, project_hidden, gnn_do_bn, linear_do_bn,
                 gnn_dropout, linear_dropout, tsf_dim, tsf_mlp_hidden, tsf_depth, tsf_heads, tsf_head_dim, tsf_dropout,
                 vit_emb_dropout, vit_pool, device, bias=True):
        super(LBGLATModel, self).__init__()
        self.device = device
        # Long-Term Layer Attention Network
        self.DGcn = GCNsBlock(layer_num=gnn_forward_layer_num, n_features=n_features, gnns_hidden=gnns_forward_hidden,
                              do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build the GCN of the forward graph
        self.RDGcn = GCNsBlock(layer_num=gnn_reverse_layer_num, n_features=n_features, gnns_hidden=gnns_reverse_hidden,
                               do_bn=gnn_do_bn, dropout=gnn_dropout, bias=bias)  # Build the GCN of the reverse graph

        # Long-Term Layer Attention
        self.patches_forward_dim = [n_features] + gnns_forward_hidden.tolist()
        self.patches_reverse_dim = [n_features] + gnns_reverse_hidden.tolist()
        self.DGltla = LongTermLayerAttention(num_patches=gnn_forward_layer_num+1, patches_dim=self.patches_forward_dim,
                                             project_hidden=project_hidden, tsf_dim=tsf_dim,
                                             tsf_mlp_hidden=tsf_mlp_hidden, depth=tsf_depth,
                                             heads=tsf_heads, head_dim=tsf_head_dim, tsf_dropout=tsf_dropout,
                                             vit_emb_dropout=vit_emb_dropout, pool=vit_pool, bias=bias)  # Build LTLA for the forward graph
        self.RDGltla = LongTermLayerAttention(num_patches=gnn_reverse_layer_num+1, patches_dim=self.patches_reverse_dim,
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
        # Convert the edge_index to a coo sparse matrix and calculate the D-1*A  # transpose matrix
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
        DGh.insert(0, x[mask])  # gnn_layer_num+1 * (batch, gnns_hidden[-1])
        RDGh.insert(0, x[mask])  # gnn_layer_num+1 * (batch, gnns_hidden[-1])

        # Long-Term Layer Attention
        # Mask out unwanted data
        DGh = self.DGltla(DGh)  # (batch, tsf_dim)
        RDGh = self.RDGltla(RDGh)  # (batch, tsf_dim)

        # Fully Connection Layers
        # path1
        h_1 = torch.cat((DGh, RDGh), dim=1)  # (batch, tsf_dim*2)
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


