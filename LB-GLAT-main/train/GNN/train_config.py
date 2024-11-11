# -*- coding: utf-8 -*-

import random
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch_geometric
import torch.backends.cudnn as cudnn

from models.GNN.DeeperGCN.models import BiGENFCModel
from models.GNN.GAT.models import *
from models.GNN.GraphSAGE.models import *
from models.GNN.LB_GLAT.models import *
from models.GNN.GCN.models import *
from utils.config_utils import *

# get config
GNN_config = get_config("GNN")
dataset_config = get_config("dataset")
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=GNN_config.get("GNN", "no_cuda") == str(True),
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=GNN_config.get("GNN", "fastmode") == str(True),
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=int(GNN_config.get("GNN", "seed")), help='Random seed.')
parser.add_argument('--epochs', type=int, default=int(GNN_config.get("GNN", "epochs")),
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=float(GNN_config.get("GNN", "lr0")),
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=float(GNN_config.get("GNN", "weight_decay")),
                    help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--dropout', type=float, default=float(GNN_config.get("GNN", "dropout")),
#                     help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

# config settings
# dataset config setting
time_num = int(dataset_config.get("Elliptic", "time_num"))
time_end = int(dataset_config.get("Elliptic", "time_end"))
# train_end_time = int(dataset_config.get("Elliptic", "train_end_time"))
criterion_weight = np.array(list(map(float, dataset_config.get("Elliptic", "criterion_weight").split())))
train_val_test_ratio = np.array(list(map(float, dataset_config.get("Elliptic", "train_val_test_ratio").split())))
down_sampling = dataset_config.get("Elliptic", "down_sampling") == str(True)
rs_NP_ratio = float(dataset_config.get("Elliptic", "rs_NP_ratio"))
# GNN config settings
no_cuda = args.no_cuda
ctd = GNN_config.get("GNN", "ctd")
fastmode = args.fastmode
# fastmode = False
seed = args.seed
n_features = int(GNN_config.get("GNN", "n_features"))
n_classes = int(GNN_config.get("GNN", "n_classes"))
gnns_forward_hidden = torch.LongTensor(list(map(int, GNN_config.get("GNN", "gnns_forward_hidden").split())))
gnn_forward_layer_num = int(GNN_config.get("GNN", "gnn_forward_layer_num"))
gnns_reverse_hidden = torch.LongTensor(list(map(int, GNN_config.get("GNN", "gnns_reverse_hidden").split())))
gnn_reverse_layer_num = int(GNN_config.get("GNN", "gnn_reverse_layer_num"))
linear_layer_num = int(GNN_config.get("GNN", "linear_layer_num"))
linears_hidden = torch.LongTensor(list(map(int, GNN_config.get("GNN", "linears_hidden").split())))
bias = GNN_config.get("GNN", "bias") == str(True)
opt = GNN_config.get("GNN", "opt")
adam_beta = tuple(map(float, GNN_config.get("GNN", "adam_beta").split()))
opt_momentum = float(GNN_config.get("GNN", "opt_momentum"))
lr0 = args.lr
decay_rate = float(GNN_config.get("GNN", "decay_rate"))
weight_decay = args.weight_decay
gnn_do_bn = GNN_config.get("GNN", "gnn_do_bn") == str(True)
linear_do_bn = GNN_config.get("GNN", "linear_do_bn") == str(True)
gnn_dropout = float(GNN_config.get("GNN", "gnn_dropout"))
linear_dropout = float(GNN_config.get("GNN", "linear_dropout"))
start_epoch = int(GNN_config.get("GNN", "start_epoch"))
epochs = args.epochs
model_name = GNN_config.get("GNN", "model_name")
model_folder = GNN_config.get("GNN", "model_folder")
# LB_GLAT config setting
project_hidden = int(GNN_config.get("LTLA", "project_hidden"))
tsf_dim = int(GNN_config.get("LTLA", "tsf_dim"))
tsf_mlp_hidden = int(GNN_config.get("LTLA", "tsf_mlp_hidden"))
tsf_depth = int(GNN_config.get("LTLA", "tsf_depth"))
tsf_heads = int(GNN_config.get("LTLA", "tsf_heads"))
tsf_head_dim = int(GNN_config.get("LTLA", "tsf_head_dim"))
tsf_dropout = float(GNN_config.get("LTLA", "tsf_dropout"))
vit_emb_dropout = float(GNN_config.get("LTLA", "vit_emb_dropout"))
vit_pool = GNN_config.get("LTLA", "vit_pool")

result_path = "../../result/GNN"

# cuda
os.environ["CUDA_VISIBLE_DEVICES"] = ctd
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("Cuda Available:{}, use {}!".format(use_cuda, device))



# Initialize model function
######################################
# LB_GLAT
def creat_LBGLAT():
    """
    The model of LB_GLAT Folder
    :return: model
    """
    return LBGLATModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        gnn_reverse_layer_num=gnn_reverse_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        gnns_reverse_hidden=gnns_reverse_hidden,
        linears_hidden=linears_hidden,
        project_hidden=project_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        tsf_dim=tsf_dim,
        tsf_mlp_hidden=tsf_mlp_hidden,
        tsf_depth=tsf_depth,
        tsf_heads=tsf_heads,
        tsf_head_dim=tsf_head_dim,
        tsf_dropout=tsf_dropout,
        vit_emb_dropout=vit_emb_dropout,
        vit_pool=vit_pool,
        device=device,
        bias=bias
    )


#######################################
# GCN
def creat_GCNFC():
    """
    The model of GCN Folder:
        1. GCN + Fully Connection Layers
    :return: model
    """
    return GCNFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        linears_hidden=linears_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        device=device,
        bias=bias
    )


def creat_BiGCNFC():
    """
    The model of GCN Folder:
        2. Bi-graph + GCN + Fully Connection Layers
    :return: model
    """
    return BiGCNFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        gnn_reverse_layer_num=gnn_reverse_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        gnns_reverse_hidden=gnns_reverse_hidden,
        linears_hidden=linears_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        device=device,
        bias=bias
    )


def creat_GCNLTLAFC():
    """
    The model of GCN Folder:
        3. GCN + Long-term Layer Attention + Fully Connection Layers
    :return: model
    """
    return GCNLTLAFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        linears_hidden=linears_hidden,
        project_hidden=project_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        tsf_dim=tsf_dim,
        tsf_mlp_hidden=tsf_mlp_hidden,
        tsf_depth=tsf_depth,
        tsf_heads=tsf_heads,
        tsf_head_dim=tsf_head_dim,
        tsf_dropout=tsf_dropout,
        vit_emb_dropout=vit_emb_dropout,
        vit_pool=vit_pool,
        device=device,
        bias=bias
    )


#######################################
# GAT
def creat_GATFC():
    """
    The model of GAT Folder:
        1. GAT + Fully Connection Layers
    :return: model
    """
    return GATFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        linears_hidden=linears_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        device=device,
        bias=bias
    )


def creat_BiGATFC():
    """
    The model of GAT Folder:
        2. Bi-graph + GAT + Fully Connection Layers
    :return: model
    """
    return BiGATFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        gnn_reverse_layer_num=gnn_reverse_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        gnns_reverse_hidden=gnns_reverse_hidden,
        linears_hidden=linears_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        device=device,
        bias=bias
    )


def creat_GATLTLAFC():
    """
    The model of GAT Folder:
        3. GAT + Long-term Layer Attention + Fully Connection Layers
    :return: model
    """
    return GATLTLAFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        linears_hidden=linears_hidden,
        project_hidden=project_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        tsf_dim=tsf_dim,
        tsf_mlp_hidden=tsf_mlp_hidden,
        tsf_depth=tsf_depth,
        tsf_heads=tsf_heads,
        tsf_head_dim=tsf_head_dim,
        tsf_dropout=tsf_dropout,
        vit_emb_dropout=vit_emb_dropout,
        vit_pool=vit_pool,
        device=device,
        bias=bias
    )


def creat_BiGATLTLAFC():
    """
    The model of GAT Folder:
        4. Bi-graph + GAT + Long-term Layer Attention + Fully Connection Layers
    :return: model
    """
    return BiGATLTLAFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        gnn_reverse_layer_num=gnn_reverse_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        gnns_reverse_hidden=gnns_reverse_hidden,
        linears_hidden=linears_hidden,
        project_hidden=project_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        tsf_dim=tsf_dim,
        tsf_mlp_hidden=tsf_mlp_hidden,
        tsf_depth=tsf_depth,
        tsf_heads=tsf_heads,
        tsf_head_dim=tsf_head_dim,
        tsf_dropout=tsf_dropout,
        vit_emb_dropout=vit_emb_dropout,
        vit_pool=vit_pool,
        device=device,
        bias=bias
    )


#######################################
# GraphSAGE
def creat_SAGEFC():
    """
    The model of GraphSAGE Folder:
        1. GraphSAGE + Fully Connection Layers
    :return: model
    """
    return SAGEFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        linears_hidden=linears_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        device=device,
        bias=bias
    )


def creat_BiSAGEFC():
    """
    The model of GraphSAGE Folder:
        2. Bi-graph + GraphSAGE + Fully Connection Layers
    :return: model
    """
    return BiSAGEFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        gnn_reverse_layer_num=gnn_reverse_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        gnns_reverse_hidden=gnns_reverse_hidden,
        linears_hidden=linears_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        device=device,
        bias=bias
    )


def creat_SAGELTLAFC():
    """
    The model of GraphSAGE Folder:
        3. GraphSAGE + Long-term Layer Attention + Fully Connection Layers
    :return: model
    """
    return SAGELTLAFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        linears_hidden=linears_hidden,
        project_hidden=project_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        tsf_dim=tsf_dim,
        tsf_mlp_hidden=tsf_mlp_hidden,
        tsf_depth=tsf_depth,
        tsf_heads=tsf_heads,
        tsf_head_dim=tsf_head_dim,
        tsf_dropout=tsf_dropout,
        vit_emb_dropout=vit_emb_dropout,
        vit_pool=vit_pool,
        device=device,
        bias=bias
    )


def creat_BiSAGELTLAFC():
    """
    The model of GraphSAGE Folder:
        4. Bi-graph + GraphSAGE + Long-term Layer Attention + Fully Connection Layers
    :return: model
    """
    return BiSAGELTLAFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        gnn_reverse_layer_num=gnn_reverse_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        gnns_reverse_hidden=gnns_reverse_hidden,
        linears_hidden=linears_hidden,
        project_hidden=project_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        tsf_dim=tsf_dim,
        tsf_mlp_hidden=tsf_mlp_hidden,
        tsf_depth=tsf_depth,
        tsf_heads=tsf_heads,
        tsf_head_dim=tsf_head_dim,
        tsf_dropout=tsf_dropout,
        vit_emb_dropout=vit_emb_dropout,
        vit_pool=vit_pool,
        device=device,
        bias=bias
    )


#######################################
# DeeperGCN
def creat_BiGENFCModel():
    """
    The model of DeeperGCN Folder:
        4. Bi-graph + DeeperGCN + Long-term Layer Attention + Fully Connection Layers
    :return: model
    """
    return BiGENFCModel(
        gnn_forward_layer_num=gnn_forward_layer_num,
        gnn_reverse_layer_num=gnn_reverse_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_forward_hidden=gnns_forward_hidden,
        gnns_reverse_hidden=gnns_reverse_hidden,
        linears_hidden=linears_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        device=device,
        bias=bias
    )


