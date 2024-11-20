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

from models.GNN.DeeperGCN.models import SingleGraphSAGEModel
from models.GNN.GAT.models import *
from models.GNN.LS_GLAT.models import *
from utils.config_utils import *

GNN_config = get_config("GNN")
dataset_config = get_config("dataset")

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=GNN_config.get("GNN", "no_cuda") == str(True))
parser.add_argument('--fastmode', action='store_true', default=GNN_config.get("GNN", "fastmode") == str(True))
parser.add_argument('--seed', type=int, default=int(GNN_config.get("GNN", "seed")))
parser.add_argument('--epochs', type=int, default=int(GNN_config.get("GNN", "epochs")))
parser.add_argument('--lr', type=float, default=float(GNN_config.get("GNN", "lr0")))
parser.add_argument('--weight_decay', type=float, default=float(GNN_config.get("GNN", "weight_decay")))

args = parser.parse_args()

time_num = int(dataset_config.get("Elliptic", "time_num"))
time_end = int(dataset_config.get("Elliptic", "time_end"))
criterion_weight = np.array(list(map(float, dataset_config.get("Elliptic", "criterion_weight").split())))
train_val_test_ratio = np.array(list(map(float, dataset_config.get("Elliptic", "train_val_test_ratio").split())))
down_sampling = dataset_config.get("Elliptic", "down_sampling") == str(True)
rs_NP_ratio = float(dataset_config.get("Elliptic", "rs_NP_ratio"))
no_cuda = args.no_cuda
ctd = GNN_config.get("GNN", "ctd")
fastmode = args.fastmode
seed = args.seed
n_features = int(GNN_config.get("GNN", "n_features"))
n_classes = int(GNN_config.get("GNN", "n_classes"))
gnns_forward_hidden = torch.LongTensor(list(map(int, GNN_config.get("GNN", "gnns_forward_hidden").split())))
gnn_forward_layer_num = int(GNN_config.get("GNN", "gnn_forward_layer_num"))
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
project_hidden = int(GNN_config.get("LTLA", "project_hidden"))
tsf_dim = int(GNN_config.get("LTLA", "tsf_dim"))
tsf_mlp_hidden = int(GNN_config.get("LTLA", "tsf_mlp_hidden"))
tsf_depth = int(GNN_config.get("LTLA", "tsf_depth"))
tsf_heads = int(GNN_config.get("LTLA", "tsf_heads"))
tsf_head_dim = int(GNN_config.get("LTLA", "tsf_head_dim"))
tsf_dropout = float(GNN_config.get("LTLA", "tsf_dropout"))
gt_emb_dropout = float(GNN_config.get("LTLA", "gt_emb_dropout"))
gt_pool = GNN_config.get("LTLA", "gt_pool")

result_path = "../../result/GNN"
os.environ["CUDA_VISIBLE_DEVICES"] = ctd

use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("Cuda Available:{}, use {}!".format(use_cuda, device))

def creat_LSGLAT():
    return LSGLATModel(
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
        gt_emb_dropout=gt_emb_dropout,
        gt_pool=gt_pool,
        device=device,
        bias=bias
    )

def create_SingleGraphSAGEModel():
    return SingleGraphSAGEModel(
        gnn_layer_num=gnn_forward_layer_num,
        linear_layer_num=linear_layer_num,
        n_features=n_features,
        n_classes=n_classes,
        gnns_hidden=gnns_forward_hidden,
        linears_hidden=linears_hidden,
        gnn_do_bn=gnn_do_bn,
        linear_do_bn=linear_do_bn,
        gnn_dropout=gnn_dropout,
        linear_dropout=linear_dropout,
        device=device,
        bias=bias,
        project_hidden=project_hidden,
        tsf_dim=tsf_dim,
        tsf_mlp_hidden=tsf_mlp_hidden,
        tsf_depth=tsf_depth,
        tsf_heads=tsf_heads,
        tsf_head_dim=tsf_head_dim,
        tsf_dropout=tsf_dropout,
        gt_emb_dropout=gt_emb_dropout,
        gt_pool=gt_pool
    )