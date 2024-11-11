# -*- coding: utf-8 -*-



from scipy.sparse import coo_matrix  # 转化成COO格式
import os
import random
import math
import torch
import numpy as np
from torch_geometric.data import Data  # 引入pyg的Data
from torch_geometric.loader import ClusterData, ClusterLoader
import utils.dataset_utils as du
from utils.common_utils import setup_seed, min_max_scaler
from utils.config_utils import get_config_option, get_config
from utils.file_utils import get_absolute_path_by_path
from tqdm import tqdm


#################################################
# train_mask val_mask test_mask

def get_dataset_list(seed):
    setup_seed(seed)
    """没有Cluster，以每个Time为一个子图batch，返回Data类型的List，并设置train_mask val_mask test_mask"""
    class_list, edge_list, node_list = du.get_dataset_all_time_np_list()
    time_end = int(get_config_option("dataset", "Elliptic", "time_end"))
    class_list = class_list[:(time_end+1)]
    print(len(class_list))
    edge_list = edge_list[:(time_end + 1)]
    node_list = node_list[:(time_end + 1)]
    data_list = []
    for i in tqdm(range(len(node_list)), desc=f"Get Data List: "):
        data_list.append(get_data(class_list[i], edge_list[i], node_list[i]))
    return data_list


def get_data(classes, edges, nodes):
    classes, edges, nodes = create_index(classes, edges, nodes)  # create index
    classes = classes[:, 1]
    nodes = nodes[:, 2:]
    class_tensor, edge_tensor, node_tensor = to_tensor(classes, edges, nodes)
    train_mask, val_mask, test_mask = get_process_mask(classes)
    train_mask = down_sampling_mask(classes, train_mask)  # under sampling
    train_mask_tensor, val_mask_tensor, test_mask_tensor = to_tensor_mask(train_mask, val_mask, test_mask)
    return Data(x=node_tensor, edge_index=edge_tensor, y=class_tensor,
                train_mask=train_mask_tensor, val_mask=val_mask_tensor, test_mask=test_mask_tensor)


def down_sampling_mask(classes, train_mask):
    """Narrow the positive and negative sample gap"""
    down_sampling = get_config_option("dataset", "Elliptic", "down_sampling") == str(True)
    if down_sampling:
        rs_NP_ratio = float(get_config_option("dataset", "Elliptic", "rs_NP_ratio"))
        P_num = (classes[train_mask] == 0).sum()  # the number of positive samples
        N_num = (classes[train_mask] == 1).sum()  # the number of negative samples
        if N_num <= math.floor(P_num * rs_NP_ratio):   # The number of negative samples is less than or equal to the expected negative samples
            return train_mask
        # Otherwise, under sampling
        Neg_index = np.where(classes[train_mask] == 1)[0]
        Neg_abandon_index = random.sample(list(Neg_index), N_num - math.floor(P_num * rs_NP_ratio))  # give up the Neg train index
        Neg_mask = np.full(classes.shape, True, dtype=bool)
        Neg_mask[Neg_abandon_index] = False   # give up the Neg train index
        train_mask = train_mask & Neg_mask
        # print(f"Random Sampling: Pos num{P_num}, Neg num{N_num}, reserve Neg num {math.ceil(P_num * rs_NP_ratio)}")
        return train_mask
    return train_mask


def get_process_mask(classes):
    """Get the mask for the Train Val Test for each Time"""
    train_val_test_ratio = np.array(
        list(map(float, get_config_option("dataset", "Elliptic", "train_val_test_ratio").split())))
    # 1. Get the subscript for each category 0 is illegal 1 is legal (the initial dataset is illegal for class 1, legal is class 2, so it is not changed), and get the train val test according to the ratio
    illicit_index = np.where(classes == 0)[0]
    train_illicit_mask, val_illicit_mask, test_illicit_mask = get_process_class_mask(classes, illicit_index, train_val_test_ratio)
    licit_index = np.where(classes == 1)[0]
    train_licit_mask, val_licit_mask, test_licit_mask = get_process_class_mask(classes, licit_index, train_val_test_ratio)
    # 2. Merge illegal and legal data
    return train_illicit_mask | train_licit_mask, val_illicit_mask | val_licit_mask, test_illicit_mask | test_licit_mask


def get_process_class_mask(classes, class_index, train_val_test_ratio):
    """Gets the Train Val Test mask for the class"""
    class_num = len(class_index)
    train_class_index = random.sample(list(class_index), math.floor(train_val_test_ratio[0] * class_num))
    class_index = np.setdiff1d(class_index, train_class_index)
    val_class_index = random.sample(list(class_index), math.floor(train_val_test_ratio[1] * class_num))
    test_class_index = np.setdiff1d(class_index, val_class_index)
    # Convert index to mask
    train_class_mask = np.full(classes.shape, False, dtype=bool)
    train_class_mask[train_class_index] = True
    val_class_mask = np.full(classes.shape, False, dtype=bool)
    val_class_mask[val_class_index] = True
    test_class_mask = np.full(classes.shape, False, dtype=bool)
    test_class_mask[test_class_index] = True
    return train_class_mask, val_class_mask, test_class_mask    # Each stage class1 needs to participate in True, the rest are False


def to_tensor(classes, edges, nodes):
    class_tensor = torch.LongTensor(classes)
    edge_tensor = torch.LongTensor(edges.transpose())
    node_tensor = torch.Tensor(nodes)
    return class_tensor, edge_tensor, node_tensor


def to_tensor_mask(train_mask, val_mask, test_mask):
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool)
    test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool)
    return train_mask_tensor, val_mask_tensor, test_mask_tensor


def create_index(classes, edges, nodes):
    idx = np.array(nodes[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges.flatten())),
                     dtype=np.int32).reshape(edges.shape)
    classes[:, 0] = np.array(list(map(idx_map.get, classes[:, 0])), dtype=np.int32)
    return classes[np.argsort(classes[:, 0])], edges, nodes

