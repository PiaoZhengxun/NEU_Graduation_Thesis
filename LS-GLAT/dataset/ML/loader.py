# -*- coding: utf-8 -*-


import numpy as np
from tqdm import tqdm
from dataset.cluster.loader import *
from utils.config_utils import get_config_option
from utils.dataset_utils import read_dataset_processed_np, get_dataset_train_test_time_np


SEED = int(get_config_option("GNN", "GNN", "seed"))
DATA_LIST = get_dataset_list_compare(SEED)

# def set_ssed_avoid_t():
#     seed = int(get_config_option("GNN", "GNN", "seed"))
#     data_list = get_dataset_list_compare(seed)
#     return data_list

def get_train_test_np():
    """It is divided into training set and test set 8:2, and is classified into two, 1 means illegal and 0 means legal"""
    # seed = int(get_config_option("GNN", "GNN", "seed"))
    # data_list = DATA_LIST
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for data in DATA_LIST:
        train_x.append(data.x[data.train_mask].cpu().numpy())
        train_y.append(data.y[data.train_mask].cpu().numpy())
        test_x.append(data.x[data.val_mask].cpu().numpy())
        test_y.append(data.y[data.val_mask].cpu().numpy())
        test_x.append(data.x[data.test_mask].cpu().numpy())
        test_y.append(data.y[data.test_mask].cpu().numpy())

    train_x = np.vstack(tuple(train_x))
    train_y = np.hstack(tuple(train_y))
    test_x = np.vstack(tuple(test_x))
    test_y = np.hstack(tuple(test_y))

    train_y = (~train_y.astype(bool)).astype(np.int32)  # Swap labels 0 and 1, 1 means illegal and 0 means legal
    test_y = (~test_y.astype(bool)).astype(np.int32)
    return train_x, train_y, test_x, test_y

def get_edge_index():
    # data_list = set_ssed_avoid_t()
    edge_index = [data.edge_index.cpu().numpy() for data in DATA_LIST]
    return edge_index


if __name__ == '__main__':
    # class_np, _, node_np = read_dataset_processed_np()
    get_train_test_np()

