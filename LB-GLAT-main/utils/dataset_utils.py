# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
from utils.config_utils import get_config, get_config_option
from utils.file_utils import get_absolute_path_by_path


###############################################################
# Required for drawing

def get_data_df_by_txId(txId, dataset_name="Elliptic"):
    """
    Through txId, all the associated data of the transaction is obtained
    :param txId:
    :param dataset_name:
    :return:
    """
    class_df, edge_df, node_df = read_dataset_processed_df(dataset_name)
    nodes_num = 0
    nodes = [txId]
    edge_df1 = pd.DataFrame()
    while len(nodes) != nodes_num:
        nodes_num = len(nodes)
        edge_df1 = edge_df[edge_df.apply(lambda x: x["txId1"] in nodes or x["txId2"] in nodes, axis=1)]
        nodes = list(set(edge_df1["txId1"].tolist() + edge_df1["txId2"].tolist()))
        print(nodes)
        print(type(nodes))
        print(len(nodes))
    node_df = node_df[node_df.apply(lambda x: x[0] in nodes, axis=1)]
    class_df = class_df[class_df.apply(lambda x: x["txId"] in nodes, axis=1)]
    print("[Dataset] ==> dataset: [{}], txId: {}, class_df len: {}, edge_df len: {}, node_df len: {}".format(
        dataset_name, txId, len(class_df), len(edge_df1), len(node_df)))
    return class_df, edge_df1, node_df


def get_data_df_by_time(time, dataset_name="Elliptic"):
    """
    Through time, all the data of a certain time slice is obtained
    :param time:
    :param dataset_name:
    :return:
    """
    class_df, edge_df, node_df = read_dataset_processed_df(dataset_name)
    node_df = node_df.loc[node_df[1] == time, :]
    class_df = class_df[class_df.apply(lambda x: x["txId"] in node_df[0].values, axis=1)]
    edge_df = edge_df[edge_df.apply(lambda x: x["txId1"] in node_df[0].values and x["txId2"] in node_df[0].values,
                                    axis=1)]
    print("[Dataset] ==> dataset: [{}], time: {}, class_df len: {}, edge_df len: {}, node_df len: {}".format(
        dataset_name, time, len(class_df), len(edge_df), len(node_df)))
    return class_df, edge_df, node_df


def get_data_df_by_time_delete_class3(time, dataset_name="Elliptic"):
    """
    Get all the data of a certain time slice by time (delete class3)
    :param time:
    :param dataset_name:
    :return:
    """
    class_df, edge_df, node_df = read_dataset_processed_df(dataset_name)
    node_df = node_df.loc[node_df[1] == time, :]
    class_df = class_df[class_df.apply(lambda x: x["txId"] in node_df[0].values, axis=1)]
    class_df = class_df[class_df["class"] != 3]
    node_df = node_df[node_df.apply(lambda x: x[0] in class_df["txId"].values, axis=1)]
    edge_df = edge_df[edge_df.apply(lambda x: x["txId1"] in node_df[0].values and x["txId2"] in node_df[0].values,
                                    axis=1)]
    print("[Dataset] ==> dataset: [{}], time: {}, class_df len: {}, edge_df len: {}, node_df len: {}".format(
        dataset_name, time, len(class_df), len(edge_df), len(node_df)))
    return class_df, edge_df, node_df


###################################################################
# Read data


def read_dataset_df(dataset_name="Elliptic"):
    config = get_config("dataset")
    class_df = pd.read_csv(get_absolute_path_by_path(config.get("path", dataset_name + "_class_path")), sep=",",
                           encoding="utf-8")
    edge_df = pd.read_csv(get_absolute_path_by_path(config.get("path", dataset_name + "_edge_path")), sep=",",
                          encoding="utf-8")
    node_df = pd.read_csv(get_absolute_path_by_path(config.get("path", dataset_name + "_node_path")), sep=",",
                          encoding="utf-8", header=None)
    assert len(class_df) == len(node_df), "Elliptic dataset error: class len != node len"
    return class_df, edge_df, node_df


def read_dataset_processed_df(dataset_name="Elliptic"):
    config = get_config("dataset")
    class_df = pd.read_csv(get_absolute_path_by_path(config.get("path", dataset_name + "_class_1_path")), sep=",",
                           encoding="utf-8")
    edge_df = pd.read_csv(get_absolute_path_by_path(config.get("path", dataset_name + "_edge_path")), sep=",",
                          encoding="utf-8")
    node_df = pd.read_csv(get_absolute_path_by_path(config.get("path", dataset_name + "_node_path")), sep=",",
                          encoding="utf-8", header=None)
    assert len(class_df) == len(node_df), "Elliptic dataset error: class len != node len"
    return class_df, edge_df, node_df


def read_dataset_processed_np(dataset_name="Elliptic"):
    """Read the data in numpy form"""
    config = get_config("dataset")
    class_np = np.genfromtxt(get_absolute_path_by_path(config.get("path", dataset_name + "_class_1_path")),
                             delimiter=",", skip_header=1)
    edge_np = np.genfromtxt(get_absolute_path_by_path(config.get("path", dataset_name + "_edge_path")),
                            delimiter=",", skip_header=1)
    node_np = np.genfromtxt(get_absolute_path_by_path(config.get("path", dataset_name + "_node_path")),
                            delimiter=",")
    assert class_np.shape[0] == node_np.shape[0], "Elliptic dataset error: class len != node len"
    return class_np, edge_np, node_np


def get_dataset_train_test_time_np(dataset_name="Elliptic"):
    """
    Read into numpy according to time sharding
    :param dataset_name:
    :return: class_list, edge_list, node_list
    """
    write_dataset_processed_time_np(dataset_name)
    config = get_config("dataset")
    time_num = int(config.get("Elliptic", "time_num"))
    train_end_time = int(config.get("Elliptic", "train_end_time"))
    has_val = config.get("Elliptic", "has_val") == str(True)
    if has_val:
        val_end_time = int(config.get("Elliptic", "val_end_time"))
        train_list = np.load(get_absolute_path_by_path(
            config.get("path", "Elliptic_train_np_list_path")).format(train_end_time), allow_pickle=True)
        val_list = np.load(get_absolute_path_by_path(
            config.get("path", "Elliptic_val_np_list_path")).format(train_end_time + 1, val_end_time), allow_pickle=True)
        test_list = np.load(get_absolute_path_by_path(
            config.get("path", "Elliptic_test_np_list_path")).format(val_end_time + 1, time_num), allow_pickle=True)
        return train_list, val_list, test_list  # train_list["train_class_list"]
    else:
        train_list = np.load(get_absolute_path_by_path(config.get("path", "Elliptic_train_np_list_path")).format(train_end_time), allow_pickle=True)
        test_list = np.load(get_absolute_path_by_path(config.get("path", "Elliptic_test_np_list_path")).format(train_end_time+1, time_num), allow_pickle=True)
        return train_list, None, test_list  # train_list["train_class_list"]


def get_dataset_time_list(start_time, end_time, class_df, edge_df, node_df):
    """Get a list of numpy data for time"""
    node_list = []  # len = time_num and each element is a numpy
    class_list = []
    edge_list = []
    for i in range(start_time, end_time):  # time = i + 1
        node_i_df = node_df.loc[node_df[1] == i + 1, :]
        class_i_df = class_df[class_df.apply(lambda x: x["txId"] in node_i_df[0].values, axis=1)]
        edge_i_df = edge_df[
            edge_df.apply(lambda x: x["txId1"] in node_i_df[0].values and x["txId2"] in node_i_df[0].values,
                          axis=1)]
        node_list.append(node_i_df.to_numpy())
        class_list.append(class_i_df.to_numpy())
        edge_list.append(edge_i_df.to_numpy())
    return class_list, edge_list, node_list


def write_dataset_processed_time_np(dataset_name="Elliptic"):
    """Write to the files training and validation sets"""
    config = get_config("dataset")
    time_num = int(config.get("Elliptic", "time_num"))
    train_end_time = int(config.get("Elliptic", "train_end_time"))
    has_val = config.get("Elliptic", "has_val") == str(True)
    if has_val:
        val_end_time = int(config.get("Elliptic", "val_end_time"))
        Elliptic_train_np_list_path = get_absolute_path_by_path(
            config.get("path", "Elliptic_train_np_list_path")).format(train_end_time)
        Elliptic_val_np_list_path = get_absolute_path_by_path(
            config.get("path", "Elliptic_val_np_list_path")).format(train_end_time + 1, val_end_time)
        Elliptic_test_np_list_path = get_absolute_path_by_path(
            config.get("path", "Elliptic_test_np_list_path")).format(val_end_time + 1, time_num)
        if (os.path.exists(Elliptic_train_np_list_path) and os.path.exists(Elliptic_val_np_list_path)
                and os.path.exists(Elliptic_test_np_list_path)):
            return

        class_df, edge_df, node_df = read_dataset_processed_df(dataset_name)
        train_class_list, train_edge_list, train_node_list = get_dataset_time_list(0, train_end_time, class_df,
                                                                                   edge_df, node_df)
        val_class_list, val_edge_list, val_node_list = get_dataset_time_list(train_end_time, val_end_time, class_df,
                                                                                   edge_df, node_df)
        test_class_list, test_edge_list, test_node_list = get_dataset_time_list(val_end_time, time_num, class_df,
                                                                                edge_df, node_df)
        assert len(train_class_list) == train_end_time and len(train_edge_list) == train_end_time and \
               len(train_node_list) == train_end_time

        np.savez(Elliptic_train_np_list_path,
                 train_class_list=train_class_list, train_edge_list=train_edge_list, train_node_list=train_node_list)
        np.savez(Elliptic_val_np_list_path,
                 val_class_list=val_class_list, val_edge_list=val_edge_list, val_node_list=val_node_list)
        np.savez(Elliptic_test_np_list_path,
                 test_class_list=test_class_list, test_edge_list=test_edge_list, test_node_list=test_node_list)
    else:
        Elliptic_train_np_list_path = get_absolute_path_by_path(
            config.get("path", "Elliptic_train_np_list_path")).format(train_end_time)
        Elliptic_test_np_list_path = get_absolute_path_by_path(
            config.get("path", "Elliptic_test_np_list_path")).format(train_end_time+1, time_num)
        if os.path.exists(Elliptic_train_np_list_path) and os.path.exists(Elliptic_test_np_list_path):
            return

        class_df, edge_df, node_df = read_dataset_processed_df(dataset_name)

        train_class_list, train_edge_list, train_node_list = get_dataset_time_list(0, train_end_time, class_df, edge_df,
                                                                                   node_df)
        test_class_list, test_edge_list, test_node_list = get_dataset_time_list(train_end_time, time_num, class_df,
                                                                                edge_df, node_df)
        assert len(train_class_list) == train_end_time and len(train_edge_list) == train_end_time and \
               len(train_node_list) == train_end_time
        # print("[Dataset] ==> dataset: {}, len(class_list): {}, len(edge_list):{}, len(node_list): {} "
        #       "[must equal time_num=49]".format(dataset_name, len(class_list), len(edge_list), len(node_list)))

        np.savez(Elliptic_train_np_list_path,
                 train_class_list=train_class_list, train_edge_list=train_edge_list, train_node_list=train_node_list)
        np.savez(Elliptic_test_np_list_path,
                 test_class_list=test_class_list, test_edge_list=test_edge_list, test_node_list=test_node_list)


##############################################################
# Testing


# def test_read_dataset_processed_df(dataset_name="Elliptic"):
#     class_df = pd.read_csv(r"E:\programming\pythonProject\pycharmProject\LabProject\blockchain\AML\transgraph-aml\data\Elliptic_dataset\test_classes.csv", sep=",",
#                            encoding="utf-8")
#     edge_df = pd.read_csv(r"E:\programming\pythonProject\pycharmProject\LabProject\blockchain\AML\transgraph-aml\data\Elliptic_dataset\test_edges.csv", sep=",",
#                           encoding="utf-8")
#     node_df = pd.read_csv(r"E:\programming\pythonProject\pycharmProject\LabProject\blockchain\AML\transgraph-aml\data\Elliptic_dataset\test_nodes.csv", sep=",",
#                           encoding="utf-8", header=None)
#     assert len(class_df) == len(node_df), "Elliptic dataset error: class len != node len"
#     return class_df, edge_df, node_df


###############################################################

def get_dataset_all_time_np_list(dataset_name="Elliptic"):
    """
    # 1. Swap the dataset into a list
    # 2. save：class_all_time_np_list.npz
    """
    config = get_config("dataset")
    Elliptic_data_all_time_np_list_path = get_absolute_path_by_path(config.get("path", "Elliptic_data_all_time_np_list_path"))
    if os.path.exists(Elliptic_data_all_time_np_list_path):
        data_list = np.load(Elliptic_data_all_time_np_list_path, allow_pickle=True)
        return data_list["class_list"], data_list["edge_list"], data_list["node_list"]
    class_df, edge_df, node_df = read_dataset_processed_df(dataset_name)
    config = get_config("dataset")
    # class_list, edge_list, node_list
    class_list, edge_list, node_list = get_dataset_time_list(0, int(config.get("Elliptic", "time_num")), class_df, edge_df, node_df)
    np.savez(Elliptic_data_all_time_np_list_path,
             class_list=class_list, edge_list=edge_list, node_list=node_list)
    return class_list, edge_list, node_list



if __name__ == '__main__':

    class_df, edge_df, node_df = read_dataset_processed_df()
    edge_node = set(edge_df["txId1"].values) | set(edge_df["txId2"].values)
    a = node_df.apply(lambda x: x[0] in edge_node, axis=1)
    print(a.sum())
    print(len(node_df))
