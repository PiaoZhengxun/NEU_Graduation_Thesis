# -*- coding: utf-8 -*-


"""
process the Elliptic dataset:
    elliptic_txs_classes.csv ==> elliptic_txs_classes_1.csv :
        unknown class ==> 3 class
"""

from utils.dataset_utils import read_dataset_df, read_dataset_processed_df


def process_class3():
    elliptic_class_df, elliptic_edge_df, elliptic_node_df = read_dataset_df()
    # elliptic_class_df, elliptic_edge_df, elliptic_node_df = read_dataset_processed_df()

    # print(elliptic_class_df.info())  # 203769
    # print(elliptic_class_df.head(5))

    elliptic_class_df["class"] = elliptic_class_df["class"].str.replace("unknown", "3").astype("int32")
    print(elliptic_class_df.info())  # 203769
    print(elliptic_class_df.head(5))
    elliptic_class_df.to_csv('../data/Elliptic_dataset/elliptic_txs_classes_2.csv', mode='w', header=True, index=False)

    # print(elliptic_edge_df.info())  # 234355
    # print(elliptic_edge_df.head(5))

    # print(elliptic_node_df.info())  # 203769, 167(1+94+72)
    # print(elliptic_node_df.head(5))


def process_data():
    elliptic_class_df, elliptic_edge_df, elliptic_node_df = read_dataset_df()
    print(elliptic_class_df.info())  # 203769
    print(elliptic_class_df.head(5))
    print("*"*100)
    elliptic_class_df["class"] = elliptic_class_df["class"].str.replace("1", "0").replace("2", "1").replace("unknown", "2").astype("int32")
    print(elliptic_class_df.info())  # 203769
    print(elliptic_class_df.head(5))
    elliptic_class_df.to_csv('../data/Elliptic_dataset/elliptic_txs_classes_1.csv', mode='w', header=True, index=False)

if __name__ == '__main__':
    process_data()
