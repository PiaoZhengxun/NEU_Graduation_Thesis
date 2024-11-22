# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from dataset.ML.loader import get_train_test_np
from train.ML.train_DT import DT
from train.ML.train_LR import LR
from train.ML.train_SVM import SVM


"""
Machine Learning
"""


def ML_results():
    # Get train data and test data
    train_x, train_y, test_x, test_y = get_train_test_np()
    columns = ["ML", "test_acc", "test_precision_pos", "test_precision_neg", "test_recall_pos", "test_recall_neg",
               "test_F1_pos", "test_F1_neg", "test_AUC"]
    results = pd.DataFrame(np.zeros(shape=(3, len(columns))), columns=columns)
    results.iloc[0, :] = LR(train_x, train_y, test_x, test_y)
    results.iloc[1, :] = DT(train_x, train_y, test_x, test_y)
    results.iloc[2, :] = SVM(train_x, train_y, test_x, test_y)
    results = results.loc[:, ["ML", "test_acc", "test_precision_pos", "test_recall_pos",
               "test_F1_pos", "test_AUC"]]
    results.to_csv(f'../../result/ML/ML_82.csv', mode='w', header=True, index=False)
    print(results)


if __name__ == '__main__':
    ML_results()
