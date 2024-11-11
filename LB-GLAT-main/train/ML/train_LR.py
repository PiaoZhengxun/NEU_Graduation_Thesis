# -*- coding: utf-8 -*-


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from dataset.ML.loader import get_train_test_np
from utils.config_utils import get_config_option

"""
Logistic Regression
"""


def LR(train_x, train_y, test_x, test_y):
    # Parameter setting
    solver = "liblinear"
    max_iter = int(get_config_option("GNN", "GNN", "epochs"))
    C = 1
    random_state = int(get_config_option("GNN", "GNN", "seed"))

    # Model
    lr = LogisticRegression(solver=solver, random_state=random_state, max_iter=max_iter)

    lr = lr.fit(train_x, train_y)  # Train

    test_y_pred = lr.predict(test_x)  # Test
    acc = accuracy_score(test_y, test_y_pred)
    precision = precision_score(test_y, test_y_pred, average=None)
    F1 = f1_score(test_y, test_y_pred, average=None)
    recall = recall_score(test_y, test_y_pred, average=None)
    # C = confusion_matrix(test_y, test_y_pred, labels=[0, 1])
    auc = roc_auc_score(test_y, test_y_pred)

    print(f"Logistic Regression Accuracy: {acc}, Precision: {precision}, F1-score: {F1}, Recall: {recall}, AUC: {auc}")
    return "LR", acc, precision[1], precision[0], recall[1], recall[0], F1[1], F1[0], auc


if __name__ == '__main__':
    # Get train data and test data
    train_x, train_y, test_x, test_y = get_train_test_np()
    LR(train_x, train_y, test_x, test_y)
