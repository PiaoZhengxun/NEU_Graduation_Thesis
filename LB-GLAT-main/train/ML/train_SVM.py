# -*- coding: utf-8 -*-


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from dataset.ML.loader import get_train_test_np
from utils.config_utils import get_config_option

"""
SVM
"""


def SVM(train_x, train_y, test_x, test_y):
    # Parameter setting
    random_state = int(get_config_option("GNN", "GNN", "seed"))
    kernel = "rbf"  # （RBF, Linear, Poly, Sigmoid, default "RBF"）
    C = 1

    # Model
    svm = SVC(kernel=kernel, random_state=random_state)

    svm = svm.fit(train_x, train_y)  # Train

    test_y_pred = svm.predict(test_x)  # Test
    acc = accuracy_score(test_y, test_y_pred)
    precision = precision_score(test_y, test_y_pred, average=None)
    F1 = f1_score(test_y, test_y_pred, average=None)
    recall = recall_score(test_y, test_y_pred, average=None)
    # C = confusion_matrix(test_y, test_y_pred, labels=[0, 1])
    auc = roc_auc_score(test_y, test_y_pred)

    print(f"SVM Accuracy: {acc}, Precision: {precision}, F1-score: {F1}, Recall: {recall}, AUC: {auc}")
    return "SVM", acc, precision[1], precision[0], recall[1], recall[0], F1[1], F1[0], auc


if __name__ == '__main__':
    # Get train data and test data
    train_x, train_y, test_x, test_y = get_train_test_np()
    SVM(train_x, train_y, test_x, test_y)
