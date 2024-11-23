# -*- coding: utf-8 -*-


import torch

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from dataset.ML.loader import get_train_test_np
from utils.config_utils import get_config_option

from MLP import MLPModel, train_mlp


"""
SVM : support vector machine
"""


def SVM(train_x, train_y, test_x, test_y):
    input_dim = int(get_config_option("GNN", "GNN", "n_features"))
    hidden_dim = int(get_config_option("GNN", "GNN", "gnns_forward_hidden").split()[0])
    output_dim = int(get_config_option("GNN", "GNN", "n_classes"))
    epochs = int(get_config_option("GNN", "GNN", "epochs"))
    learning_rate = float(get_config_option("GNN", "GNN", "lr0"))
    random_state = int(get_config_option("GNN", "GNN", "seed"))
    kernel = "rbf" #Radial Basis Function

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.long).to(device)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.long).to(device)

    model = MLPModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    trained_mlp = train_mlp(model, train_x, train_y, epochs=epochs, learning_rate=learning_rate, device=device)


    model.eval()
    with torch.no_grad():
        train_features = trained_mlp.mlp[:-1](train_x).to(device) #.cpu().numpy()
        test_features = trained_mlp.mlp[:-1](test_x).to(device) #.cpu().numpy()

    svm = SVC(kernel=kernel, random_state=random_state, probability=True)
    svm.fit(train_features, train_y)

    test_y_pred = svm.predict(test_features)
    test_y_prob = svm.predict_proba(test_features)[:, 1]
    acc = accuracy_score(test_y, test_y_pred)
    precision = precision_score(test_y, test_y_pred, average=None)
    recall = recall_score(test_y, test_y_pred, average=None)
    F1 = f1_score(test_y, test_y_pred, average=None)
    auc = roc_auc_score(test_y, test_y_prob)

    print(f"SVM+MLP Accuracy: {acc:.4f}, Precision: {precision}, Recall: {recall}, F1: {F1}, AUC: {auc:.4f}")
    return "SVM+MLP", acc, precision[1], precision[0], recall[1], recall[0], F1[1], F1[0], auc


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = get_train_test_np()
    SVM(train_x, train_y, test_x, test_y)
