# -*- coding: utf-8 -*-

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dataset.ML.loader import get_train_test_np
from utils.config_utils import get_config_option
from MLP import MLPModel, train_mlp, test_mlp


def load_config():
    config = {
        "seed": int(get_config_option("GNN", "GNN", "seed")),
        "epochs": int(get_config_option("GNN", "GNN", "epochs")),
        "learning_rate": float(get_config_option("GNN", "GNN", "lr0")),
        "hidden_dim": int(get_config_option("GNN", "GNN", "linears_hidden").split()[0]),
        "output_dim": int(get_config_option("GNN", "GNN", "n_classes")),
        "input_dim": int(get_config_option("GNN", "GNN", "n_features")),
    }
    return config


def run_logistic_regression(train_x, train_y, test_x, test_y, config):
    solver = "liblinear"

    lr = LogisticRegression(
        solver=solver,
        random_state=config["seed"],
        max_iter=config["epochs"],
    )
    lr.fit(train_x, train_y)

    test_y_pred = lr.predict(test_x)
    metrics = evaluate_metrics(test_y, test_y_pred, lr.predict_proba(test_x)[:, 1])
    print_metrics("Logistic Regression", metrics)
    return metrics


def run_mlp(train_x, train_y, test_x, test_y, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, train_y = torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long)
    test_x, test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long)

    model = MLPModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
    )

    train_mlp(model, train_x, train_y, config["epochs"], config["learning_rate"], device)
    metrics = test_mlp(model, test_x, test_y, device)
    print_metrics("LR + MLP", metrics)
    return metrics

def print_metrics(model_name, metrics):
    print(
        f"{model_name}: "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Precision={metrics['precision'][0]:.4f}/{metrics['precision'][1]:.4f}, "
        f"Recall={metrics['recall'][0]:.4f}/{metrics['recall'][1]:.4f}, "
        f"F1={metrics['f1_score'][0]:.4f}/{metrics['f1_score'][1]:.4f}, "
        f"AUC={metrics['auc']:.4f}"
    )

def evaluate_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=None),
        "recall": recall_score(y_true, y_pred, average=None),
        "f1_score": f1_score(y_true, y_pred, average=None),
        "auc": roc_auc_score(y_true, y_prob),
    }
    return metrics

if __name__ == '__main__':
    config = load_config()
    train_x, train_y, test_x, test_y = get_train_test_np()

    run_logistic_regression(train_x, train_y, test_x, test_y, config)

    run_mlp(train_x, train_y, test_x, test_y, config)