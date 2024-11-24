# -*- coding: utf-8 -*-

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.nn import GATConv
from dataset.ML.loader import get_train_test_np, get_edge_index
from utils.config_utils import get_config_option


class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)


def train_gat(model, train_x, train_y, edge_index, epochs, learning_rate, device):
    model = model.to(device)
    train_x, train_y, edge_index = train_x.to(device), train_y.to(device), edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_x, edge_index)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()


def test_gat(model, test_x, test_y, edge_index, device):
    model = model.to(device)
    test_x, test_y, edge_index = test_x.to(device), test_y.to(device), edge_index.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(test_x, edge_index)
        predictions = outputs.argmax(dim=1)
        probabilities = outputs[:, 1]  # Assuming binary classification
    metrics = evaluate_metrics(test_y.cpu().numpy(), predictions.cpu().numpy(), probabilities.cpu().numpy())
    print_metrics("GAT", metrics)
    return metrics


def evaluate_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=None, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=None, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }
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


if __name__ == "__main__":
    config = {
        "input_dim": int(get_config_option("GNN", "GNN", "n_features")),
        "hidden_dim": int(get_config_option("GNN", "GNN", "gnns_forward_hidden").split()[0]),
        "output_dim": int(get_config_option("GNN", "GNN", "n_classes")),
        "epochs": int(get_config_option("GNN", "GNN", "epochs")),
        "learning_rate": float(get_config_option("GNN", "GNN", "lr0")),
    }

    train_x, train_y, test_x, test_y = get_train_test_np()
    edge_index = get_edge_index()

    if isinstance(edge_index, list):
        edge_index = torch.cat([torch.tensor(e, dtype=torch.long) for e in edge_index], dim=1)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GATModel(config["input_dim"], config["hidden_dim"], config["output_dim"])
    train_gat(
        model,
        torch.tensor(train_x, dtype=torch.float32).to(device),
        torch.tensor(train_y, dtype=torch.long).to(device),
        edge_index.to(device),
        config["epochs"],
        config["learning_rate"],
        device
    )

    test_gat(
        model,
        torch.tensor(test_x, dtype=torch.float32).to(device),
        torch.tensor(test_y, dtype=torch.long).to(device),
        edge_index.to(device),
        device
    )