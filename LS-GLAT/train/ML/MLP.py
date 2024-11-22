import torch
import torch.nn as nn
import torch.optim as optim

class MLPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.mlp(x))

def train_mlp(model: MLPModel, train_x, train_y, epochs: int, learning_rate: float, device: torch.device):
    train_x, train_y = train_x.to(device), train_y.to(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Learning Rate: {current_lr}")
    return model

def test_mlp(model: MLPModel, test_x, test_y, device: torch.device):
    test_x, test_y = test_x.to(device), test_y.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(test_x)
        _, predictions = torch.max(outputs, 1)
    predictions = predictions.cpu().numpy()
    test_y = test_y.cpu().numpy()
    probabilities = outputs[:, 1].cpu().numpy()
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    metrics = {
        "accuracy": accuracy_score(test_y, predictions),
        "precision": precision_score(test_y, predictions, average=None),
        "recall": recall_score(test_y, predictions, average=None),
        "f1_score": f1_score(test_y, predictions, average=None),
        "auc": roc_auc_score(test_y, probabilities)
    }
    return metrics