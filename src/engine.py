"""Funcoes de treino e avaliacao (uma epoca de treino / calculo das metricas)."""
import matplotlib

matplotlib.use("Agg")  # gera imagens sem abrir janela
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.data import CLASSES


def make_loader(X, y, batch_size, shuffle, seed):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for X, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Retorna (dicionario de metricas, matriz de confusao)."""
    model.eval()
    total, y_true, y_pred = 0.0, [], []
    for X, y in loader:
        logits = model(X)
        total += criterion(logits, y).item() * len(X)
        y_pred += (torch.sigmoid(logits) >= 0.5).int().squeeze(1).tolist()
        y_true += y.int().squeeze(1).tolist()
    metrics = {
        "loss": total / len(loader.dataset),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return metrics, confusion_matrix(y_true, y_pred, labels=[0, 1])


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            cor = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14, color=cor)
    fig.tight_layout()
    return fig