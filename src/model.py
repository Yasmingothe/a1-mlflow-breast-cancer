"""Modelo: MLP (rede totalmente conectada) para classificacao binaria."""
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_features, hidden_layers):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # 1 saida (logit): maligno ou nao
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)