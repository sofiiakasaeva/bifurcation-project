# PINN: fully-connected network with tanh activations
from __future__ import annotations
import torch
import torch.nn as nn
from utils.config import ModelConfig


class PINN(nn.Module):
    def __init__(self, cfg: ModelConfig = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig(**kwargs)
        layers = []
        in_dim = cfg.input_dim
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, cfg.output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
