# вспомогательные функции
import torch
import torch.nn as nn
from typing import List, Tuple


def get_trainable_params(model: nn.Module) -> List[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params])


def param_shapes(model: nn.Module) -> List[Tuple[int, ...]]:
    return [p.shape for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
