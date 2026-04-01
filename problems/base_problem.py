# абстрактный класс, подходящий под разные УЧП
from abc import ABC, abstractmethod
import torch


class BaseProblem(ABC):

    @abstractmethod
    def sample_interior_fixed(self, n: int, device: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_boundary_fixed(self, n_per_side: int, device: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def residual_interior(self, model: torch.nn.Module, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def residual_boundary(self, model: torch.nn.Module, xb: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def branch_observable(self, model: torch.nn.Module, device: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def problem_name(self) -> str:
        raise NotImplementedError
