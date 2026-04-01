# 2D Bratu equation
import math
import torch
from .base_problem import BaseProblem


class Bratu2DProblem(BaseProblem):
    def sample_interior_fixed(self, n: int, device: str) -> torch.Tensor:
        side = math.isqrt(n)
        if side * side < n:
            side += 1
        t = torch.linspace(0.0, 1.0, side + 2, device=device)[1:-1]
        xx, yy = torch.meshgrid(t, t, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    def sample_boundary_fixed(self, n_per_side: int, device: str) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, n_per_side + 2, device=device)[1:-1]
        zeros = torch.zeros(n_per_side, device=device)
        ones = torch.ones(n_per_side, device=device)
        return torch.cat([
            torch.stack([t, zeros], dim=1),
            torch.stack([t, ones],  dim=1),
            torch.stack([zeros, t], dim=1),
            torch.stack([ones, t],  dim=1),
        ], dim=0)

    def residual_interior(self, model: torch.nn.Module, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = model(x).squeeze(-1)
        grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(grads[:, 0].sum(), x, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(grads[:, 1].sum(), x, create_graph=True)[0][:, 1]
        return u_xx + u_yy + lam * torch.exp(u)

    def residual_boundary(self, model: torch.nn.Module, xb: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return model(xb).squeeze(-1)

    def branch_observable(self, model: torch.nn.Module, device: str) -> dict:
        center_pt = torch.tensor([[0.5, 0.5]], device=device)
        with torch.no_grad():
            u_center = model(center_pt).item()

        t = torch.linspace(0.0, 1.0, 22, device=device)[1:-1]
        xx, yy = torch.meshgrid(t, t, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        with torch.no_grad():
            l2 = float(model(grid).squeeze(-1).pow(2).mean().sqrt())

        return {"center": float(u_center), "l2": l2}

    def problem_name(self) -> str:
        return "bratu_2d"
