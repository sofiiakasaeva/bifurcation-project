# строим резидуал векторы и loss
import math
import torch
import torch.nn as nn
from problems.base_problem import BaseProblem


def build_residual_vector(problem: BaseProblem, model: nn.Module, lam: torch.Tensor,
    interior_pts: torch.Tensor, boundary_pts: torch.Tensor, bc_weight: float = 1.0) -> torch.Tensor:
    r_int = problem.residual_interior(model, interior_pts, lam).reshape(-1)
    r_bc  = problem.residual_boundary(model, boundary_pts, lam).reshape(-1)
    return torch.cat([r_int, math.sqrt(bc_weight) * r_bc], dim=0)


def build_scalar_loss(problem: BaseProblem, model: nn.Module, lam: torch.Tensor,
    interior_pts: torch.Tensor, boundary_pts: torch.Tensor, bc_weight: float = 1.0) -> tuple:
    r_int = problem.residual_interior(model, interior_pts, lam).reshape(-1)
    r_bc  = problem.residual_boundary(model, boundary_pts, lam).reshape(-1)
    loss_pde = r_int.pow(2).mean()
    loss_bc  = r_bc.pow(2).mean()
    return loss_pde + bc_weight * loss_bc, loss_pde, loss_bc
