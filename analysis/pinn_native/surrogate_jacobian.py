# якобиан J_W = dR/dθ для проекционного детектора
# Этот модуль строит якобиан невязки по параметрам сети,
# а не по физическим переменным. Используется в projection_detector.py
# для проверки принадлежности r_lambda образу J_W.
#
# Важно: J_W сам по себе rank-deficient когда m < p (точек меньше чем параметров),
# поэтому sigma_min(J_W) как самостоятельный критерий не работает.
# Но для проекционного критерия это не проблема — нам нужен только проектор на Im(J_W).

import time
import logging
import torch
import torch.nn as nn
from problems.base_problem import BaseProblem
from pinn.residual_vector import build_residual_vector

logger = logging.getLogger(__name__)


# J_W = dR/dθ, считаем построчно через autograd
def jacobian_wrt_parameters(problem: BaseProblem, model: nn.Module, lam: torch.Tensor,
                            interior_pts: torch.Tensor, boundary_pts: torch.Tensor, bc_weight: float = 1.0) -> torch.Tensor:
    model.train()
    lam_d = lam.detach().requires_grad_(False)
    R = build_residual_vector(problem, model, lam_d, interior_pts, boundary_pts, bc_weight)
    params = [p for p in model.parameters() if p.requires_grad]
    m = R.numel()
    rows = []
    for i in range(m):
        # градиент i-й компоненты невязки по всем параметрам сети
        grads = torch.autograd.grad(
            R[i], params,
            retain_graph=(i < m - 1),
            create_graph=False,
            allow_unused=False,
        )
        rows.append(torch.cat([g.reshape(-1) for g in grads]).detach())

    J = torch.stack(rows)
    return J


# r_lambda = dR/dlambda, считаем конечной разностью
# (можно было бы через autograd, но lambda входит линейно, так что разность точна)
def jacobian_wrt_lambda(problem: BaseProblem, model: nn.Module, lam: torch.Tensor,
                        interior_pts: torch.Tensor, boundary_pts: torch.Tensor,
                        bc_weight: float = 1.0, fd_eps: float = 1e-4) -> torch.Tensor:
    lam_val = lam.item()
    lam_p = torch.tensor(lam_val + fd_eps, dtype=lam.dtype, device=lam.device)
    lam_m = torch.tensor(lam_val,          dtype=lam.dtype, device=lam.device)
    # нужен enable_grad, потому что внутри residual считается лапласиан через autograd
    with torch.enable_grad():
        R_p = build_residual_vector(problem, model, lam_p, interior_pts, boundary_pts, bc_weight)
        R_m = build_residual_vector(problem, model, lam_m, interior_pts, boundary_pts, bc_weight)
    return ((R_p - R_m) / fd_eps).unsqueeze(-1).detach()


# классификация через расширенную матрицу [J_W | r_lambda]
# аналог критерия Келлера-Антмана, но в пространстве параметров сети
def classify_candidate(J: torch.Tensor, r_lambda: torch.Tensor, tol: float, ls_residual_threshold: float = 0.1) -> dict:
    r_lam_vec = r_lambda.reshape(-1, 1).to(dtype=J.dtype)

    sv_J = torch.linalg.svdvals(J)
    rank_J = int((sv_J > tol).sum())
    sigma_min_J = float(sv_J[-1]) if sv_J.numel() > 0 else float("nan")

    # расширенная матрица: если ранг вырос, значит r_lambda не в образе
    J_star = torch.cat([J, r_lam_vec], dim=1)
    rank_Jstar = int((torch.linalg.svdvals(J_star) > tol).sum())

    # дополнительная проверка: решаем J @ a ≈ r_lambda методом наименьших квадратов
    a, ls_res, _, _ = torch.linalg.lstsq(J, r_lam_vec, rcond=None)
    r_lam_norm = float(r_lam_vec.norm())
    abs_res = float((J @ a - r_lam_vec).norm()) if ls_res.numel() == 0 else float(ls_res[0].sqrt())
    rel_res = abs_res / (r_lam_norm + 1e-30)

    if sigma_min_J >= tol:
        label = "regular_point"
    elif rank_Jstar > rank_J or rel_res > ls_residual_threshold:
        label = "candidate_limit_point"  # r_lambda не в образе J
    else:
        label = "candidate_bifurcation_point"

    return {
        "label": label,
        "sigma_min_J": sigma_min_J,
        "rank_J": rank_J,
        "rank_Jstar": rank_Jstar,
        "ls_relative_residual": rel_res,
    }