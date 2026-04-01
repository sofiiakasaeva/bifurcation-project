# Physical Frechet operator for the Bratu PDE
import torch
import torch.nn as nn
from typing import Dict, Any

# Строим дискретный Лапласиан (оператор превращаем в матрицу)
# 5-точечная аппроксимация, так что получаем матрицу [N² × N²]
def _build_fd_laplacian(N: int, h: float, device: str) -> torch.Tensor:
    m = N * N
    L = torch.zeros(m, m, device=device, dtype=torch.float64)
    inv_h2 = 1.0 / (h * h)
    for i in range(N):
        for j in range(N):
            k = i * N + j  # current node
            L[k, k] = -4.0 * inv_h2
            if j + 1 < N:
                L[k, k + 1] = inv_h2
                L[k + 1, k] = inv_h2
            if i + 1 < N:
                L[k, k + N] = inv_h2
                L[k + N, k] = inv_h2
    return L

# Строим матрицу Фреше
def build_frechet_matrix(model: nn.Module, lam: float, N: int, device: str) -> Dict[str, Any]:
    """Build F = Δ_FD + λ·diag(exp(u*)) on an N×N interior grid.

    Returns dict with:
        F         : [N², N²] Frechet matrix
        f_lambda  : [N², 1]  ∂R/∂λ = exp(u*)
        u_vals    : [N²]     PINN solution at grid points
        grid_pts  : [N², 2]  grid coordinates
    """
    h = 1.0 / (N + 1)
    L_fd = _build_fd_laplacian(N, h, device)

    t = torch.linspace(h, 1.0 - h, N, device=device)
    xx, yy = torch.meshgrid(t, t, indexing="ij")  # строим двумерную сетку
    grid_pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    model.eval()
    with torch.no_grad():
        u_vals = model(grid_pts).squeeze(-1).double()

    exp_u = torch.exp(u_vals)
    F = L_fd + lam * torch.diag(exp_u)
    f_lambda = exp_u.unsqueeze(-1) # столбец производных по параметру
    return {"F": F, "f_lambda": f_lambda, "u_vals": u_vals, "grid_pts": grid_pts}

# анализируем сингулярные числа для детекции
def compute_frechet_svd(F: torch.Tensor, tol_factor: float = 10.0) -> Dict[str, Any]:
    sv = torch.linalg.svdvals(F)  # descending
    sv_asc = sv.flip(0)
    m = F.shape[0]
    eps = torch.finfo(F.dtype).eps
    tol = tol_factor * m * float(sv[0]) * eps # оцениваем порог

    return {
        "singular_values": sv,
        "sigma_min": float(sv_asc[0]),
        "sigma_second": float(sv_asc[1]) if sv_asc.numel() > 1 else float("nan"),
        "sigma_max": float(sv[0]),
        "rank_est": int((sv > tol).sum()),
        "corank_est": m - int((sv > tol).sum()),
        "tol": tol,
    }

# классифицируем: предельная точка или бифуркация
def classify_frechet_candidate(F: torch.Tensor, f_lambda: torch.Tensor, tol: float,
    sigma_physical_threshold: float = 1.0, left_nullvec_threshold: float = 0.1,) -> Dict[str, Any]:
    """Classify using the left null-vector of F.

    At a limit point (fold): f_λ ∉ image(F), so the left null-vector w of F
    has large projection onto f_λ: |<w, f_λ>| / ||f_λ|| ≈ 1.

    At a bifurcation point: f_λ ∈ image(F), so |<w, f_λ>| / ||f_λ|| ≈ 0.
    """
    f_lam_vec = f_lambda.reshape(-1).to(dtype=F.dtype)
    U, S, _Vh = torch.linalg.svd(F, full_matrices=True)
    sigma_min_F = float(S[-1])

    w = U[:, -1]  # левый singular vector для sigma_min
    dot = float(torch.dot(w, f_lam_vec).abs())
    left_null_proj = dot / (float(f_lam_vec.norm()) + 1e-30) # проекция на f_lambda (считаем нормированную величину)

    if sigma_min_F >= sigma_physical_threshold:
        label = "regular_point"
    elif left_null_proj > left_nullvec_threshold:
        label = "candidate_limit_point"
    else:
        label = "candidate_bifurcation_point"

    return {
        "label": label,
        "sigma_min_F": sigma_min_F,
        "rank_F": int((S > tol).sum()),
        "left_null_proj": left_null_proj,
    }
