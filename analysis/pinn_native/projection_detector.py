# Проекционный детектор fold-точек (PINN-native)
#
# Идея: проверяем, лежит ли r_lambda = dR/dlambda в образе якобиана J_W = dR/dθ.
# Если лежит — точка регулярная, сеть может скомпенсировать изменение lambda.
# Если нет — fold-подобное поведение, сеть "не справляется".
#
# Это аналог критерия Келлера-Антмана, перенесённый в пространство параметров сети.
#
# Score: eta = ||(I - P) r_lambda|| / ||r_lambda||
#   eta ≈ 0  — регулярная точка
#   eta >> 0 — кандидат на fold
#
# Два варианта проектора P:
#   TSVD — жесткое обрезание малых сингулярных чисел (чувствителен к порогу)
#   Tikhonov  — плавная фильтрация, более устойчив на практике

from __future__ import annotations
import logging
from statistics import median
from typing import Dict, Any, List
import torch
import torch.nn as nn
from problems.base_problem import BaseProblem
from analysis.pinn_native.surrogate_jacobian import jacobian_wrt_parameters, jacobian_wrt_lambda
logger = logging.getLogger(__name__)


# порог для определения численного ранга
def _svd_tolerance(s: torch.Tensor, m: int, p: int, tol_factor: float = 10.0) -> float:
    if s.numel() == 0:
        return 0.0
    eps = torch.finfo(s.dtype).eps
    return float(tol_factor * max(m, p) * float(s[0]) * eps)


# адаптивный выбор ранга для TSVD
def _choose_tsvd_rank(s: torch.Tensor, m: int, p: int, tol_factor: float = 10.0,
                      energy_frac: float = 0.999, max_rank: int | None = None) -> int:
    if s.numel() == 0:
        return 0

    tol = _svd_tolerance(s, m, p, tol_factor)
    numerical_rank = int((s > tol).sum())
    numerical_rank = max(numerical_rank, 1)

    # кумулятивная энергия сингулярных чисел
    s_eff = s[:numerical_rank]
    energy = torch.cumsum(s_eff.square(), dim=0)
    total = float(energy[-1])
    if total <= 0.0:
        r = 1
    else:
        target = energy_frac * total
        r = int(torch.searchsorted(energy, torch.tensor(target, dtype=energy.dtype, device=energy.device)).item()) + 1

    r = min(r, numerical_rank)
    if max_rank is not None:
        r = min(r, max_rank)
    return max(r, 1)


# основная функция: считает оба варианта проекционного score для одного lambda
def compute_projection_diagnostics(problem: BaseProblem, model: nn.Module, lam: torch.Tensor,
    interior_pts: torch.Tensor, boundary_pts: torch.Tensor,
    bc_weight: float = 1.0, tol_factor: float = 10.0,
    energy_frac: float = 0.999, max_rank: int | None = None,
    alpha_rel: float = 1e-4) -> Dict[str, Any]:
    # строим J_W и r_lambda
    J = jacobian_wrt_parameters(
        problem, model, lam, interior_pts, boundary_pts, bc_weight=bc_weight,
    ).double()
    r_lambda = jacobian_wrt_lambda(
        problem, model, lam, interior_pts, boundary_pts, bc_weight=bc_weight,
    ).reshape(-1, 1).double()

    m, p = J.shape
    # SVD якобиана — ключевая операция, делаем в float64 для устойчивости
    U, s, _Vh = torch.linalg.svd(J, full_matrices=False)
    q = s.numel()

    tol = _svd_tolerance(s, m, p, tol_factor)
    numerical_rank = int((s > tol).sum()) if q > 0 else 0
    tsvd_rank = _choose_tsvd_rank(
        s, m, p, tol_factor=tol_factor, energy_frac=energy_frac, max_rank=max_rank,
    ) if q > 0 else 0

    r_norm = float(r_lambda.norm()) + 1e-30

    # TSVD-проектор
    # P_TSVD = U_r @ U_r^T, где U_r — первые r столбцов U
    if tsvd_rank > 0:
        U_r = U[:, :tsvd_rank]
        proj_tsvd = U_r @ (U_r.T @ r_lambda)
    else:
        proj_tsvd = torch.zeros_like(r_lambda)
    resid_tsvd = r_lambda - proj_tsvd
    eta_tsvd = float(resid_tsvd.norm()) / r_norm

    # Тихоновский проектор
    # P_alpha = U @ diag(σ²/(σ² + α²)) @ U^T
    # фильтр-факторы φ_i = σ_i² / (σ_i² + α²) плавно подавляют малые σ_i
    sigma_max = float(s[0]) if q > 0 else 0.0
    alpha_abs = alpha_rel * sigma_max  # α пропорционален максимальному σ
    if q > 0:
        coeffs = U.T @ r_lambda
        filt = (s.square() / (s.square() + alpha_abs * alpha_abs)).reshape(-1, 1)
        proj_tikh = U @ (filt * coeffs)
    else:
        proj_tikh = torch.zeros_like(r_lambda)
    resid_tikh = r_lambda - proj_tikh
    eta_tikhonov = float(resid_tikh.norm()) / r_norm

    # вспомогательная диагностика: зазор между удержанными и отброшенными σ
    sigma_r = float(s[tsvd_rank - 1]) if q > 0 and tsvd_rank >= 1 else float("nan")
    sigma_rp1 = float(s[tsvd_rank]) if q > 0 and tsvd_rank < q else 0.0
    gap_after_r = sigma_r / (sigma_rp1 + 1e-30) if q > 0 and tsvd_rank >= 1 else float("nan")

    # stable rank — эффективная размерность (||J||_F² / σ_1²)
    stable_rank = float(J.norm().square() / (s[0].square() + 1e-30)) if q > 0 else 0.0

    return {
        "J": J,
        "r_lambda": r_lambda,
        "singular_values": s,
        "tol": tol,
        "numerical_rank": numerical_rank,
        "tsvd_rank": tsvd_rank,
        "sigma_max": sigma_max,
        "sigma_r": sigma_r,
        "sigma_rp1": sigma_rp1,
        "gap_after_r": gap_after_r,
        "stable_rank": stable_rank,
        "alpha_abs": alpha_abs,
        "eta_tsvd": eta_tsvd,
        "eta_tikhonov": eta_tikhonov,
        "residual_norm_tsvd": float(resid_tsvd.norm()),
        "residual_norm_tikhonov": float(resid_tikh.norm()),
        "r_lambda_norm": r_norm,
        "projection_norm_tsvd": float(proj_tsvd.norm()),
        "projection_norm_tikhonov": float(proj_tikh.norm()),
    }


# детекция кандидатов по истории: сравниваем текущий "эту" с baseline и недавней медианой
def detect_projection_candidate(diagnostics_history: List[Dict[str, Any]],
                                current_idx: int, score_key: str = "eta_tikhonov",
                                warmup_steps: int = 5, abs_threshold: float = 0.15,
                                baseline_multiplier: float = 5.0, recent_multiplier: float = 2.0) -> Dict[str, Any]:
    score = float(diagnostics_history[current_idx][score_key])
    # первые шаги — разогрев, не детектим (нужна статистика)
    if current_idx + 1 < warmup_steps:
        return {
            "is_candidate": False,
            "reason": f"warmup<{warmup_steps}",
            "score": score,
            "baseline": None,
            "recent_median": None,
        }

    # baseline — медиана первых warmup_steps значений η
    baseline_vals = [float(d[score_key]) for d in diagnostics_history[:warmup_steps]]
    baseline = median(baseline_vals)

    # recent — медиана последних warmup_steps значений
    lo = max(0, current_idx - warmup_steps + 1)
    recent_vals = [float(d[score_key]) for d in diagnostics_history[lo:current_idx + 1]]
    recent_med = median(recent_vals)

    # три способа задетектить: абсолютный порог, рост относительно baseline, рост относительно recent
    reasons: List[str] = []
    if score > abs_threshold:
        reasons.append(f"{score_key}={score:.3e} > abs={abs_threshold:.3e}")
    if baseline > 0 and score > baseline_multiplier * baseline:
        reasons.append(f"{score_key}/baseline={score / baseline:.2f} > {baseline_multiplier:.2f}")
    if recent_med > 0 and score > recent_multiplier * recent_med:
        reasons.append(f"{score_key}/recent={score / recent_med:.2f} > {recent_multiplier:.2f}")

    return {
        "is_candidate": len(reasons) > 0,
        "reason": " | ".join(reasons) if reasons else "none",
        "score": score,
        "baseline": baseline,
        "recent_median": recent_med,
    }


# простая классификация: большой η -> fold, маленький -> не затетекчено
def classify_projection_candidate(projection_info: Dict[str, Any], eta_threshold: float = 0.15) -> Dict[str, Any]:
    eta = float(projection_info["eta_tikhonov"])
    if eta > eta_threshold:
        label = "candidate_limit_point"
    else:
        label = "regular_or_not_detected"

    return {
        "label": label,
        "eta_tikhonov": eta,
        "eta_tsvd": float(projection_info["eta_tsvd"]),
        "tsvd_rank": int(projection_info["tsvd_rank"]),
        "numerical_rank": int(projection_info["numerical_rank"]),
        "gap_after_r": float(projection_info["gap_after_r"]),
    }