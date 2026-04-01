# Уточнение положения fold-точки бисекцией по lambda
# Идея: σ_min(F) > порог слева (регулярная), σ_min(F) < порог справа (кандидат).
# Делим отрезок пополам, обучаем PINN в середине, проверяем σ_min(F), сужаем.

import copy
import logging
import torch
import torch.nn as nn

from problems.base_problem import BaseProblem
from analysis.physical.frechet_pde import build_frechet_matrix, compute_frechet_svd
from continuation.warmstart_trainer import train_fixed_lambda
from utils.config import TrainConfig

logger = logging.getLogger(__name__)


def refine_fold_bisection(problem: BaseProblem, model: nn.Module, lam_left: float, lam_right: float,
    sd_left: dict, train_cfg: TrainConfig, device: str, frechet_n: int = 10, sigma_threshold: float = 1.0,
    tol_lam: float = 0.01, max_iter: int = 20) -> dict:
    sigma_min = float("nan")

    for iteration in range(max_iter):
        lam_mid = (lam_left + lam_right) / 2.0

        # проверяем сходимость
        if abs(lam_right - lam_left) < tol_lam:
            logger.info("Bisection converged: Δλ=%.4f < tol=%.4f", abs(lam_right - lam_left), tol_lam)
            break

        logger.info("Iter %d: [%.4f, %.4f] mid=%.4f", iteration + 1, lam_left, lam_right, lam_mid)

        # обучаем PINN в средней точке (warm-start от левого края)
        model.load_state_dict(copy.deepcopy(sd_left))
        bisect_cfg = TrainConfig(
            epochs=min(train_cfg.epochs, 2000),
            lr=train_cfg.lr,
            bc_weight=train_cfg.bc_weight,
            n_int_train=train_cfg.n_int_train,
            n_bnd_train=train_cfg.n_bnd_train,
            log_every=train_cfg.epochs,
        )
        train_fixed_lambda(problem, model, lam_mid, bisect_cfg, device)

        # считаем σ_min(F) в средней точке
        sigma_min = compute_frechet_svd(build_frechet_matrix(model, lam_mid, frechet_n, device)["F"])["sigma_min"]
        logger.info("lam_mid=%.4f  sigma_min=%.4e", lam_mid, sigma_min)

        # сужаем отрезок: если σ_min мал — fold справа, если велик — fold слева
        if sigma_min < sigma_threshold:
            lam_right = lam_mid
        else:
            lam_left = lam_mid
            sd_left = copy.deepcopy(model.state_dict())

    lam_star = (lam_left + lam_right) / 2.0
    logger.info("Fold at λ* ≈ %.4f  interval=[%.4f, %.4f]", lam_star, lam_left, lam_right)

    return {
        "lam_star": lam_star,
        "lam_interval": (lam_left, lam_right),
        "sigma_min_at_fold": sigma_min,
        "iterations": iteration + 1,
    }