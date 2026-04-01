# проходим по сетке lambda с warm-start обучением
import copy
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from problems.base_problem import BaseProblem
from continuation.warmstart_trainer import train_fixed_lambda
from pinn.residual_vector import build_scalar_loss
from utils.config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class BranchPoint:
    step: int
    lam: float
    observable_center: float  # u(0.5, 0.5)
    observable_l2: float  # ||u||
    loss_total: float
    residual_mse_eval: float
    sigma_min: Optional[float]  # заполняется позже, при SVD-анализе
    sigma_second: Optional[float]
    rank_est: Optional[int]
    candidate_type: Optional[str] # regular_point / candidate_limit_point / ...
    state_dict: dict = field(repr=False)


# основной цикл: для каждого lambda обучаем PINN и сохраняем точку ветви
def trace_branch(problem: BaseProblem, model: nn.Module, lam_values: List[float], train_cfg: TrainConfig, device: str = "cpu") -> List[BranchPoint]:
    """Trace a solution branch over a list of lambda values via warm-start.

    At each step: train at fixed lambda, evaluate observables, save snapshot.
    SVD/Frechet analysis is done separately (see run_bratu_detector.py).
    """
    branch: List[BranchPoint] = []

    for step, lam in enumerate(lam_values):
        logger.info("=== Step %d/%d  λ=%.4f ===", step, len(lam_values) - 1, lam)

        # обучаем при данном lambda (warm-start: веса от предыдущего шага)
        train_result = train_fixed_lambda(problem, model, lam, train_cfg, device)

        # дополнительно оцениваем невязку на отдельном наборе точек
        model.eval()
        lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
        int_eval = problem.sample_interior_fixed(200, device)
        bnd_eval = problem.sample_boundary_fixed(50, device)
        with torch.enable_grad():
            loss_eval, _, _ = build_scalar_loss(
                problem, model, lam_t, int_eval, bnd_eval, bc_weight=train_cfg.bc_weight,
            )

        # sigma_min и candidate_type пока None — заполним при анализе Фреше
        branch.append(BranchPoint(
            step=step,
            lam=lam,
            observable_center=train_result["observable_center"],
            observable_l2=train_result["observable_l2"],
            loss_total=train_result["loss_total"],
            residual_mse_eval=float(loss_eval.item()),
            sigma_min=None,
            sigma_second=None,
            rank_est=None,
            candidate_type=None,
            state_dict=copy.deepcopy(model.state_dict()),
        ))

    return branch