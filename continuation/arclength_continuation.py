# Pseudo-arclength/norm continuation: trace a solution branch through folds
import copy
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import torch
import torch.nn as nn
from problems.base_problem import BaseProblem
from pinn.residual_vector import build_scalar_loss
from utils.config import ContinuationConfig, TrainConfig
from continuation.warmstart_trainer import train_fixed_lambda

logger = logging.getLogger(__name__)


@dataclass
class ContinuationPoint:
    step: int
    lam: float
    norm_u: float
    observable_center: float
    observable_l2: float
    loss_total: float
    loss_pde: float = 0.0
    state_dict: dict = field(default=None, repr=False)
    sigma_min: Optional[float] = None
    sigma_second: Optional[float] = None
    candidate_type: Optional[str] = None
    corank: Optional[int] = None

# ||u||
def compute_solution_norm(problem: BaseProblem, model: nn.Module, device: str, n_grid: int = 400) -> torch.Tensor:
    pts = problem.sample_interior_fixed(n_grid, device)
    u_vals = model(pts).squeeze(-1)
    return u_vals.pow(2).mean().sqrt()

# предиктим веса
def _extrapolate_state_dicts(sd1: dict, sd2: dict) -> dict:
    sd_new = {}
    for key in sd2:
        v1 = sd1[key].float()
        v2 = sd2[key].float()
        sd_new[key] = (2.0 * v2 - v1).to(sd2[key].dtype)
    return sd_new

# защита для предиктора
def _has_nan(state_dict: dict) -> bool:
    return any(v.isnan().any() or v.isinf().any()
               for v in state_dict.values() if v.is_floating_point())

# критерий остановки
def _detect_oscillation(branch: list, window: int = 12) -> bool:
    if len(branch) < window + 5:
        return False
    recent = [bp.lam for bp in branch[-window:]]
    earlier = [bp.lam for bp in branch[-window * 2:-window]] if len(branch) >= window * 2 else [branch[0].lam]
    lam_range = max(recent) - min(recent)
    movement = abs(sum(recent) / len(recent) - sum(earlier) / len(earlier))
    return lam_range < 0.15 and movement < 0.1


def train_continuation_step(problem: BaseProblem, model: nn.Module, lam_param: torch.Tensor, prev_norm: float,
                            prev_lam: float, cfg: ContinuationConfig, device: str, adaptive_epochs: Optional[int] = None) -> dict:
    model.train()
    n_epochs = adaptive_epochs or cfg.epochs_per_step
    # лямбда тоже обучаемая!
    all_params = list(model.parameters()) + [lam_param]
    optimizer = torch.optim.Adam(all_params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=cfg.lr * 0.05)

    interior_pts = problem.sample_interior_fixed(cfg.n_int_train, device)
    boundary_pts = problem.sample_boundary_fixed(cfg.n_bnd_train, device)

    loss = loss_pde = loss_bc = loss_cont = torch.tensor(float("nan"))

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss_total, loss_pde, loss_bc = build_scalar_loss(
            problem, model, lam_param, interior_pts, boundary_pts, bc_weight=cfg.bc_weight,
        )
        current_norm = compute_solution_norm(problem, model, device)
        # дальше у нас два варианта continuation
        # либо простой по разности норм (для Брату подойдет)
        # либо настоящий arclength
        if cfg.beta2 == 0.0:
            loss_cont = (current_norm - (prev_norm + cfg.gamma)).pow(2)
        else:
            term1 = (cfg.beta1 * (current_norm - prev_norm)).pow(2)
            term2 = (cfg.beta2 * (lam_param - prev_lam)).pow(2)
            loss_cont = (torch.sqrt(term1 + term2 + 1e-12) - cfg.delta).pow(2)

        loss = loss_total + cfg.alpha_cont * loss_cont
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            # ограничиваем параметр
            lam_param.clamp_(cfg.lam_clamp_min, cfg.lam_clamp_max)

        if epoch % max(n_epochs // 3, 1) == 0 or epoch == n_epochs:
            logger.debug(
                "  epoch %d/%d  loss=%.4e  pde=%.4e  cont=%.4e  λ=%.4f  ‖u‖=%.4f",
                epoch, n_epochs, loss.item(), loss_pde.item(), loss_cont.item(),
                lam_param.item(), current_norm.item(),
            )

    model.eval()
    obs = problem.branch_observable(model, device)
    return {
        "lam": float(lam_param.item()),
        "norm_u": float(current_norm.item()),
        "observable_center": obs["center"],
        "observable_l2": obs["l2"],
        "loss_total": float(loss.item()),
        "loss_pde": float(loss_pde.item()),
        "loss_bc": float(loss_bc.item()),
    }


def run_arclength_continuation(problem: BaseProblem, model: nn.Module, lam_start: float, cfg: ContinuationConfig,
                               device: str = "cpu", on_step_done: Optional[Callable] = None) -> List[ContinuationPoint]:
    branch: List[ContinuationPoint] = []
    init_train_cfg = TrainConfig(
        epochs=cfg.epochs_per_step, lr=cfg.lr, bc_weight=cfg.bc_weight,
        n_int_train=cfg.n_int_train, n_bnd_train=cfg.n_bnd_train, log_every=cfg.epochs_per_step,
    )

    # ищем первую точку сети (нам нужно две, чтобы начать наше continuation)
    logger.info("Seeding: point 0, λ=%.4f", lam_start)
    res0 = train_fixed_lambda(problem, model, lam_start, init_train_cfg, device)
    norm0 = compute_solution_norm(problem, model, device).item()
    branch.append(ContinuationPoint(
        step=0, lam=lam_start, norm_u=norm0,
        observable_center=res0["observable_center"], observable_l2=res0["observable_l2"],
        loss_total=res0["loss_total"], loss_pde=res0.get("loss_pde", 0.0),
        state_dict=copy.deepcopy(model.state_dict()),
    ))
    logger.info("  λ=%.4f  ‖u‖=%.4f  u_center=%.4f", lam_start, norm0, res0["observable_center"])
    if on_step_done:
        on_step_done(branch, 0)

    # ищем вторую точку сети с шагом lam_init_step
    lam1 = float(max(cfg.lam_clamp_min, min(lam_start + cfg.lam_init_step, cfg.lam_clamp_max)))
    logger.info("Seeding: point 1, λ=%.4f", lam1)
    res1 = train_fixed_lambda(problem, model, lam1, init_train_cfg, device)
    norm1 = compute_solution_norm(problem, model, device).item()
    branch.append(ContinuationPoint(
        step=1, lam=lam1, norm_u=norm1,
        observable_center=res1["observable_center"], observable_l2=res1["observable_l2"],
        loss_total=res1["loss_total"], loss_pde=res1.get("loss_pde", 0.0),
        state_dict=copy.deepcopy(model.state_dict()),
    ))
    logger.info("  λ=%.4f  ‖u‖=%.4f  u_center=%.4f", lam1, norm1, res1["observable_center"])
    if on_step_done:
        on_step_done(branch, 1)

    passed_fold = False # прошли ли мы fold

    for step in range(2, cfg.max_steps):
        logger.info("=== Step %d ===", step)
        # экстраполяция весов
        sd_init = _extrapolate_state_dicts(branch[-2].state_dict, branch[-1].state_dict)
        if _has_nan(sd_init):
            logger.warning("NaN in extrapolation, going back to previous weights")
            sd_init = copy.deepcopy(branch[-1].state_dict)
        model.load_state_dict(sd_init)
        lam_init = 2.0 * branch[-1].lam - branch[-2].lam
        lam_init = float(max(cfg.lam_clamp_min, min(lam_init, cfg.lam_clamp_max)))
        lam_param = torch.tensor(lam_init, dtype=torch.float32, device=device, requires_grad=True)

        # детектим fold
        # если до этого lambda росла, а теперь падает => turning point
        if len(branch) >= 3 and not passed_fold:
            d1 = branch[-1].lam - branch[-2].lam
            d2 = branch[-2].lam - branch[-3].lam
            if d1 * d2 < 0:
                passed_fold = True
                logger.info("  *** FOLD: λ reversed (%.4f → %.4f → %.4f)",
                            branch[-3].lam, branch[-2].lam, branch[-1].lam)

        # адаптивные эпохи
        prev_loss_pde = branch[-1].loss_pde
        if passed_fold and prev_loss_pde > 0.01:
            # теперь их больше, так как верхняя ветвь ищется сложнее
            adaptive_epochs = min(cfg.epochs_per_step * 2, 6000)
        else:
            adaptive_epochs = cfg.epochs_per_step

        logger.info("  predictor λ=%.4f  ‖u‖_prev=%.4f  epochs=%d",
                    lam_init, branch[-1].norm_u, adaptive_epochs)

        result = train_continuation_step(
            problem, model, lam_param,
            prev_norm=branch[-1].norm_u, prev_lam=branch[-1].lam,
            cfg=cfg, device=device, adaptive_epochs=adaptive_epochs,
        )

        bp = ContinuationPoint(
            step=step, lam=result["lam"], norm_u=result["norm_u"],
            observable_center=result["observable_center"], observable_l2=result["observable_l2"],
            loss_total=result["loss_total"], loss_pde=result.get("loss_pde", 0.0),
            state_dict=copy.deepcopy(model.state_dict()),
        )
        branch.append(bp)

        logger.info("  λ=%.4f  ‖u‖=%.4f  u_center=%.4f  loss=%.3e  pde=%.3e",
                    bp.lam, bp.norm_u, bp.observable_center, bp.loss_total, bp.loss_pde)

        if on_step_done:
            on_step_done(branch, step)

        # останавлияваемся если норма разрастается
        if bp.norm_u > cfg.norm_target_max:
            logger.info("  ‖u‖=%.4f > max %.1f — stopping.", bp.norm_u, cfg.norm_target_max)
            break
        # останавливаемся если осцилляции
        if passed_fold and _detect_oscillation(branch, window=15):
            logger.info("  Oscillation detected — stopping at step %d.", step)
            break

    logger.info("Done: %d points, λ ∈ [%.4f, %.4f], fold_passed=%s",
                len(branch), min(bp.lam for bp in branch), max(bp.lam for bp in branch), passed_fold)
    return branch
