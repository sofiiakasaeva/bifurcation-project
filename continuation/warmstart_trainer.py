# Обучение PINN при фиксированном lambda (с поддержкой warm-start)
import logging
import time
import torch
import torch.nn as nn
from problems.base_problem import BaseProblem
from pinn.residual_vector import build_scalar_loss
from utils.config import TrainConfig

logger = logging.getLogger(__name__)


# основная функция обучения: если модель уже имеет веса, получается warm-start
def train_fixed_lambda(problem: BaseProblem, model: nn.Module, lam: float, train_cfg: TrainConfig, device: str) -> dict:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    # фиксированные точки коллокации (не пересэмплируем каждую эпоху)
    interior_pts = problem.sample_interior_fixed(train_cfg.n_int_train, device)
    boundary_pts = problem.sample_boundary_fixed(train_cfg.n_bnd_train, device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)

    t0 = time.time()
    for epoch in range(1, train_cfg.epochs + 1):
        optimizer.zero_grad()
        # считаем L = L_PDE + w_BC * L_BC
        loss_total, loss_pde, loss_bc = build_scalar_loss(
            problem, model, lam_t, interior_pts, boundary_pts, bc_weight=train_cfg.bc_weight,
        )
        loss_total.backward()
        optimizer.step()
        if epoch % train_cfg.log_every == 0 or epoch == train_cfg.epochs:
            logger.debug("λ=%.4f  epoch %d/%d  loss=%.4e  pde=%.4e  bc=%.4e",
                         lam, epoch, train_cfg.epochs, loss_total.item(), loss_pde.item(), loss_bc.item())

    # после обучения считаем наблюдаемые: u(0.5, 0.5) и ||u||
    model.eval()
    obs = problem.branch_observable(model, device)
    logger.info("λ=%.4f  done in %.1fs  loss=%.3e  u_center=%.4f",
                lam, time.time() - t0, loss_total.item(), obs["center"])
    return {
        "lam": lam,
        "loss_total": float(loss_total.item()),
        "loss_pde": float(loss_pde.item()),
        "loss_bc": float(loss_bc.item()),
        "observable_center": obs["center"],
        "observable_l2": obs["l2"],
        "train_time_s": time.time() - t0,
    }