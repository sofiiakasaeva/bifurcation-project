"""Projection-based PINN fold detector along a fixed-lambda branch scan
Usage:
    python experiments/run_projection_detector.py
"""
import sys
import os
import json
import logging
import copy
from dataclasses import dataclass, asdict, field
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from problems.bratu_2d import Bratu2DProblem
from pinn.model import PINN
from continuation.warmstart_trainer import train_fixed_lambda
from analysis.physical.frechet_pde import build_frechet_matrix, compute_frechet_svd
from analysis.pinn_native.projection_detector import (
    compute_projection_diagnostics,
    detect_projection_candidate,
    classify_projection_candidate,
)
from utils.config import TrainConfig, ModelConfig
from utils.plotting import _savefig, BRATU_LAMBDA_CRIT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = "cpu"

LAM_START = 0.5
LAM_END = 7.5
LAM_STEP = 0.2

MODEL_CFG = ModelConfig(input_dim=2, hidden_dim=32, num_hidden_layers=3, output_dim=1)
TRAIN_CFG = TrainConfig(
    epochs=4000, lr=1e-3, bc_weight=10.0,
    n_int_train=2000, n_bnd_train=200, log_every=4000,
)

N_INT_SURROGATE = 2500
N_BND_SURROGATE = 200
ENERGY_FRAC = 0.999
ALPHA_REL = 1e-4
WARMUP_STEPS = 5
ETA_ABS_THRESHOLD = 0.15
ETA_BASELINE_MULT = 5.0
ETA_RECENT_MULT = 2.0
ETA_CLASSIFY_THRESHOLD = 0.05

FRECHET_N = 10

RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"


@dataclass
class ProjectionPoint:
    step: int
    lam: float
    observable_center: float
    observable_l2: float
    loss_total: float
    eta_tikhonov: float
    eta_tsvd: float
    sigma_min_F: float
    tsvd_rank: int
    numerical_rank: int
    gap_after_r: float
    is_candidate: bool
    candidate_type: str
    candidate_reason: str
    baseline_eta: float
    recent_eta_median: float
    state_dict: dict = field(repr=False)


def _candidate_indices(points: List[ProjectionPoint]) -> List[int]:
    return [i for i, p in enumerate(points) if p.is_candidate]


def _plot_results(points: List[ProjectionPoint]) -> None:
    lams = np.array([p.lam for p in points])
    eta_tik = np.array([p.eta_tikhonov for p in points])
    eta_tsvd = np.array([p.eta_tsvd for p in points])
    sigma_F = np.array([p.sigma_min_F for p in points])
    tsvd_rank = np.array([p.tsvd_rank for p in points])
    gap = np.array([p.gap_after_r for p in points])
    cand_idx = _candidate_indices(points)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(lams, eta_tik, "b.-", linewidth=1.5, markersize=5, label="η (Tikhonov)")
    ax1.plot(lams, eta_tsvd, "g.--", linewidth=1, markersize=4, label="η (TSVD)")
    if cand_idx:
        ax1.scatter(lams[cand_idx], eta_tik[cand_idx], color="red", zorder=5, s=80, label="candidates")
    ax1.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1, label=f"ref λ* ≈ {BRATU_LAMBDA_CRIT}")
    ax1.set_xlabel("λ")
    ax1.set_ylabel("projection score η")
    ax1.set_title("PINN projection detector")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(lams, sigma_F, "b.-", linewidth=1.5, markersize=5, label="σ_min(F)")
    if cand_idx:
        ax2.scatter(lams[cand_idx], sigma_F[cand_idx], color="red", zorder=5, s=80, label="candidates")
    ax2.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1, label=f"ref λ* ≈ {BRATU_LAMBDA_CRIT}")
    ax2.set_xlabel("λ")
    ax2.set_ylabel("σ_min(F)")
    ax2.set_title("Physical Fréchet reference")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    _savefig(fig, os.path.join(FIGURES_DIR, "projection_detector_main.png"), "projection scores")
    fig, ax = plt.subplots(figsize=(8, 5))
    eta_norm = eta_tik / max(float(eta_tik.max()), 1e-30)
    sigma_norm = sigma_F / max(float(sigma_F.max()), 1e-30)
    ax.plot(lams, eta_norm, "b.-", linewidth=1.5, markersize=5, label="η / η_max")
    ax.plot(lams, 1.0 - sigma_norm, "g.--", linewidth=1, markersize=4, label="1 − σ_min/σ_max")
    if cand_idx:
        ax.scatter(lams[cand_idx], eta_norm[cand_idx], color="red", zorder=5, s=80, label="candidates")
    ax.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1, label=f"ref λ* ≈ {BRATU_LAMBDA_CRIT}")
    ax.set_xlabel("λ")
    ax.set_ylabel("normalised scale")
    ax.set_title("η growth vs Fréchet degeneration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(FIGURES_DIR, "projection_detector_overlay.png"), "overlay")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(lams, tsvd_rank, "b.-", linewidth=1.5, markersize=5)
    if cand_idx:
        ax1.scatter(lams[cand_idx], tsvd_rank[cand_idx], color="red", zorder=5, s=80)
    ax1.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("λ")
    ax1.set_ylabel("TSVD rank")
    ax1.set_title("Adaptive TSVD rank")
    ax1.grid(True, alpha=0.3)

    ax2.plot(lams, gap, "b.-", linewidth=1.5, markersize=5)
    if cand_idx:
        ax2.scatter(lams[cand_idx], gap[cand_idx], color="red", zorder=5, s=80)
    ax2.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("λ")
    ax2.set_ylabel("σ_r / σ_{r+1}")
    ax2.set_title("Singular-value gap after retained rank")
    ax2.grid(True, alpha=0.3)

    _savefig(fig, os.path.join(FIGURES_DIR, "projection_detector_aux.png"), "auxiliary diagnostics")


def _write_report(points: List[ProjectionPoint]) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "projection_detector_report.md")

    candidates = [p for p in points if p.is_candidate]
    lines = [
        "Projection detector report", "",
        "PINN-native score:",
        "eta = ||(I - P) r_lambda|| / ||r_lambda||, where P is a regularised projection onto Im(J_W).",
        "",
        f"Total points: {len(points)}",
        f"Candidates: {len(candidates)}",
        "",
        "| step | lambda | u_center | eta_tikh | eta_tsvd | sigma_min(F) | tsvd_rank | is_candidate | type | reason |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for p in points:
        lines.append(
            f"| {p.step} | {p.lam:.2f} | {p.observable_center:.4f} | {p.eta_tikhonov:.4e} | "
            f"{p.eta_tsvd:.4e} | {p.sigma_min_F:.4e} | {p.tsvd_rank} | "
            f"{'yes' if p.is_candidate else 'no'} | {p.candidate_type} | {p.candidate_reason or '—'} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for d in [RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)

    problem = Bratu2DProblem()
    model = PINN(MODEL_CFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 70)
    logger.info("PROJECTION DETECTOR: J_W image vs r_lambda")
    logger.info("=" * 70)
    logger.info("Model parameters: %d", n_params)

    int_pts = problem.sample_interior_fixed(N_INT_SURROGATE, DEVICE)
    bnd_pts = problem.sample_boundary_fixed(N_BND_SURROGATE, DEVICE)
    logger.info("Fixed collocation for detector: m=%d", int_pts.shape[0] + bnd_pts.shape[0])

    lam_values = list(np.arange(LAM_START, LAM_END + 1e-9, LAM_STEP))
    history = []
    points: List[ProjectionPoint] = []

    for step, lam in enumerate(lam_values):
        logger.info("=== Step %d/%d  lambda=%.4f ===", step, len(lam_values) - 1, lam)

        train_result = train_fixed_lambda(problem, model, lam, TRAIN_CFG, DEVICE)
        lam_t = torch.tensor(lam, dtype=torch.float32, device=DEVICE)

        proj = compute_projection_diagnostics(
            problem, model, lam_t, int_pts, bnd_pts,
            bc_weight=TRAIN_CFG.bc_weight,
            energy_frac=ENERGY_FRAC,
            alpha_rel=ALPHA_REL,
        )
        history.append(proj)

        det = detect_projection_candidate(
            history, len(history) - 1,
            score_key="eta_tikhonov",
            warmup_steps=WARMUP_STEPS,
            abs_threshold=ETA_ABS_THRESHOLD,
            baseline_multiplier=ETA_BASELINE_MULT,
            recent_multiplier=ETA_RECENT_MULT,
        )
        cls = classify_projection_candidate(proj, eta_threshold=ETA_CLASSIFY_THRESHOLD)

        if det["is_candidate"]:
            candidate_type = cls["label"] if cls["label"] != "regular_or_not_detected" else "candidate_projection_point"
        else:
            candidate_type = "regular_or_not_detected"
        fdata = build_frechet_matrix(model, lam, FRECHET_N, DEVICE)
        sigma_min_F = compute_frechet_svd(fdata["F"])["sigma_min"]

        logger.info(
            "  eta_tikh=%.4e  eta_tsvd=%.4e  rank_tsvd=%d  rank_num=%d  gap=%.3f  sigma_min(F)=%.4e",
            proj["eta_tikhonov"], proj["eta_tsvd"], proj["tsvd_rank"], proj["numerical_rank"],
            proj["gap_after_r"], sigma_min_F,
        )
        if det["is_candidate"]:
            logger.info("  CANDIDATE: %s", det["reason"])

        points.append(ProjectionPoint(
            step=step,
            lam=lam,
            observable_center=train_result["observable_center"],
            observable_l2=train_result["observable_l2"],
            loss_total=train_result["loss_total"],
            eta_tikhonov=proj["eta_tikhonov"],
            eta_tsvd=proj["eta_tsvd"],
            sigma_min_F=sigma_min_F,
            tsvd_rank=int(proj["tsvd_rank"]),
            numerical_rank=int(proj["numerical_rank"]),
            gap_after_r=float(proj["gap_after_r"]),
            is_candidate=bool(det["is_candidate"]),
            candidate_type=candidate_type,
            candidate_reason=det["reason"],
            baseline_eta=float(det["baseline"] or 0.0),
            recent_eta_median=float(det["recent_median"] or 0.0),
            state_dict=copy.deepcopy(model.state_dict()),
        ))

    with open(os.path.join(RESULTS_DIR, "projection_detector.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(p) | {"state_dict": None} for p in points], f, indent=2)

    _plot_results(points)
    _write_report(points)

    candidates = [p for p in points if p.is_candidate]
    logger.info("=" * 70)
    logger.info("DONE. %d candidate(s).", len(candidates))
    for p in candidates:
        logger.info(
            "  lambda=%.4f eta_tikh=%.4e sigma_min(F)=%.4e type=%s reason=%s",
            p.lam, p.eta_tikhonov, p.sigma_min_F, p.candidate_type, p.candidate_reason,
        )


if __name__ == "__main__":
    main()
