# Continuation + projection detector for 2D Bratu
from __future__ import annotations
import sys
import os
import json
import logging
from typing import List, Dict, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from problems.bratu_2d import Bratu2DProblem
from pinn.model import PINN
from continuation.arclength_continuation import run_arclength_continuation, ContinuationPoint
from analysis.pinn_native.projection_detector import (
    compute_projection_diagnostics,
    detect_projection_candidate,
    classify_projection_candidate,
)
from analysis.physical.frechet_pde import build_frechet_matrix, compute_frechet_svd
from utils.config import ContinuationConfig, ModelConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = "cpu"

CONTINUATION_CFG = ContinuationConfig(
    gamma=0.03,
    alpha_cont=15.0,
    max_steps=120,
    norm_target_max=1.85,
    epochs_per_step=3000,
    lr=1e-3,
    bc_weight=15.0,
    n_int_train=1500,
    n_bnd_train=200,
)

MODEL_CFG = ModelConfig(
    input_dim=2,
    hidden_dim=32,
    num_hidden_layers=3,
    output_dim=1,
)

N_INT_SURROGATE = 2500
N_BND_SURROGATE = 200

ENERGY_FRAC = 0.999
ALPHA_REL = 1e-4
WARMUP_STEPS = 5
ETA_ABS_THRESHOLD = 0.15
ETA_BASELINE_MULT = 5.0
ETA_RECENT_MULT = 2.0
ETA_CLASSIFY_THRESHOLD = 0.15

WITH_FRECHET_BASELINE = True
FRECHET_N = 10

RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

def branch_to_jsonable(branch: List[ContinuationPoint]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bp in branch:
        row = {
            "step": bp.step,
            "lambda": float(bp.lam),
            "norm_u": float(bp.norm_u),
            "observable_center": float(bp.observable_center),
            "observable_l2": float(bp.observable_l2),
            "loss_total": float(bp.loss_total),
            "loss_pde": float(bp.loss_pde),
            "eta_tikhonov": _maybe_float(getattr(bp, "eta_tikhonov", None)),
            "eta_tsvd": _maybe_float(getattr(bp, "eta_tsvd", None)),
            "tsvd_rank": _maybe_int(getattr(bp, "tsvd_rank", None)),
            "numerical_rank": _maybe_int(getattr(bp, "numerical_rank", None)),
            "gap_after_r": _maybe_float(getattr(bp, "gap_after_r", None)),
            "projection_is_candidate": bool(getattr(bp, "projection_is_candidate", False)),
            "projection_reason": getattr(bp, "projection_reason", None),
            "projection_type": getattr(bp, "projection_type", None),
            "baseline_eta": _maybe_float(getattr(bp, "baseline_eta", None)),
            "recent_eta_median": _maybe_float(getattr(bp, "recent_eta_median", None)),
            "sigma_min_F": _maybe_float(getattr(bp, "sigma_min_F", None)),
            "sigma_second_F": _maybe_float(getattr(bp, "sigma_second_F", None)),
        }
        rows.append(row)
    return rows


def _maybe_float(x):
    return None if x is None else float(x)


def _maybe_int(x):
    return None if x is None else int(x)


def annotate_branch_with_projection_detector(branch: List[ContinuationPoint], problem: Bratu2DProblem,
    model: torch.nn.Module, device: str, int_pts: torch.Tensor, bnd_pts: torch.Tensor) -> None:
    history: List[Dict[str, Any]] = []

    logger.info("Running projection detector on %d branch points...", len(branch))
    for i, bp in enumerate(branch):
        if bp.state_dict is None:
            logger.warning("  step=%d has no state_dict; skipping", bp.step)
            continue

        model.load_state_dict(bp.state_dict)
        model.eval()

        lam_t = torch.tensor(bp.lam, dtype=torch.float32, device=device)
        proj = compute_projection_diagnostics(
            problem,
            model,
            lam_t,
            int_pts,
            bnd_pts,
            bc_weight=CONTINUATION_CFG.bc_weight,
            energy_frac=ENERGY_FRAC,
            alpha_rel=ALPHA_REL,
        )
        history.append(proj)

        det = detect_projection_candidate(
            history,
            len(history) - 1,
            score_key="eta_tikhonov",
            warmup_steps=WARMUP_STEPS,
            abs_threshold=ETA_ABS_THRESHOLD,
            baseline_multiplier=ETA_BASELINE_MULT,
            recent_multiplier=ETA_RECENT_MULT,
        )
        cls = classify_projection_candidate(proj, eta_threshold=ETA_CLASSIFY_THRESHOLD)

        if det["is_candidate"]:
            projection_type = cls["label"] if cls["label"] != "regular_or_not_detected" else "candidate_projection_point"
        else:
            projection_type = "regular_point"

        bp.eta_tikhonov = float(proj["eta_tikhonov"])
        bp.eta_tsvd = float(proj["eta_tsvd"])
        bp.tsvd_rank = int(proj["tsvd_rank"])
        bp.numerical_rank = int(proj["numerical_rank"])
        bp.gap_after_r = float(proj["gap_after_r"])
        bp.projection_is_candidate = bool(det["is_candidate"])
        bp.projection_reason = det["reason"]
        bp.projection_type = projection_type
        bp.baseline_eta = det["baseline"]
        bp.recent_eta_median = det["recent_median"]

        if WITH_FRECHET_BASELINE:
            fdata = build_frechet_matrix(model, bp.lam, FRECHET_N, device)
            fsvd = compute_frechet_svd(fdata["F"])
            bp.sigma_min_F = float(fsvd["sigma_min"])
            bp.sigma_second_F = float(fsvd["sigma_second"])

        if i % 5 == 0 or det["is_candidate"]:
            msg = (
                "  [%d/%d] λ=%.4f ||u||=%.4f eta_tikh=%.4e eta_tsvd=%.4e "
                "rank_tsvd=%d type=%s"
            )
            logger.info(
                msg,
                i,
                len(branch),
                bp.lam,
                bp.norm_u,
                bp.eta_tikhonov,
                bp.eta_tsvd,
                bp.tsvd_rank,
                bp.projection_type,
            )
            if det["is_candidate"]:
                logger.info("      candidate reason: %s", det["reason"])


def _candidate_mask(branch: List[ContinuationPoint]) -> List[bool]:
    return [bool(getattr(bp, "projection_is_candidate", False)) for bp in branch]


def plot_projection_vs_lambda(branch: List[ContinuationPoint], out_path: str) -> None:
    lams = [bp.lam for bp in branch]
    eta_tikh = [getattr(bp, "eta_tikhonov", np.nan) for bp in branch]
    eta_tsvd = [getattr(bp, "eta_tsvd", np.nan) for bp in branch]
    sigma_F = [getattr(bp, "sigma_min_F", np.nan) for bp in branch]
    cand = _candidate_mask(branch)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(lams, eta_tikh, "o-", label="eta_tikhonov")
    ax1.plot(lams, eta_tsvd, "s-", label="eta_tsvd")
    ax1.scatter([l for l, c in zip(lams, cand) if c], [e for e, c in zip(eta_tikh, cand) if c],
                marker="*", s=160, label="candidates")
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("projection score")
    ax1.set_title("Projection detector along continuation branch")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if np.isfinite(np.array(sigma_F)).any():
        ax2.semilogy(lams, sigma_F, "o-", label="sigma_min(F)")
        ax2.scatter([l for l, c in zip(lams, cand) if c], [s for s, c in zip(sigma_F, cand) if c],
                    marker="*", s=160, label="candidates")
        ax2.set_ylabel("sigma_min(F)")
        ax2.legend()
    else:
        ax2.plot(lams, [np.nan]*len(lams))
    ax2.set_xlabel("lambda")
    ax2.set_title("Physical Fréchet baseline")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_projection_vs_norm(branch: List[ContinuationPoint], out_path: str) -> None:
    norms = [bp.norm_u for bp in branch]
    eta_tikh = [getattr(bp, "eta_tikhonov", np.nan) for bp in branch]
    eta_tsvd = [getattr(bp, "eta_tsvd", np.nan) for bp in branch]
    sigma_F = [getattr(bp, "sigma_min_F", np.nan) for bp in branch]
    cand = _candidate_mask(branch)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(norms, eta_tikh, "o-", label="eta_tikhonov")
    ax1.plot(norms, eta_tsvd, "s-", label="eta_tsvd")
    ax1.scatter([n for n, c in zip(norms, cand) if c], [e for e, c in zip(eta_tikh, cand) if c],
                marker="*", s=160, label="candidates")
    ax1.set_xlabel("||u||")
    ax1.set_ylabel("projection score")
    ax1.set_title("Projection detector vs branch coordinate ||u||")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if np.isfinite(np.array(sigma_F)).any():
        ax2.semilogy(norms, sigma_F, "o-", label="sigma_min(F)")
        ax2.scatter([n for n, c in zip(norms, cand) if c], [s for s, c in zip(sigma_F, cand) if c],
                    marker="*", s=160, label="candidates")
        ax2.legend()
    else:
        ax2.plot(norms, [np.nan]*len(norms))
    ax2.set_xlabel("||u||")
    ax2.set_ylabel("sigma_min(F)")
    ax2.set_title("Physical Fréchet vs ||u||")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_bifurcation_colored_by_eta(branch: List[ContinuationPoint], out_path: str) -> None:
    lams = np.array([bp.lam for bp in branch], dtype=float)
    u_center = np.array([bp.observable_center for bp in branch], dtype=float)
    eta = np.array([getattr(bp, "eta_tikhonov", np.nan) for bp in branch], dtype=float)
    cand = np.array(_candidate_mask(branch), dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(lams, u_center, c=eta, s=50, cmap="viridis")
    if cand.any():
        ax.scatter(lams[cand], u_center[cand], marker="*", s=180, color="red", label="projection candidates")
        ax.legend()
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("eta_tikhonov")
    ax.set_xlabel("lambda")
    ax.set_ylabel("u(0.5,0.5)")
    ax.set_title("Bifurcation diagram colored by projection score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_aux_diagnostics(branch: List[ContinuationPoint], out_path: str) -> None:
    lams = [bp.lam for bp in branch]
    rank_tsvd = [getattr(bp, "tsvd_rank", np.nan) for bp in branch]
    gap = [getattr(bp, "gap_after_r", np.nan) for bp in branch]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(lams, rank_tsvd, "o-")
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("tsvd rank")
    ax1.set_title("Adaptive TSVD rank")
    ax1.grid(True, alpha=0.3)

    ax2.plot(lams, gap, "o-")
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("gap after retained rank")
    ax2.set_title("Gap after TSVD truncation")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_report(branch: List[ContinuationPoint], out_path: str) -> None:
    candidates = [bp for bp in branch if bool(getattr(bp, "projection_is_candidate", False))]
    lines: List[str] = [
        "# Continuation + Projection Detector Report",
        "",
        "Projection score:",
        "eta = ||(I - P) r_lambda|| / ||r_lambda||, where P is a regularised projection onto Im(J_W).",
        "",
        f"Total branch points: {len(branch)}",
        f"Projection candidates: {len(candidates)}",
        "",
        "## Candidate points",
        "",
        "| step | lambda | ||u|| | u(0.5,0.5) | eta_tikh | eta_tsvd | sigma_min(F) | type | reason |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for bp in candidates:
        lines.append(
            f"| {bp.step} | {bp.lam:.4f} | {bp.norm_u:.4f} | {bp.observable_center:.4f} | "
            f"{getattr(bp, 'eta_tikhonov', float('nan')):.4e} | "
            f"{getattr(bp, 'eta_tsvd', float('nan')):.4e} | "
            f"{getattr(bp, 'sigma_min_F', float('nan')):.4e} | "
            f"{getattr(bp, 'projection_type', '—')} | {getattr(bp, 'projection_reason', '—')} |"
        )

    lines += [
        "",
        "## Branch summary (every 5th point)",
        "",
        "| step | lambda | ||u|| | u(0.5,0.5) | eta_tikh | sigma_min(F) | type |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for bp in branch[::5]:
        lines.append(
            f"| {bp.step} | {bp.lam:.4f} | {bp.norm_u:.4f} | {bp.observable_center:.4f} | "
            f"{getattr(bp, 'eta_tikhonov', float('nan')):.4e} | "
            f"{getattr(bp, 'sigma_min_F', float('nan')):.4e} | {getattr(bp, 'projection_type', '—')} |"
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

    logger.info("=" * 72)
    logger.info("CONTINUATION + PROJECTION DETECTOR")
    logger.info("=" * 72)
    logger.info("Model parameters: %d", n_params)
    logger.info(
        "Continuation cfg: gamma=%.3f alpha=%.1f max_steps=%d epochs/step=%d",
        CONTINUATION_CFG.gamma,
        CONTINUATION_CFG.alpha_cont,
        CONTINUATION_CFG.max_steps,
        CONTINUATION_CFG.epochs_per_step,
    )

    logger.info("PHASE 1: continuation branch tracing")
    branch = run_arclength_continuation(
        problem,
        model,
        lam_start=0.5,
        cfg=CONTINUATION_CFG,
        device=DEVICE,
        on_step_done=None,
    )
    logger.info("Continuation finished: %d branch points", len(branch))

    logger.info("Preparing fixed collocation for projection detector")
    int_pts = problem.sample_interior_fixed(N_INT_SURROGATE, DEVICE)
    bnd_pts = problem.sample_boundary_fixed(N_BND_SURROGATE, DEVICE)
    logger.info("Detector collocation size: m=%d", int_pts.shape[0] + bnd_pts.shape[0])

    logger.info("PHASE 2: projection detector over saved branch points")
    annotate_branch_with_projection_detector(branch, problem, model, DEVICE, int_pts, bnd_pts)

    json_path = os.path.join(RESULTS_DIR, "projection_on_continuation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(branch_to_jsonable(branch), f, indent=2)

    plot_projection_vs_lambda(branch, os.path.join(FIGURES_DIR, "projection_on_cont_lambda.png"))
    plot_projection_vs_norm(branch, os.path.join(FIGURES_DIR, "projection_on_cont_norm.png"))
    plot_bifurcation_colored_by_eta(branch, os.path.join(FIGURES_DIR, "projection_on_cont_bifurcation.png"))
    plot_aux_diagnostics(branch, os.path.join(FIGURES_DIR, "projection_on_cont_aux.png"))
    build_report(branch, os.path.join(REPORTS_DIR, "projection_on_continuation_report.md"))

    candidates = [bp for bp in branch if bool(getattr(bp, "projection_is_candidate", False))]
    logger.info("=" * 72)
    logger.info("DONE. %d projection candidate(s).", len(candidates))
    for bp in candidates:
        logger.info(
            "  step=%d λ=%.4f ||u||=%.4f eta_tikh=%.4e sigma_min(F)=%.4e type=%s",
            bp.step,
            bp.lam,
            bp.norm_u,
            getattr(bp, "eta_tikhonov", float("nan")),
            getattr(bp, "sigma_min_F", float("nan")),
            getattr(bp, "projection_type", "—"),
        )


if __name__ == "__main__":
    main()
