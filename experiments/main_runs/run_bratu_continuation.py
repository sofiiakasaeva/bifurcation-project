"""Experiment: pseudo-arclength continuation for 2D Bratu + fold detection.
Usage:
    python experiments/run_bratu_continuation.py
"""
import sys
import os
import json
import logging
import copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from problems.bratu_2d import Bratu2DProblem
from pinn.model import PINN
from continuation.arclength_continuation import run_arclength_continuation
from analysis.physical.frechet_pde import (
    build_frechet_matrix,
    compute_frechet_svd,
    classify_frechet_candidate,
)
from analysis.physical.classifier import classify_keller_antman, compute_corank
from utils.config import ContinuationConfig, ModelConfig
from utils.io import continuation_branch_to_csv
from utils.plotting import (
    plot_bifurcation_diagram,
    plot_sigma_tracking,
    plot_continuation_path,
)

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
    max_steps=200,
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

FRECHET_N = 20
SIGMA_PHYS_THRESHOLD = 1.0
LEFT_NULLVEC_THRESH = 0.1

RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

# строим матрицу Фреше + SVD + classify для точки
def analyse_point(model, lam, device):
    frechet_data = build_frechet_matrix(model, lam, FRECHET_N, device)
    F = frechet_data["F"]
    f_lam = frechet_data["f_lambda"]

    svd_info = compute_frechet_svd(F, tol_factor=10.0)
    tol = svd_info["tol"]

    cls_lnv = classify_frechet_candidate(F, f_lam, tol=tol,
        sigma_physical_threshold=SIGMA_PHYS_THRESHOLD, left_nullvec_threshold=LEFT_NULLVEC_THRESH)

    cls_ka = None
    if svd_info["sigma_min"] < 2.0:
        cls_ka = classify_keller_antman(F, f_lam, tol=tol,
                                         sigma_physical_threshold=SIGMA_PHYS_THRESHOLD)

    corank_info = compute_corank(F, tol=tol)

    return {
        "sigma_min": svd_info["sigma_min"],
        "sigma_second": svd_info["sigma_second"],
        "candidate_type": cls_lnv["label"],
        "left_null_proj": cls_lnv.get("left_null_proj"),
        "ka_label": cls_ka["label"] if cls_ka else None,
        "corank": corank_info["corank"],
    }

# saves CSV + plots
def incremental_save(branch, step):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # CSV всегда
    csv_path = os.path.join(RESULTS_DIR, "bratu_continuation_branch.csv")
    continuation_branch_to_csv(branch, csv_path)

    # plots только каждые 5 шагов
    if step % 5 == 0 or step < 3:
        try:
            plot_bifurcation_diagram(
                branch,
                out_path=os.path.join(FIGURES_DIR, "bratu_cont_bifurcation.png"),
                title=f"2D Bratu continuation (step {step}): u(0.5,0.5) vs λ",
            )
            plot_continuation_path(
                branch,
                out_path=os.path.join(FIGURES_DIR, "bratu_cont_norm_path.png"),
                title=f"2D Bratu continuation path (step {step})",
            )
            has_sigma = any(getattr(bp, "sigma_min", None) is not None for bp in branch)
            if has_sigma:
                plot_sigma_tracking(
                    branch,
                    out_path=os.path.join(FIGURES_DIR, "bratu_cont_sigma.png"),
                    title=f"σ_min(F) along branch (step {step})",
                )
        except Exception as e:
            logger.warning("  Plot failed at step %d: %s", step, e)

    # checkpoint обязательно!!!!
    if step % 10 == 0:
        ckpt_path = os.path.join(RESULTS_DIR, "continuation_checkpoint.pt")
        torch.save({
            "step": step,
            "lam": branch[-1].lam,
            "norm_u": branch[-1].norm_u,
            "state_dict": branch[-1].state_dict,
        }, ckpt_path)


# svd анализ на каждую точку
def annotate_branch_with_svd(branch, model, device):
    logger.info("Running SVD analysis on %d branch points", len(branch))
    for i, bp in enumerate(branch):
        if bp.state_dict is None:
            continue
        model.load_state_dict(bp.state_dict)
        model.eval()
        try:
            info = analyse_point(model, bp.lam, device)
            bp.sigma_min = info["sigma_min"]
            bp.sigma_second = info["sigma_second"]
            bp.candidate_type = info["candidate_type"]
            bp.corank = info["corank"]
            if i % 10 == 0:
                logger.info("  [%d/%d] λ=%.4f  σ_min=%.4e  type=%s  corank=%d",
                            i, len(branch), bp.lam, bp.sigma_min,
                            bp.candidate_type, bp.corank)
        except Exception as e:
            logger.warning("SVD failed at step %d: %s", bp.step, e)

# репортик
def build_final_report(branch, out_dir, fold_lam_approx=None):
    os.makedirs(out_dir, exist_ok=True)
    candidates = [bp for bp in branch if getattr(bp, "candidate_type", None) not in (None, "regular_point")]
    if fold_lam_approx is None:
        lams = [bp.lam for bp in branch]
        if lams:
            fold_lam_approx = max(lams)

    json_path = os.path.join(out_dir, "bratu_continuation_candidates.json")
    records = []
    for bp in candidates:
        records.append({
            "step": bp.step, "lambda": bp.lam,
            "norm_u": bp.norm_u,
            "observable_center": bp.observable_center,
            "sigma_min": bp.sigma_min,
            "candidate_type": bp.candidate_type,
            "corank": bp.corank,
        })
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    md_path = os.path.join(out_dir, "bratu_continuation_report.md")
    lines = [
        "Bratu 2D — Pseudo-arclength Continuation Report", "",
        f"Total branch points: {len(branch)}",
        f"Candidates (limit/bifurcation): {len(candidates)}",
        f"Approximate fold λ*: **{fold_lam_approx:.4f}**",
        "",
        "## Candidate special points", "",
        "| step | λ | ‖u‖ | u(0.5,0.5) | σ_min | corank | type |",
        "|------|---|-----|-----------|-------|--------|------|",
    ]
    for bp in candidates:
        sm = f"{bp.sigma_min:.3e}" if bp.sigma_min is not None else "—"
        cr = bp.corank if bp.corank is not None else "—"
        lines.append(f"| {bp.step} | {bp.lam:.4f} | {bp.norm_u:.4f} "
                     f"| {bp.observable_center:.4f} | {sm} | {cr} | **{bp.candidate_type}** |")

    lines += [
        "", "Branch summary (every 5th point)", "",
        "| step | λ | ‖u‖ | u(0.5,0.5) | σ_min | loss_pde | type |",
        "|------|---|-----|-----------|-------|----------|------|",
    ]
    for bp in branch[::5]:
        sm = f"{bp.sigma_min:.3e}" if getattr(bp, "sigma_min", None) is not None else "—"
        ct = getattr(bp, "candidate_type", "—") or "—"
        lp = f"{bp.loss_pde:.3e}" if bp.loss_pde else "—"
        lines.append(f"| {bp.step} | {bp.lam:.4f} | {bp.norm_u:.4f} "
                     f"| {bp.observable_center:.4f} | {sm} | {lp} | {ct} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("[report] JSON → %s", json_path)
    logger.info("[report] MD   → %s", md_path)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    problem = Bratu2DProblem()
    model   = PINN(MODEL_CFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 60)
    logger.info("Bratu 2D — Pseudo-arclength Continuation")
    logger.info("=" * 60)
    logger.info("Model: %d parameters (hidden=%d, layers=%d)",
                n_params, MODEL_CFG.hidden_dim, MODEL_CFG.num_hidden_layers)
    logger.info("Continuation: gamma=%.3f, alpha=%.1f, max_steps=%d, epochs=%d",
                CONTINUATION_CFG.gamma, CONTINUATION_CFG.alpha_cont,
                CONTINUATION_CFG.max_steps, CONTINUATION_CFG.epochs_per_step)

    for d in [RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)

    logger.info("PHASE 1: pseudo-arclength continuation")
    branch = run_arclength_continuation(
        problem, model, lam_start=0.5,
        cfg=CONTINUATION_CFG, device=DEVICE,
        on_step_done=incremental_save
    )
    logger.info("Branch traced: %d points", len(branch))
    logger.info("PHASE 2: SVD analysis on all branch points")
    annotate_branch_with_svd(branch, model, DEVICE)
    continuation_branch_to_csv(
        branch, os.path.join(RESULTS_DIR, "bratu_continuation_branch.csv")
    )
    plot_bifurcation_diagram(
        branch,
        out_path=os.path.join(FIGURES_DIR, "bratu_cont_bifurcation.png"),
        title="2D Bratu — full continuation with SVD",
    )
    plot_sigma_tracking(
        branch,
        out_path=os.path.join(FIGURES_DIR, "bratu_cont_sigma.png"),
        title="σ_min(F) along branch",
    )
    plot_continuation_path(
        branch,
        out_path=os.path.join(FIGURES_DIR, "bratu_cont_norm_path.png"),
        title="2D Bratu — continuation path",
    )
    fold_lam = max(bp.lam for bp in branch) if branch else None
    build_final_report(branch, REPORTS_DIR, fold_lam_approx=fold_lam)

    candidates = [bp for bp in branch if getattr(bp, "candidate_type", None) not in (None, "regular_point")]
    logger.info("=" * 60)
    logger.info("DONE.")
    logger.info("Branch points: %d", len(branch))
    logger.info("Fold λ* ≈ %.4f  (max λ on branch)", fold_lam or 0)
    logger.info("Candidates: %d", len(candidates))
    for bp in candidates[:5]:
        logger.info("step=%d  λ=%.4f  σ_min=%.4e  type=%s",
                    bp.step, bp.lam,
                    bp.sigma_min if bp.sigma_min else 0,
                    bp.candidate_type)


if __name__ == "__main__":
    main()