# рисуем
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BRATU_LAMBDA_CRIT = 7.03


def _savefig(fig, path, msg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {msg} → {path}")


def plot_branch(branch_points, out_path, observable="center", title="Bifurcation branch"):
    lams = [bp.lam for bp in branch_points]
    vals = [bp.observable_center if observable == "center" else bp.observable_l2 for bp in branch_points]
    ylabel = "u(0.5, 0.5)" if observable == "center" else "||u||_L2"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lams, vals, "b.-", linewidth=1.5, markersize=5, label="branch")
    for bp, y in zip(branch_points, vals):
        if bp.candidate_type not in (None, "regular_point"):
            ax.scatter(bp.lam, y, color="red", zorder=5, s=80)
    ax.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1, label=f"ref λ* ≈ {BRATU_LAMBDA_CRIT}")
    ax.set_xlabel("λ")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_path, "branch")


def plot_sigma_min(branch_points, out_path, title="σ_min along branch", also_sigma2=True):
    """Log-scale plot of sigma_min (and optionally sigma_2) vs lambda."""
    pts = [bp for bp in branch_points if bp.sigma_min is not None]
    if not pts:
        print("[plot] No sigma_min data; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([bp.lam for bp in pts], [bp.sigma_min for bp in pts], "b.-", linewidth=1.5, markersize=5, label="σ_min")
    if also_sigma2:
        s2_pts = [(bp.lam, bp.sigma_second) for bp in pts if bp.sigma_second is not None]
        if s2_pts:
            ax.semilogy(*zip(*s2_pts), "g.--", linewidth=1, markersize=4, label="σ_2")
    for bp in pts:
        if bp.candidate_type not in (None, "regular_point"):
            ax.scatter(bp.lam, bp.sigma_min, color="red", zorder=5, s=80)
    ax.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1, label=f"ref λ* ≈ {BRATU_LAMBDA_CRIT}")
    ax.set_xlabel("λ")
    ax.set_ylabel("singular value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _savefig(fig, out_path, "sigma_min")


def plot_bifurcation_diagram(branch_points, out_path, title=""):
    """u(0.5,0.5) vs lambda for continuation results, with limit points marked."""
    lams = [bp.lam for bp in branch_points]
    obs = [getattr(bp, "observable_center", None) or getattr(bp, "norm_u", 0.0) for bp in branch_points]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lams, obs, "b-o", markersize=3, linewidth=1.5, label="branch")
    for bp, y in zip(branch_points, obs):
        ct = getattr(bp, "candidate_type", None)
        if ct == "candidate_limit_point":
            ax.plot(bp.lam, y, "rs", markersize=8, zorder=5, label="limit point")
        elif ct == "candidate_bifurcation_point":
            ax.plot(bp.lam, y, "g^", markersize=8, zorder=5, label="bifurcation point")
    ax.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1, label=f"ref λ* ≈ {BRATU_LAMBDA_CRIT}")
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    if unique:
        ax.legend(*zip(*unique))
    ax.set_xlabel("λ")
    ax.set_ylabel("u(0.5, 0.5)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_path, "bifurcation diagram")


def plot_sigma_tracking(branch_points, out_path, title=""):
    """Side-by-side log-scale panels for sigma_min and sigma_second."""
    pts = [bp for bp in branch_points if getattr(bp, "sigma_min", None) is not None]
    if not pts:
        print("[plot] No sigma data; skipping.")
        return

    lams = [bp.lam for bp in pts]
    s1 = [bp.sigma_min for bp in pts]
    s2_items = [(bp.lam, bp.sigma_second) for bp in pts if getattr(bp, "sigma_second", None) is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(lams, s1, "b-o", markersize=3, linewidth=1.5)
    for bp, y in zip(pts, s1):
        if getattr(bp, "candidate_type", None) not in (None, "regular_point"):
            ax1.scatter(bp.lam, y, color="red", s=60, zorder=5)
    ax1.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("λ")
    ax1.set_ylabel("σ_min(F)")
    ax1.set_title("σ₁ (smallest)")
    ax1.grid(True, which="both", alpha=0.3)

    if s2_items:
        ax2.semilogy(*zip(*s2_items), "r-o", markersize=3, linewidth=1.5)
        ax2.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("λ")
    ax2.set_ylabel("σ₂(F)")
    ax2.set_title("σ₂ (second smallest)")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(title)
    _savefig(fig, out_path, "sigma tracking")


def plot_continuation_path(branch_points, out_path, title="Continuation path"):
    """(lambda, ||u||) path showing how the branch wraps around the fold."""
    pairs = [(bp.lam, getattr(bp, "norm_u", getattr(bp, "observable_l2", None)))
             for bp in branch_points]
    pairs = [(l, n) for l, n in pairs if n is not None]
    if not pairs:
        print("[plot] No norm_u data; skipping.")
        return

    lams, norms = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(lams, norms, "b-o", markersize=3, linewidth=1.5)
    for bp, n in zip(branch_points, norms):
        if getattr(bp, "candidate_type", None) == "candidate_limit_point":
            ax.scatter(bp.lam, n, color="red", s=80, zorder=5)
    ax.axvline(BRATU_LAMBDA_CRIT, color="gray", linestyle="--", linewidth=1, label=f"ref λ* ≈ {BRATU_LAMBDA_CRIT}")
    ax.set_xlabel("λ")
    ax.set_ylabel("‖u‖")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_path, "continuation path")
