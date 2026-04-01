# сохранение ветви
import csv
import os
from typing import List


def branch_to_csv(branch_points: List, csv_path: str) -> None:
    """Write BranchPoint list (fixed-lambda scan) to CSV."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fields = ["step", "lam", "observable_center", "observable_l2",
              "loss_total", "residual_mse_eval", "sigma_min", "sigma_second", "rank_est", "candidate_type"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for bp in branch_points:
            writer.writerow({k: getattr(bp, k) for k in fields})
    print(f"[io] → {csv_path}")


def continuation_branch_to_csv(branch_points: List, csv_path: str) -> None:
    """Write ContinuationPoint list (arclength continuation) to CSV."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fields = ["step", "lam", "norm_u", "observable_center", "observable_l2",
              "loss_total", "loss_pde", "sigma_min", "sigma_second", "candidate_type", "corank"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for bp in branch_points:
            writer.writerow({fn: getattr(bp, fn, None) for fn in fields})
    print(f"[io] → {csv_path}")
