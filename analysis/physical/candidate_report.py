# Генерация отчетов по найденным кандидатам на особые точки (JSON + Markdown)
import json
import os
from typing import List
from continuation.branch_tracer import BranchPoint


def build_candidate_report(branch_points: List[BranchPoint], out_dir: str, problem_name: str = "bratu_2d") -> None:
    os.makedirs(out_dir, exist_ok=True)
    # отбираем только кандидатов (не регулярные точки)
    candidates = [bp for bp in branch_points if bp.candidate_type not in (None, "regular_point")]

    # JSON для автоматической обработки
    json_path = os.path.join(out_dir, f"{problem_name}_candidates.json")
    with open(json_path, "w") as f:
        json.dump([{
            "step": bp.step, "lambda": bp.lam,
            "observable_center": bp.observable_center, "observable_l2": bp.observable_l2,
            "sigma_min": bp.sigma_min, "sigma_second": bp.sigma_second,
            "rank_est": bp.rank_est, "candidate_type": bp.candidate_type,
        } for bp in candidates], f, indent=2)

    # Markdown для чтения человеком
    md_path = os.path.join(out_dir, f"{problem_name}_detector_report.md")
    lines = [
        f"# Special-Point Detector Report: {problem_name}", "",
        f"Branch points: {len(branch_points)}  |  Candidates: {len(candidates)}", "",
        "| step | λ | u(0.5,0.5) | σ_min | σ_2 | rank | type |",
        "|------|---|-----------|-------|-----|------|------|",
    ]
    for bp in candidates:
        lines.append(f"| {bp.step} | {bp.lam:.4f} | {bp.observable_center:.4f} "
                     f"| {bp.sigma_min:.3e} | {bp.sigma_second:.3e} | {bp.rank_est} | **{bp.candidate_type}** |")

    # таблица всех точек ветви для полноты
    lines += ["", "## All branch points", "",
              "| step | λ | u(0.5,0.5) | L2 | σ_min | type |",
              "|------|---|-----------|-----|-------|------|"]
    for bp in branch_points:
        sigma_str = f"{bp.sigma_min:.3e}" if bp.sigma_min is not None else "—"
        lines.append(f"| {bp.step} | {bp.lam:.4f} | {bp.observable_center:.4f} "
                     f"| {bp.observable_l2:.4f} | {sigma_str} | {bp.candidate_type or '—'} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
