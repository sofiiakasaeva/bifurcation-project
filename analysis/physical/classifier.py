# Классификация кандидатов на особые точки по матрице Фреше
import torch
from typing import Dict, Any


# критерий Келлера-Антмана: limit point vs bifurcation point
def classify_keller_antman(F: torch.Tensor, f_lambda: torch.Tensor, tol: float, sigma_physical_threshold: float = 1.0) -> dict:
    n = F.shape[0]
    f_lam = f_lambda.reshape(-1, 1).to(dtype=F.dtype)

    # сначала проверяем: может точка вообще регулярная
    sv_F = torch.linalg.svdvals(F)
    sigma_min_F = float(sv_F[-1])

    if sigma_min_F >= sigma_physical_threshold:
        return {
            "label": "regular_point",
            "sigma_min_F": sigma_min_F,
            "ranks": [],
        }

    # расширенная матрица [F | f_lambda]
    F_star = torch.cat([F, f_lam], dim=1)

    # удаляем по одному столбцу и смотрим ранг
    ranks = []
    for k in range(n + 1):
        cols = list(range(n + 1))
        cols.pop(k)
        rank_k = int((torch.linalg.svdvals(F_star[:, cols]) > tol).sum())
        ranks.append(rank_k)

    # если хотя бы одна подматрица невырождена — limit point
    label = "candidate_bifurcation_point" if all(r < n for r in ranks) else "candidate_limit_point"

    return {
        "label": label,
        "sigma_min_F": sigma_min_F,
        "ranks": ranks,
        "all_singular": all(r < n for r in ranks),
    }


# коранг матрицы Фреше: сколько сингулярных чисел близки к нулю
def compute_corank(F: torch.Tensor, tol: float) -> dict:
    n = F.shape[0]
    sv = torch.linalg.svdvals(F)
    rank = int((sv > tol).sum())
    corank = n - rank

    # соответствие коранга и типа катастрофы
    type_map = {0: "regular", 1: "cuspoid", 2: "umbilic"}
    return {
        "corank": corank,
        "rank": rank,
        "sigma_min": float(sv[-1]),
        "sigma_second": float(sv[-2]) if n > 1 else float("nan"),
        "catastrophe_type": type_map.get(corank, f"higher_corank_{corank}"),
    }