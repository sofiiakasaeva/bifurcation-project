# PINN Special-Point Detector for 2D Bratu

Bifurcation analysis via PINN: detect and classify singular points of the Frechet matrix surrogate along the solution branch of

```
Δu + λ·exp(u) = 0,   (x,y) ∈ (0,1)²,   u = 0 on ∂Ω
```

## Project structure

```
bifurcation_project/
  problems/
    base_problem.py        # abstract PDE interface
    bratu_2d.py            # 2D Bratu implementation
  pinn/
    model.py               # MLP with tanh (configurable depth/width)
    autograd_ops.py        # Laplacian, parameter helpers
    residual_vector.py     # R(W,λ) — raw residual vector (not scalar loss)
    parameter_utils.py     # flatten / count parameters
  continuation/
    warmstart_trainer.py   # train PINN at fixed λ with warm-start
    branch_tracer.py       # walk λ grid, collect BranchPoint snapshots
  analysis/
    frechet_surrogate.py   # J_W = dR/dW,  r_λ = dR/dλ
    svd_detector.py        # σ_min tracking, candidate detection
    classifier.py          # classify via extended matrix J_* = [J_W | r_λ]
    candidate_report.py    # JSON + Markdown report
  experiments/
    run_bratu_detector.py  # end-to-end script
  utils/
    config.py              # dataclass configs
    io.py                  # CSV export
    plotting.py            # bifurcation diagram + σ_min plot
  tests/
    test_shapes_and_values.py
```

## Installation

```bash
pip install torch numpy matplotlib pytest
```

## Run the experiment

```bash
cd bifurcation_project
python experiments/run_bratu_detector.py
```

Outputs:
| Path | Description |
|------|-------------|
| `results/bratu_branch.csv` | full branch table |
| `results/bratu_candidates.json` | candidate special points |
| `figures/bratu_branch.png` | bifurcation diagram |
| `figures/bratu_sigma_min.png` | σ_min(λ) plot |
| `reports/bratu_detector_report.md` | human-readable report |

## Run sanity tests

```bash
cd bifurcation_project
pytest tests/ -v
```

## Method overview

1. **Branch tracing**: warm-start PINN at each λ step.
2. **Residual vector**: `R(W,λ)` — concatenation of PDE and BC residuals at *fixed* collocation points (not random).
3. **Frechet surrogate**: `J_W = dR/dW` (row-by-row autograd), `r_λ = dR/dλ` (autograd or FD).
4. **SVD detector**: track `σ_min(J_W)`; flag candidate when it is small or drops sharply.
5. **Classifier**: check whether `r_λ ∈ image(J_W)` via least squares:
   - `candidate_limit_point` — fold (r_λ not in image)
   - `candidate_bifurcation_point` — branching (r_λ in image)

Reference critical λ for 2D Bratu: **≈ 7.03** (literature sanity check).

## Configuration

All hyperparameters live in `utils/config.py`:
- `TrainConfig` — epochs, lr, bc_weight, collocation counts
- `DetectorConfig` — detection collocation sizes, SVD thresholds
- `ModelConfig` — PINN architecture
- `ExperimentConfig` — λ grid, device, output dirs

Edit `experiments/run_bratu_detector.py` to change any of these.
