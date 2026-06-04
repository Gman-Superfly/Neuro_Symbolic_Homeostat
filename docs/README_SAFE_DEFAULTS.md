# Conservative defaults for EnergyCoordinator (paper-validation friendly)

Status: Recommended starter config for reproducible, monotone relaxation in this repo.
Scope: Stability guard on, tangent noise with adaptive magnitude, precision-aware scaling, metric-aware projection optional.

---

## Recommended coordinator configuration

```python
from core.coordinator import EnergyCoordinator

coord = EnergyCoordinator(
    modules=modules,                 # your EnergyModule list
    couplings=couplings,             # [(i, j, coupling), ...]
    constraints=constraints,         # dict with any required keys

    # Stability and acceptance
    stability_guard=True,            # keep steps within contraction margin
    assert_monotonic_energy=True,    # deterministic runs enforce ΔF ≤ 0
    line_search=True,                # backtrack when uphill
    max_backtrack=5,
    backtrack_factor=0.5,

    # Steps and scaling
    use_stiffness_updates=True,      # per-coordinate η update by diagonal curvature
    auto_step_from_lipschitz=True,   # derive capped step size from local Lipschitz estimate

    # Exploration (Precision-Scaled Orthogonal Noise, PSON)
    enable_orthogonal_noise=True,    # tangent-plane noise (does not fight descent)
    auto_noise_controller=True,      # adapt magnitude from descent/backtracks/rotation
    precision_aware_noise_controller=True,  # allocate noise toward low curvature
    noise_magnitude=1e-2,            # maximum noise scale (annealed)
    noise_schedule_decay=0.99,       # exponential decay per step

    # Optional: Metric-aware projection (use when you have an SPD metric)
    # metric_aware_noise_controller=True,
    # metric_matrix=your_spd_matrix,          # or
    # metric_vector_product=your_linear_op,   # callable v -> M v

    # Keep advanced solvers off by default for paper demos
    operator_splitting=False,
    use_admm=False,
)
```

Notes:
- This configuration matches the paper’s monotone descent + PSON story with minimal tuning.
- If you enable metric-aware projection, the noise is projected with respect to \(M\) and re‑projected after precision re-weighting to preserve \(M\)-orthogonality.

---

## Quick repro (Windows PowerShell)

```powershell
# CGBC/wormhole demo
python -m experiments.demo_wormhole

# Metric-orthogonal projection demo
python -m experiments.demo_metric_orthogonal

# Operator-splitting demo (optional)
python -m experiments.demo_operator_splitting
```

---

## Why these defaults?
- Stability guard + line search help keep accepted ΔF traces well-behaved under the configured assumptions.
- Stiffness updates approximate Newton-like scaling without requiring full Hessians.
- PSON explores in the tangent plane; precision-aware scaling targets flat directions.
- Metric-aware projection (optional) respects problem geometry when an SPD metric is available.


