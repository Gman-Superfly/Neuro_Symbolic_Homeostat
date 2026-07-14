# Auto scheduling: noise and curvature-capped step size

Status: available in this repository (opt-in flags)
Scope: Automatic scheduling of (1) orthogonal noise magnitude and (2) step size from a stability bound.

---

## 1) Auto Noise Controller

Flag: `auto_noise_controller=True`
Controller:
- OrthogonalNoiseController (default), or
- PrecisionNoiseController if `precision_aware_noise_controller=True`

What it does:
- Adapts the orthogonal noise magnitude each step using simple signals:
  - Descent rate (slow progress → more noise)
  - Backtracks (line‑search events → more noise)
  - Gradient rotation (curved valley → more noise)
- Applies exponential decay (annealing) over steps.

Heuristic (see `core/noise_controller.py`):

```
s_t = clamp(w_rate*(1 - rate) + w_backtrack*backtrack + w_rotation*(1 - cosθ), 0, 1)
noise_magnitude = s_t * noise_max * (decay ** iter_idx)
```

Precision‑aware option:
- If `precision_aware_noise_controller=True`, the controller redistributes noise along low‑curvature directions via weights ∝ 1/(ε + curvature). After re‑weighting, we re‑project to preserve orthogonality.

Metric‑aware note:
- If an SPD metric is provided through `metric_matrix` or `metric_solve` and `metric_aware_noise_controller=True`, the projection uses the metric gradient while preserving zero first-order energy change. The controller still only sets magnitude.

Recommended combo (automatic magnitude and capped step):

```python
coord = EnergyCoordinator(
    ...,
    enable_orthogonal_noise=True,
    auto_noise_controller=True,              # schedule magnitude
    precision_aware_noise_controller=True,   # redistribute by 1/curvature
    noise_magnitude=1e-2,                    # max noise scale
    noise_schedule_decay=0.99,               # anneal
    # optional metric-aware projection:
    # metric_aware_noise_controller=True,
    # metric_matrix=np.diag([1.0, 10.0, ...]),
)
```

Observability:
- Use `EnergyBudgetTracker` to log `contraction_margin`, `monotonicity_violation`, and `redemption_gain` (energy drop per second). Tangent noise should keep ΔF traces well‑behaved with stability guard on.

---

## 2) Auto Step From Lipschitz

Flags: `stability_guard=True`, `auto_step_from_lipschitz=True`
Bound: capped step ≤ stability_cap_fraction · (2 / L)

What it does:
- With `stability_guard=True`, we estimate a conservative Lipschitz bound L (Gershgorin‑style) and cap each step:
  `step_to_use = min(step_size, stability_cap_fraction * (2/L))`.
- With `auto_step_from_lipschitz=True`, we set the step to the cap directly:
  `step_to_use = stability_cap_fraction * (2/L)` (hands‑free step size).

Why it’s useful:
- Uses the same composed curvature estimate as the stability guard and reduces manual step tuning.
- Keeps iterations inside the contraction condition when the estimated bound upper-bounds the local Lipschitz constant.

Recommended setup:

```python
coord = EnergyCoordinator(
    ...,
    stability_guard=True,
    stability_cap_fraction=0.9,     # ~90% of the conservative bound
    auto_step_from_lipschitz=True,  # choose bound as the step
)
```

Compatibility:
- Works with stiffness‑based updates (`use_stiffness_updates=True`), PSON, and adapters. The cap applies to the effective step either way.

---

## Quick recipes

### A) Tangent exploration + auto step
```python
coord = EnergyCoordinator(
    ...,
    # Noise
    enable_orthogonal_noise=True,
    auto_noise_controller=True,
    precision_aware_noise_controller=True,
    noise_magnitude=1e-2,
    noise_schedule_decay=0.99,
    # Stability
    stability_guard=True,
    stability_cap_fraction=0.9,
    auto_step_from_lipschitz=True,
    # Solver
    use_stiffness_updates=True,
)
```

### B) Metric-aware geometry (optional)
```python
coord = EnergyCoordinator(
    ...,
    metric_aware_noise_controller=True,
    metric_matrix=np.diag([1.0, 5.0, 0.5, 2.0]),
)
```

---

## Notes and caveats
- The Lipschitz estimate L is conservative (Gershgorin) and may fluctuate; use `stability_cap_fraction≈0.8–0.95` for margin.
- Precision‑aware redistribution changes direction; we re‑project to remain orthogonal.
- If you want fully deterministic runs for tests, keep `noise_magnitude=0.0` and/or `enable_orthogonal_noise=False`.

---

## See also
- `docs/README_TANGENT_NOISE_PSON.md` (design of Euclidean and metric-consistent tangent noise)
- `docs/STABILITY_GUARANTEES.md` (Gershgorin bounds, contraction margins, and the optional Small-Gain allocator)
- `experiments/demo_metric_orthogonal.py` (metric-aware projection demo)
