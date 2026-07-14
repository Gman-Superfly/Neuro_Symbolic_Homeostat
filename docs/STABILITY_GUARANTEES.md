# Stability bounds in complexity from constraints

## Overview

This document describes the stability controls used by the current repository. The implemented mechanism is a conservative step cap plus accepted-step checks. In quadratic and SPD regimes, the cap follows the standard gradient-descent condition. In mixed regimes, the same machinery is a guard and telemetry path, not a proof of nonlinear convergence.

Status: implemented and tested on the synthetic scenarios in `tests/` and `experiments/`.

## Quick start

Enable the stability guard when coupling strength or curvature is uncertain:

```python
from core.coordinator import EnergyCoordinator

coord = EnergyCoordinator(
    modules=my_modules,
    couplings=my_couplings,
    constraints={},
    stability_guard=True,
    stability_cap_fraction=0.9,
    log_contraction_margin=True,
    warn_on_margin_shrink=True,
    margin_warn_threshold=1e-6,
)

etas = coord.relax_etas(etas0, steps=50)
```

The Small-Gain adapter is optional. Use it when an experiment needs adaptive coupling weights under the same margin telemetry:

```python
from core.weight_adapters import SmallGainWeightAdapter

coord = EnergyCoordinator(
    modules=my_modules,
    couplings=my_couplings,
    constraints={},
    weight_adapter=SmallGainWeightAdapter(
        budget_fraction=0.7,
        max_step_change=0.10,
    ),
    stability_guard=True,
    expose_lipschitz_details=True,
)
```

## Implemented contract

The coordinator estimates a conservative row-sum Lipschitz bound:

```text
L_hat = max_i (d_i + sum_j |c_ij|)
```

Here `d_i` is the local curvature contribution for coordinate `i`, and `c_ij` is the coupling curvature contribution between coordinates. Modules can report local curvature through `SupportsPrecision.curvature`. Couplings can report row-wise curvature through `SupportsCouplingCurvature.coupling_curvature_bounds`. Built-in quadratic and active hinge couplings also contribute curvature directly.

Given a requested step size `alpha_requested`, the guard uses:

```text
alpha_used = min(alpha_requested, stability_cap_fraction * 2 / L_hat)
```

If `auto_step_from_lipschitz=True`, then the requested step is set from the same bound. If a trial step increases the guarded objective, then the coordinator rejects the step and restores the previous accepted order parameters.

Boundary: the cap is a sufficient condition only when `L_hat` upper-bounds the true gradient Lipschitz constant. The accepted-step check protects the current run from harmful trial steps, but it does not turn mixed nonlinear constraints into a global convergence proof.

## Mathematical scope

For an L-smooth deterministic objective, gradient descent with `alpha < 2/L` gives:

```text
F(x_next) <= F(x) - alpha * (1 - alpha * L / 2) * ||grad F(x)||^2
```

The right-hand term is non-positive when `alpha < 2/L`, so the step decreases or preserves energy under the stated smoothness assumption.

In the quadratic/SPD case, if `L_hat >= lambda_max(H)` and `alpha_used < 2 / L_hat`, then the iteration matrix `I - alpha_used * H` is contractive. In mixed hinge, gate, and product-coupling regimes, the repository treats this as a conservative step rule plus rejection guard.

## Observed regimes

The current tests record two useful boundaries.

`tests/test_precision_conditioning.py::test_curvature_awareness_converges_where_plain_gd_stalls_above_2_over_L`

- Coupled quadratic pair with `lambda_max = 33`, so `2 / L = 0.0606`.
- Requested step `0.1` lies above that threshold.
- Plain gradient descent is rejected and stalls.
- Diagonal preconditioning or the Gershgorin step cap converges in this synthetic setting.

`tests/test_precision_conditioning.py::test_gershgorin_cap_can_be_conservative_on_mixed_preconditioned_problem`

- Mixed sum, product, and hinge couplings.
- The initial cap is below requested step `0.1`.
- Guarded and unguarded preconditioned runs both converge.
- Over the same fixed step budget, the guarded run can end at higher final energy.

Interpretation: curvature-aware safeguards can change the outcome in tight quadratic regimes. In mixed regimes, the same guard can trade speed for margin.

## Small-Gain adapter

`SmallGainWeightAdapter` ranks coupling families by a smoothed value-to-cost score. Value is approximated by gradient norm squared. Cost is the estimated increase in the Lipschitz bound. The adapter applies bounded weight changes while staying inside a global predicted-spend cap.

Current implementation boundary:

- It enforces a global predicted-spend cap.
- It records row margin telemetry.
- It does not yet enforce per-edge row-incidence booking.
- Its benefit depends on the graph and objective. The paper treats fewer steps and lower final energy as empirical outcomes, not formal guarantees.

## Telemetry

When `log_contraction_margin=True`, the coordinator records:

- `contraction_margin`: `(2 / L_hat) - alpha_used`
- `_contraction_margin_history`: recent margins
- `_last_lipschitz_details`: row sums, row margins, global margin, family costs, and edge costs when requested

`EnergyBudgetTracker` can log related fields:

- `contraction_margin`
- `margin:global`
- `margin:row:<i>`
- `cost:<family>`
- `spent:global`
- `alloc:<family>`
- `precision:min`
- `precision:mean`
- `precision:max`

## Tuning guidance

If the estimated Lipschitz bound is large, then the cap can make steps small. Diagnose the bound before adding new machinery:

1. Reduce coupling weights.
2. Reduce `step_size`.
3. Enable diagonal precision preconditioning or stiffness updates when modules and couplings report reliable curvature.
4. Inspect row-level curvature telemetry for an overly conservative component bound.
5. Lower `stability_cap_fraction` only when more margin is needed; this makes the cap stricter rather than faster.
6. Use `SmallGainWeightAdapter` only when adaptive coupling weights are part of the experiment.

If energy increases despite the guard, then inspect which extras are active. Noise, line search, adaptive term weights, and experimental ADMM paths can change the acceptance path. For a minimal stability check, use `noise_mode="none"`, `weight_adapter=None`, `stability_guard=True`, and `assert_monotonic_energy=True`.

## Curvature-contract audit

Run the sampled finite-difference auditor with:

```powershell
uv run python -m experiments.audit_curvature_contract --samples 32 --strict
```

Strict mode exits nonzero when a reported module or coupling bound falls below an observed Hessian entry. The recorded seven-family run covered 6,080 component-state records with no observed underreporting. This audit detects sampled violations; it does not prove a global bound between sampled states.

## Test coverage

Current stability-related coverage includes:

- `tests/test_precision_conditioning.py`: Lipschitz edge handling, preconditioning benefit, quadratic step-cap behavior, conservative mixed-regime behavior, randomized exact-Hessian coverage, and a structural custom-coupling check.
- `tests/test_end_to_end_relaxation.py`: accepted energy monotonicity, gradient norm decrease, positive contraction margins, large-noise rejection, and sparse Small-Gain monotonicity.
- `tests/test_small_gain_weight_adapter.py`: greedy allocation, bounds, fallback behavior, monotone energy on a small problem, and SPD contraction under the Gershgorin cap.
- `tests/test_stiffness_updates.py`: stiffness update descent, coupling curvature in the precision cache, Jacobi trajectory equivalence, and rejected-step restoration.

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Use `uv run -m pytest tests -v` when the local `uv` cache is accessible in the active environment.

## Summary

The stability machinery is a composable energy-relaxation contract with curvature-aware safeguards. The repo-specific part is the way modules and couplings report curvature to one coordinator. The formal contraction claim is limited to the stated quadratic/SPD condition. Mixed regimes rely on conservative caps, accepted-step checks, telemetry, and local tests.

## References

- Boyd, S., Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. Use: standard gradient-method descent condition for smooth objectives. Local implication: step caps need an upper bound on gradient Lipschitz curvature. Limits: mixed nonlinear constraints need local tests and rejection guards.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*. SIAM. Use: Jacobi and Gauss-Seidel convergence conditions for linear systems. Local implication: stiffness updates match a Jacobi-style path in quadratic/SPD blocks. Limits: the repository does not implement a dedicated Gauss-Seidel stiffness schedule.
- Vidyasagar, M. (1993). *Nonlinear Systems Analysis*. Prentice Hall. Use: small-gain framing for bounded feedback interactions. Local implication: margin telemetry is a useful guard for coupled updates. Limits: the current Small-Gain adapter is a conservative allocator, not a full nonlinear certificate.
