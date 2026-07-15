# Observability: trackers and diagnostics

Status: implemented.

Scope: trajectory logging, stability telemetry, update-geometry inspection, and adapter budgets.

## RelaxationTracker

`RelaxationTracker` records the accepted trajectory of energy and order parameters.

```python
from cf_logging.observability import RelaxationTracker

tracker = RelaxationTracker(
    name="my_experiment_trace",
    run_id="run_001",
    log_per_eta=True,
)
tracker.attach(coord)

coord.relax_etas(etas0, steps=50)
tracker.flush()
```

The output includes:

- `step`, `energy`, and `delta_energy`.
- `min_eta`, `max_eta`, and `mean_eta`.
- `eta:<i>` when `log_per_eta=True`.
- `compute_cost` in seconds per step.
- `redemption_gain`, the recorded energy drop per second.

Callbacks fire only after a proposal is accepted. A rejected proposal restores the prior state and is absent from the accepted trajectory. Use `last_relaxation_metrics()` when attempted and rejected counts are also required.

Before any solver dispatch, the coordinator snapshots top-level constraints in a read-only mapping. Gate-benefit deltas are converted to finite floats and remain fixed for the complete solver call. The original caller mapping is restored afterward. Energy, gradient, curvature, and acceptance telemetry therefore refer to one frozen external-input snapshot; adapter-owned term weights remain explicitly versioned between accepted steps.

## EnergyBudgetTracker

`EnergyBudgetTracker` records energy decomposition, gradient summaries, stability fields, and adaptive-weight budgets.

```python
from cf_logging.observability import EnergyBudgetTracker

tracker = EnergyBudgetTracker(
    name="budget_log",
    run_id="run_001",
    warn_on_margin_shrink=True,
    log_free_energy_decomposition=True,
)
tracker.attach(coord)

coord.relax_etas(etas0, steps=50)
tracker.flush()
```

The output can include:

- Terms: `energy:local:<family>`, `energy:coup:<family>`, and term gradient norms.
- Stability: `step_cap_slack`, `step_cap_slack_warn`, `last_backtracks`, `total_backtracks`, and `acceptance_reason`.
- Deprecated compatibility: `contraction_margin` aliases `step_cap_slack`, and `margin_warn` aliases `step_cap_slack_warn`.
- Small-Gain: `margin:global`, `margin:row:<i>`, `cost:<family>`, `alloc:<family>`, and `spent:global`.
- Thermodynamics: `U_internal_energy`, `S_entropy`, `F_free_energy`, and `T_temperature` when free-energy decomposition is enabled.
- Precision: `precision:min`, `precision:mean`, and `precision:max`.
- Events: monotonicity and acceptance provenance fields supplied by the coordinator.

The preferred stability quantity is

\[
\operatorname{step\_cap\_slack}
=\frac{2}{L_{\mathrm{update}}}-\alpha_{\mathrm{used}}.
\]

It measures distance from the uncushioned \(2/L_{\mathrm{update}}\) boundary. It is not the spectral contraction factor

\[
q=\max_{\lambda}|1-\alpha\lambda|.
\]

For a preconditioned quadratic, the eigenvalues in \(q\) belong to \(P^{-1/2}HP^{-1/2}\). For an ordinary quadratic, they belong to \(H\).

## Raw and update-geometry curvature

Use the public snapshot API when an experiment needs the state and both curvature views at one point:

```python
snapshot = coord.inspect_state(etas)

snapshot.energy
snapshot.gradient
snapshot.precision_diagonal
snapshot.preconditioner_diagonal
snapshot.lipschitz_bound
snapshot.update_lipschitz_bound
snapshot.term_weights
snapshot.term_gradient_norms
```

The fields have distinct meanings:

- `precision_diagonal` is the nonnegative cached curvature before the active epsilon floor.
- `preconditioner_diagonal` is the exact positive diagonal \(P\) used to divide the gradient.
- `lipschitz_bound` is the raw, unpreconditioned Hessian row-sum bound \(L_H\).
- `update_lipschitz_bound` is the normalized \(L_P\) bound when stiffness or precision preconditioning is active. It equals the raw bound for an ordinary update.

For a preconditioned SPD quadratic,

\[
L_P\ge\lambda_{\max}(P^{-1/2}HP^{-1/2})
\]

is the premise used by the step guard. Logging only \(L_H\) is insufficient to diagnose the executed \(P^{-1}H\) dynamics.

Detailed `_last_lipschitz_details` data, when enabled, also identifies the active geometry, the preconditioner used for normalized rows, row sums, row margins, family costs, and edge costs. This object is coordinator instrumentation. Use the snapshot fields for stable point-in-time diagnostics.

## Line-search provenance

Projected Armijo evaluates the objective gradient \(g\) against the realized box-constrained displacement

\[
s=\Pi_{[0,1]^n}(x-\alpha d)-x.
\]

It accepts when \(g^\top s\le0\) and

\[
F(x+s)\le F(x)+c\,g^\top s.
\]

`last_relaxation_metrics()` exposes `last_acceptance_reason` and the per-attempt `acceptance_reasons` list. `armijo_accepted` records success. `armijo_failed_no_step` records exhausted backtracking; the returned state is unchanged. Callback trackers receive accepted-state emissions, while rejected-attempt provenance remains available through these run metrics.

## Noise diagnostics

`coord.build_noise_vector(raw_noise, snapshot.gradient)` exposes the configured noise transformation for controlled ablations. Precision-orthogonal modes re-project after inverse-precision weighting. The relaxation path applies the largest uniform scalar that keeps the full vector inside the unit box, which preserves first-order tangency above the numerical gradient threshold.

At gradient norm below \(10^{-8}\), the projection helper returns the noise unchanged because no reliable tangent normal exists. Diagnostics should label this branch as stationary-point exploration rather than a null-space certificate.

## Operating guidance

- Attach trackers immediately after creating the coordinator.
- Flush after each experiment block.
- Use unique `run_id` values.
- Disable per-coordinate logging for large throughput runs when the additional data is unnecessary.
- Compare `lipschitz_bound` with `update_lipschitz_bound` before diagnosing a preconditioned step.
- Treat deprecated contraction-margin fields as step-cap slack aliases in existing logs.
