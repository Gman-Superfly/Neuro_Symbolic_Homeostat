# Precision-scaled orthogonal noise (PSON)

Status: implemented, with synthetic mechanism measurements

Scope: first-order tangent noise, inverse-precision redistribution, box handling, and the evidence recorded in this repository

## Mechanism

For gradients above the numerical projection threshold, PSON adds a bounded exploration displacement whose directional derivative is zero at the point where the gradient is evaluated. For an energy $F$, state $x$, gradient $g=\nabla F(x)$, and noise displacement \(\delta\), the first-order condition is

\[
g^\top \delta = 0.
\]

Here, $g$ is the ordinary energy gradient and \(\delta\) is the constructed noise vector. The equation makes \(\delta\) tangent to the local energy level set to first order. It does not place \(\delta\) in the Hessian null space, establish zero curvature, or guarantee that a finite noisy proposal lowers energy.

The implemented precision-aware pipeline has four geometric stages:

1. Project a raw Gaussian draw into the first-order tangent plane.
2. Scale its coordinates by inverse diagonal precision.
3. Project again because coordinate scaling generally breaks the first orthogonality condition.
4. Apply one uniform box-feasible scale to the final vector.

The second projection and uniform box scale are required to preserve the first-order condition. Per-coordinate clipping would generally change direction and destroy it.

## Projection and precision scaling

### Euclidean projection

Given a raw draw $z$, the Euclidean projection is

\[
q = z - \frac{z^\top g}{g^\top g}g.
\]

The vector $q$ is the initially projected draw. For a gradient above the numerical stationary threshold, direct substitution gives \(g^\top q=0\). This is a first-order identity at the gradient evaluation point.

### Metric projection

If the caller supplies a symmetric positive definite metric $M$, the implementation can project along the metric gradient \(M^{-1}g\):

\[
q_M = z - \frac{g^\top z}{g^\top M^{-1}g}M^{-1}g.
\]

Here, $M$ is the supplied metric and \(q_M\) is tangent to the ordinary energy level set because \(g^\top q_M=0\). The metric changes the correction direction. It does not change the first-order energy condition being enforced.

### Inverse-precision redistribution and final projection

Let \(p_i\ge 0\) be the cached diagonal precision for coordinate $i$, and let \(\varepsilon>0\) be `precision_epsilon`. PSON forms coordinate weights proportional to

\[
w_i \propto \frac{1}{\varepsilon+p_i}.
\]

The weights allocate more of a fixed noise norm to coordinates with smaller reported diagonal curvature. Starting from $q$ or \(q_M\), the implementation forms \(\tilde q_i=w_i q_i\). Since \(g^\top\tilde q\) is generally nonzero, it projects \(\tilde q\) again with the same Euclidean or metric projection and then normalizes the result to the configured magnitude.

Precision scaling is diagonal. It uses reported local and coupling curvature and does not invert the full Hessian.

## Stationary exploration

The projection utilities use a default gradient-norm threshold of \(10^{-8}\). When \(\lVert g\rVert_2<10^{-8}\), they return the input draw unchanged because the gradient does not define a numerically reliable tangent normal. In precision mode, inverse-precision scaling still applies, but the projection calls do not impose a meaningful direction at that threshold.

This branch is stationary exploration. It is distinct from tangent exploration, and it carries no orthogonality claim beyond the gradient already being numerically small.

The noise builder also returns the zero vector when the requested magnitude or constructed vector norm is at most \(10^{-9}\). These thresholds prevent division by a nearly zero norm.

## Uniform box-feasible scaling

The coordinator first computes the deterministic projected update. Let \(y\in[0,1]^n\) be that state and let \(\delta\) be the normalized noise vector. It then chooses the largest \(s\in[0,1]\) for which

\[
y+s\delta\in[0,1]^n.
\]

The applied noise displacement is \(s\delta\). Since $s$ is one scalar, \(g^\top(s\delta)=s(g^\top\delta)=0\) whenever the constructed vector was tangent. The condition remains relative to the gradient used to build the proposal, which was evaluated before the deterministic update.

This box operation preserves direction. It can reduce the requested noise magnitude, including to zero when the state is on a boundary and the noise points outward.

## Second-order energy cost

For an energy that is twice continuously differentiable near \(x\), with Hessian \(H=\nabla^2F(x)\), a pure tangent displacement has the local expansion

\[
F(x+\delta)=F(x)+\frac{1}{2}\delta^\top H\delta+o(\lVert\delta\rVert^2).
\]

The first-order term is absent because \(g^\top\delta=0\). The quadratic term can be positive, zero, or negative depending on local curvature and direction. Tangency therefore does not make a finite noise proposal energy-neutral.

The coordinator's configured acceptance guard evaluates the full proposal. In the down-only energy mode, an uphill proposal is rejected and the previous accepted state is restored. This establishes accepted-state monotonicity under that guard. It does not turn the stochastic proposal map into a contraction or guarantee progress.

## Curvature measurements

The PSON ablation records requested-magnitude and realized box-feasible quantities for the same paired Gaussian draws. Let \(q\) be the vector returned by the noise builder and let \(\delta=sq\) be the displacement after the largest uniform box-feasible scale is applied at the initial state.

The diagonal curvature proxy is

\[
C_{\mathrm{diag}}(\delta)=\sum_i \Lambda_{ii}\delta_i^2.
\]

Here, \(\Lambda_{ii}\) is the raw cached diagonal precision before the positive epsilon floor used to construct the update preconditioner \(P\). The CSV fields `noise_diagonal_curvature_proxy` and `noise_diagonal_curvature_proxy_draws` contain the requested-vector proxy \(C_{\mathrm{diag}}(q)\). The older fields `noise_curvature_cost` and `noise_curvature_cost_draws` are compatibility aliases for those values. The fields `noise_realized_diagonal_curvature_proxy` and `noise_realized_diagonal_curvature_proxy_draws` contain \(C_{\mathrm{diag}}(\delta)\).

The exact synthetic full-Hessian metric is

\[
C_H(\delta)=\delta^\top H\delta.
\]

Here, $H$ is the analytic Hessian of the generated synthetic objective at its initial state. The generator includes quadratic and quartic local terms, quadratic couplings, active piecewise-quadratic hinge branches, and frozen linear gate-benefit terms. A frozen linear gate-benefit term contributes zero Hessian. States exactly at a hinge kink are rejected because the Hessian is undefined there.

The CSV fields `noise_full_hessian_cost` and `noise_full_hessian_cost_draws` contain the requested-vector value \(C_H(q)\). The fields `noise_realized_full_hessian_cost` and `noise_realized_full_hessian_cost_draws` contain the box-feasible value \(C_H(\delta)\) used by the primary table. `noise_box_scale_mean` and `noise_box_scaled_fraction` record how often and how strongly the initial-state box changes the requested magnitude. These metrics omit the common factor \(1/2\) from the Taylor expansion, which does not affect a paired relative comparison of the same metric.

## Recorded full-Hessian result

`logs/pson_noise_ablation_summary.csv` records 30 paired seeds, 32 paired noise draws per seed, and 10,000 paired hierarchical bootstrap samples for each comparison. The table reports the mean reduction in initial-state box-feasible \(\delta^\top H\delta\) for precision-orthogonal noise.

| Synthetic family | Versus isotropic | Versus orthogonal |
|---|---:|---:|
| Quadratic chain | 59.6% [57.8%, 61.2%] | 55.7% [53.9%, 57.2%] |
| Mixed gate chain | 59.7% [57.8%, 61.2%] | 55.7% [53.9%, 57.2%] |
| Quadratic star | 77.9% [76.3%, 79.2%] | 74.3% [72.5%, 75.7%] |
| Quadratic dense | 56.6% [53.8%, 59.0%] | 47.7% [44.2%, 50.8%] |
| Ill-conditioned ring | 83.8% [82.4%, 84.7%] | 82.1% [80.8%, 83.1%] |
| Nonlinear quartic | 45.9% [42.5%, 49.1%] | 41.8% [38.3%, 45.2%] |
| Active hinges | 47.2% [45.5%, 48.8%] | 41.4% [39.8%, 43.0%] |

Brackets show the recorded 95% paired hierarchical bootstrap interval. These measurements support one mechanism-level statement: inverse-precision redistribution lowered the initial synthetic full-Hessian noise cost for the listed generators and settings. They do not establish improved convergence, acceptance rate, final energy, constraint satisfaction, or task performance.

The controlled escape experiment is also synthetic. It places one lower-curvature double-well coordinate beside seven stiff distractors and tests precision allocation in that construction. Its result does not establish a general nonconvex escape advantage.

## Configuration

The explicit `noise_mode` values are:

- `"none"`
- `"isotropic"`
- `"orthogonal"`
- `"precision_orthogonal"`
- `"metric_orthogonal"`
- `"metric_precision_orthogonal"`

The older Boolean flags remain available and are resolved into one of these modes when `noise_mode` is `None`. New configurations should prefer the explicit mode.

```python
coord = EnergyCoordinator(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    noise_mode="precision_orthogonal",
    noise_magnitude=1e-2,
    noise_schedule_decay=0.99,
)
```

For metric-aware projection, provide exactly one metric representation:

```python
coord = EnergyCoordinator(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    noise_mode="metric_precision_orthogonal",
    noise_magnitude=1e-2,
    metric_matrix=M,
    # Or use metric_solve=solve_M instead of metric_matrix.
)
```

The supplied matrix or solve must define a positive symmetric geometry. Full metric-curvature scaling is outside the current implementation.

## Reproduction and checks

Run the projection demonstration:

```powershell
uv run python -m experiments.demo_metric_orthogonal
```

Run the paired synthetic ablation:

```powershell
uv run python -m experiments.ablate_pson_noise --trials 30 --steps 80 --noise-cost-samples 32 --bootstrap-samples 10000
```

Run the controlled analytic reference and escape construction:

```powershell
uv run python -m experiments.validate_pson_reference --samples 100000
uv run python -m experiments.benchmark_pson_escape --trials 200 --steps 40 --bootstrap-samples 10000
```

Run the focused tests:

```powershell
uv run python -m pytest -q tests/test_projection_properties.py tests/test_noise_modes.py tests/test_pson_ablation.py
```

For guard telemetry, prefer `log_step_cap_slack=True` and the `last_step_cap_slack` and `step_cap_slacks` metrics. The older contraction-margin names are compatibility aliases for the same cap-slack values, not measured spectral contraction rates.

## Boundaries

- First-order tangency is evaluated against the preproposal gradient.
- Inverse-precision scaling uses a diagonal approximation and requires the final re-projection.
- Uniform box scaling preserves direction but can reduce the noise norm.
- The stationary branch has no reliable tangent normal.
- Second-order and higher-order terms can raise energy.
- The recorded full-Hessian reductions are synthetic mechanism measurements, not task-level results.

See `docs/STABILITY_GUARANTEES.md` for the deterministic gradient-step conditions and `docs/README_AUTO_SCHEDULING.md` for the automatic magnitude and step schedules.
