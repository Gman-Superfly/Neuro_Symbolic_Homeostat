# Auto scheduling for noise and step size

Status: implemented for the primary gradient solver

Scope: automatic tangent-noise magnitude and curvature-capped gradient steps

## Dispatch boundary

The schedules in this document operate inside the primary gradient relaxation loop. Selecting the experimental proximal or ADMM-like solver dispatches before that loop, so these controls do not govern those solver iterations.

## Automatic tangent-noise magnitude

Enable the controller with:

```python
auto_noise_controller=True
```

The controller is active for the orthogonal noise modes:

- `orthogonal`
- `precision_orthogonal`
- `metric_orthogonal`
- `metric_precision_orthogonal`

Isotropic noise uses only the configured magnitude and exponential decay. `noise_mode="none"` disables noise.

### Controller signals

For proposal $t$, the controller combines three bounded signals:

\[
r_t=\max(0,1-10d_{t-1}),
\]

where \(d_{t-1}\) is the relative energy drop recorded after the previous completed proposal. Small prior progress makes \(r_t\) larger. This is a scheduling heuristic, not a convergence statistic.

The backtrack signal is

\[
b_t=\mathbf{1}[k_{t-1}>0],
\]

where \(k_{t-1}\) is the line-search backtrack count from the previous completed proposal. Proposal zero receives \(b_0=0\). The coordinator saves each completed proposal's count for the next controller decision.

When the current and previous gradient norms exceed \(10^{-9}\), the rotation signal is

\[
q_t=\frac{1}{2}\left(1-\frac{g_t^\top g_{t-1}}{\lVert g_t\rVert_2\lVert g_{t-1}\rVert_2}\right).
\]

Here, \(g_t\) and \(g_{t-1}\) are consecutive gradients. The signal ranges from zero for aligned gradients to one for opposite gradients. It is set to zero when either norm is below the threshold.

The effective magnitude is

\[
m_t=m_{\max}\,\gamma^t\,
\operatorname{clamp}(w_r r_t+w_b b_t+w_q q_t, s_{\min},s_{\max}).
\]

Here, \(m_{\max}\) is `noise_magnitude`, \(\gamma\) is `noise_schedule_decay`, and the default signal weights are \(w_r=0.5\), \(w_b=0.2\), and \(w_q=0.3\). The default scale interval is \([0,1]\).

Controller state resets once at the start of each relaxation call. With the default initial progress value, no previous gradient, and no previous backtracks, the first automatic magnitude is zero. Later proposals respond to the most recently completed proposal and the current gradient rotation.

### Precision and metric modes

`PrecisionNoiseController` uses the same magnitude schedule. It additionally redistributes the constructed direction with inverse diagonal precision before the final re-projection. Metric modes change the projection direction while preserving the ordinary first-order energy condition.

See `docs/README_TANGENT_NOISE_PSON.md` for the initial projection, inverse-precision scaling, final re-projection, stationary branch, and uniform box-feasible scaling.

### Configuration

```python
coordinator = EnergyCoordinator(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    noise_mode="precision_orthogonal",
    auto_noise_controller=True,
    noise_magnitude=1e-2,
    noise_schedule_decay=0.99,
)
```

The controller schedules magnitude. It does not certify that a stochastic proposal will lower energy. The configured proposal-acceptance guard evaluates that proposal separately.

## Automatic curvature-capped step size

Enable the curvature guard and automatic step selection with:

```python
stability_guard=True
auto_step_from_lipschitz=True
```

The bound follows the geometry of the update that the coordinator actually applies.

### Ordinary gradient update

For the unpreconditioned update

\[
x^+=x-\alpha\nabla F(x),
\]

the coordinator uses a Gershgorin row bound \(\widehat L\) for the Hessian $H$. Here, \(\alpha\) is the scalar step size and $F$ is the configured energy.

### Diagonally preconditioned update

When stiffness updates or precision preconditioning are active, the implemented update is

\[
x^+=x-\alpha P^{-1}\nabla F(x),
\]

where $P$ is the positive diagonal actually used to divide the gradient. The coordinator bounds the symmetric normalized Hessian

\[
A=P^{-1/2}HP^{-1/2}.
\]

If \(d_i\) is the diagonal curvature bound and \(c_{ij}\) is an absolute off-diagonal curvature bound, the implemented Gershgorin estimate is

\[
\widehat L_P=
\max_i\left(
\frac{d_i}{p_i}
+\sum_{j\ne i}\frac{c_{ij}}{\sqrt{p_i p_j}}
\right).
\]

Here, \(p_i>0\) is the $i$-th diagonal entry of $P$. Using the same $P$ in the update and the bound is required. A bound on the unpreconditioned Hessian alone does not control the preconditioned map.

### Step selection

Let \(f\in(0,1)\) be `stability_cap_fraction`. The configured cap is

\[
\alpha_{\mathrm{cap}}=f\frac{2}{\widehat L_{\mathrm{update}}},
\]

where \(\widehat L_{\mathrm{update}}\) is \(\widehat L\) for an ordinary update and \(\widehat L_P\) for a preconditioned update.

With `auto_step_from_lipschitz=True`, the coordinator selects \(\alpha=\alpha_{\mathrm{cap}}\). Otherwise, the guard uses

\[
\alpha=\min(\alpha_{\mathrm{requested}},\alpha_{\mathrm{cap}}).
\]

The estimate is recomputed at each proposal. `auto_step_from_lipschitz` has no effect when `stability_guard=False`.

For an SPD quadratic with a valid upper bound on \(\lambda_{\max}(P^{-1/2}HP^{-1/2})\), a step satisfying \(0<\alpha<2/\widehat L_P\) makes the affine preconditioned iteration contractive in the $P$-norm. For the ordinary path, set \(P=I\).

For nonlinear or piecewise objectives, the estimate must bound curvature over the relevant proposal region for the same conclusion to apply locally. The acceptance guard can restore an uphill proposal, but rejection alone does not prove contraction or progress.

Gradient normalization changes the update map, so the coordinator does not apply the Lipschitz cap when `normalize_grads=True`. Gradient-norm clipping also changes the affine map. In clipped regimes, the cap remains a step-size guard, while the unmodified quadratic spectral statement does not describe every clipped iteration.

### Configuration

```python
coordinator = EnergyCoordinator(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    stability_guard=True,
    stability_cap_fraction=0.9,
    auto_step_from_lipschitz=True,
    use_precision_preconditioning=True,
)
```

`use_stiffness_updates=True` selects the same normalized-bound path with the stiffness update's positive diagonal and epsilon.

## Step-cap slack telemetry

Enable the preferred diagnostic with:

```python
log_step_cap_slack=True
```

The recorded value is

\[
s_{\mathrm{cap}}=\frac{2}{\widehat L_{\mathrm{update}}}-\alpha_{\mathrm{used}}.
\]

Here, \(\alpha_{\mathrm{used}}\) is the scalar step after capping. The preferred metrics are:

- `last_step_cap_slack`
- `step_cap_slacks`

This quantity is distance in step-size units to the \(2/\widehat L_{\mathrm{update}}\) boundary. It is not a measured spectral radius, contraction factor, or contraction rate.

The following names remain only for configuration and log-reader compatibility:

- `log_contraction_margin`
- `last_contraction_margin`
- `contraction_margins`

They activate or expose the same step-cap-slack values. New code and prose should use the step-cap-slack names.

`warn_on_margin_shrink` and `margin_warn_threshold` currently retain their older public names. Their warning text reports step-cap slack.

## Combined recipe

```python
coordinator = EnergyCoordinator(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    # Tangent exploration
    noise_mode="precision_orthogonal",
    auto_noise_controller=True,
    noise_magnitude=1e-2,
    noise_schedule_decay=0.99,
    # Deterministic gradient step
    stability_guard=True,
    stability_cap_fraction=0.9,
    auto_step_from_lipschitz=True,
    use_precision_preconditioning=True,
    # Telemetry
    log_step_cap_slack=True,
)
```

The noise schedule and deterministic step cap have different roles. The first chooses a stochastic exploration norm. The second controls the deterministic gradient map under its curvature assumptions. Full-proposal acceptance remains a separate check.

## Focused checks

```powershell
uv run python -m pytest -q tests/test_end_to_end_relaxation.py tests/test_precision_conditioning.py tests/test_noise_modes.py
```

See `docs/STABILITY_GUARANTEES.md` for the formal stability scope and `docs/README_OPERATOR_SPLITTING.md` for the separate experimental solver paths.
