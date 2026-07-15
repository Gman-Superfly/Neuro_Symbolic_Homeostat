# Small-Gain weight allocation

Status: implemented and tested on the synthetic repository scenarios.

Scope: family-level coupling-weight adaptation under the coordinator's
curvature-aware step contract.

## Mechanism

The coordinator estimates a Gershgorin upper bound in the geometry of the
executed update. For an ordinary gradient step this is the raw Hessian bound
\(L_H\). For a diagonal-preconditioned step it is the normalized bound

\[
L_P=\max_i\left(
\frac{\bar h_{ii}}{p_i}
+\sum_{j\ne i}\frac{\bar h_{ij}}{\sqrt{p_i p_j}}
\right),
\]

which bounds \(P^{-1/2}HP^{-1/2}\). The diagonal \(P\) is the exact positive
preconditioner used to divide the gradient. Write the selected bound as
\(L_{\mathrm{update}}\). For step \(\alpha\) and safety fraction \(f\), the
allocator uses the curvature limit

\[
L_{limit} = \frac{2f}{\alpha}.
\]

Here \(f\) is `stability_cap_fraction`. The available global margin is

\[
m = \max(0, L_{limit} - L_{\mathrm{update}}).
\]

The adapter may spend `budget_fraction * m`. Family and edge costs use the same
raw or normalized row geometry as \(L_{\mathrm{update}}\). This budget does not
replace the step cap, the quadratic contraction condition, projected Armijo,
or the accepted-step energy guard.

Built-in directed and asymmetric hinges contribute worst-case curvature across
their active and inactive regions, which covers a proposal that crosses an
active-set boundary. A nonlinear or custom report must remain valid over the
realized proposal segment for a fixed-step theorem. Otherwise, use projected
Armijo.

Gate-benefit deltas are finite, read-only snapshots for the complete solver
call. Plain gate benefit and damped power \(p=1\) are linear and have zero
Hessian cost. For damped energy \(-a\eta^p\) with \(p\ge2\), the allocator sees
the exact box-wide absolute diagonal cost \(|a|p(p-1)\). This magnitude bound
does not imply positive curvature; the term is concave when \(a>0\), and
contraction still requires an SPD total Hessian. For nonzero \(a\), damped
powers \(1<p<2\) have no finite closed-box gradient-Lipschitz bound at zero, so
a fixed-step guarded run fails closed unless projected Armijo is enabled. A zero
frozen coefficient removes the term and gives a zero curvature report. Powers
below one are rejected.

For each coupling family, the coordinator reports a curvature cost. The
adapter uses squared gradient norm as a local value proxy:

\[
value_k = \lVert g_k \rVert^2,
\qquad
score_k = \operatorname{EMA}\left(\frac{value_k}{cost_k + \epsilon}\right).
\]

Families are considered in descending score order. An accepted allocation
increases the family weight by at most `max_step_change`, subject to the global
spend limit and configured weight bounds.

Boundary: the score is a heuristic for local usefulness. It does not prove
that increasing a family weight improves task behavior. Row margins are logged,
but the current adapter does not book spend against edge-to-row incidence. The
enforced allocation limit is global.

## Configuration

```python
from core.coordinator import EnergyCoordinator
from core.weight_adapters import SmallGainWeightAdapter

coordinator = EnergyCoordinator(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    weight_adapter=SmallGainWeightAdapter(
        budget_fraction=0.7,
        max_step_change=0.10,
        floor=0.1,
        ceiling=3.0,
        ema_alpha=0.3,
    ),
    stability_guard=True,
    expose_lipschitz_details=True,
)
```

`stability_guard=True` supplies the geometry-matched curvature limit used by
the surrounding relaxation contract. `expose_lipschitz_details=True` requests
family costs, row margins, the active geometry label, and the preconditioner
used for normalized rows. The benchmark preset enables both settings.

## Current evidence

The following results were rerun after the publication cleanup. The baseline
scenario used 60 requested steps. The dense scenario used 16 modules and 40
requested steps. Noise magnitude was zero. ΔF90 includes the initial energy and
reports the first trace index that reaches 90 percent of that run's eventual
fixed-reference energy drop. Fixed-reference energy uses the original term
weights. Adaptive energy uses the current adapter-maintained weights.

| Scenario | Configuration | Reference ΔF90 | Reference final | Adaptive final | Backtracks |
|---|---:|---:|---:|---:|---:|
| Baseline | Analytic | 23 | 0.002730 | 0.002730 | 0 |
| Baseline | GradNorm | 17 | -0.002508 | -0.005015 | 10 |
| Baseline | Small-Gain | 20 | 0.005019 | -0.020208 | 9 |
| Dense | Analytic | 29 | 0.132823 | 0.132823 | 0 |
| Dense | GradNorm | 33 | 0.130522 | 0.261044 | 0 |
| Dense | Small-Gain | 31 | 0.180603 | -0.031118 | 0 |

Observed result: Small-Gain reached the lowest adaptive energy in both tested
scenarios because it changed coupling-family weights. It did not reach the
lowest fixed-reference energy in either scenario. GradNorm reached the lowest
fixed-reference energy in these two runs, with small improvements over the
analytic baseline. Adaptive energy is therefore reported as solver state, not
as a cross-configuration quality metric.

These runs support the implementation and synthetic comparison only. They do
not establish performance on larger graph families, noisy benefit estimates,
or task-level data.

## Observability

`EnergyBudgetTracker` records the active allocation fields:

- `spent:global`: curvature budget spent during the current adapter update.
- `alloc:coup:<family>`: accepted family-weight increment.
- `cost:coup:<family>`: estimated family curvature cost.
- `margin:global`: available global curvature margin.
- `margin:row:<index>`: row margin telemetry.

The separate `step_cap_slack` field is
\((2/L_{\mathrm{update}})-\alpha_{\mathrm{used}}\). It is distance from the
uncushioned step boundary and is not a spectral contraction factor. The older
`contraction_margin` field remains a deprecated compatibility alias.

```python
from cf_logging.observability import EnergyBudgetTracker

tracker = EnergyBudgetTracker(name="energy_budget", run_id="run_001")
tracker.attach(coordinator)
coordinator.relax_etas(initial_etas, steps=50)
tracker.flush()
```

## Verification

```powershell
uv run -m pytest tests/test_small_gain_weight_adapter.py -v
uv run python -m experiments.benchmark_delta_f90 `
  --configs analytic gradnorm smallgain --scenario baseline --steps 60
uv run python -m experiments.benchmark_delta_f90 `
  --configs analytic gradnorm smallgain --scenario dense --dense_size 16 --steps 40
```

The unit tests cover ranking, bounds, fallback behavior, coordinator cost-key
integration, positive margin spend, monotone accepted energy on a small graph,
and the quadratic/SPD condition in the geometry of the executed update.

## References

- Varga, R. S. (2004). *Gersgorin and His Circles*. Springer. Use: row-sum bounds for eigenvalue localization. Local implication: the coordinator can form a conservative curvature estimate from local and coupling contributions. Limits: the bound can be loose.
- Boyd, S., Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. Use: gradient descent and projected-gradient step conditions for smooth objectives. Local implication: the fixed-step quadratic/SPD contract requires \(\alpha < 2/L_{\mathrm{update}}\) in the executed geometry. Limits: state-dependent terms require segment-valid bounds or line search.
- Vidyasagar, M. (1993). *Nonlinear Systems Analysis*. Prentice Hall. Use: small-gain framing for bounded feedback interactions. Local implication: coupling adaptation should expose and limit estimated interaction spend. Limits: this family-level allocator is not a nonlinear stability certificate.
