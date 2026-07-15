# Stability bounds in complexity from constraints

## Overview

The coordinator guards the geometry of the update it actually executes. An ordinary gradient step uses an unnormalized Hessian bound. A diagonal-preconditioned step uses a bound on the normalized Hessian \(P^{-1/2}HP^{-1/2}\), where \(P\) is the same positive diagonal used to divide the gradient.

This distinction gives a contraction theorem for box-projected diagonal preconditioning on quadratic objectives with an SPD Hessian. State-dependent and custom terms require a curvature bound that remains valid over the realized proposal segment, or projected Armijo backtracking. The accepted-step guard restores the previous state after any remaining uphill proposal.

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
    log_step_cap_slack=True,
    warn_on_margin_shrink=True,
    margin_warn_threshold=1e-6,
)

etas = coord.relax_etas(etas0, steps=50)
```

Set `use_precision_preconditioning=True` or `use_stiffness_updates=True` to use a diagonal-preconditioned direction. The coordinator then builds one positive diagonal \(P\), uses it in the update, and passes that exact diagonal to the normalized stability estimator.

The Small-Gain adapter remains optional. Use it when an experiment needs adaptive coupling weights under the same update-geometry telemetry:

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

## Curvature composition

Let \(H\) denote the Hessian for a quadratic objective, or a symmetric curvature bound for the segment under consideration. The local part of this bound is finite-differenced from local gradients. It is independent of the `SupportsPrecision.curvature` values stored in the diagonal precision cache. Couplings contribute diagonal and absolute off-diagonal bounds through built-in formulas or `SupportsCouplingCurvature`. For the unpreconditioned geometry, the coordinator forms

\[
L_H = \max_i\left(\bar h_{ii}+\sum_{j\ne i}\bar h_{ij}\right),
\]

where \(\bar h_{ii}\) bounds \(|H_{ii}|\), and \(\bar h_{ij}\) bounds \(|H_{ij}|\). Gershgorin's theorem gives \(\lambda_{\max}(H)\le L_H\) when these component bounds cover \(H\). The row sum is an upper bound on the eigenvalue, which corrects the reversed inequality used in an earlier draft.

For a preconditioned update, let

\[
P=\operatorname{diag}(p_1,\ldots,p_n), \qquad p_i>0.
\]

The update diagonal includes the configured epsilon floor:

\[
p_i=\max(\varepsilon,\Lambda_{ii}),
\]

where \(\Lambda_{ii}\) is the nonnegative cached curvature. Module `SupportsPrecision.curvature` values and supported coupling diagonals feed this cache. The active stiffness or precision epsilon supplies \(\varepsilon\). Both the direction and the stability estimator consume this same \(P\), but the Hessian component bounds normalized by \(P\) are composed separately from the cache.

Define the normalized Hessian and its row-sum bound by

\[
A=P^{-1/2}HP^{-1/2},
\]

\[
L_P=\max_i\left(
\frac{\bar h_{ii}}{p_i}
+\sum_{j\ne i}\frac{\bar h_{ij}}{\sqrt{p_i p_j}}
\right).
\]

The matrix \(A\) is symmetric when \(H\) is symmetric. If the component bounds cover \(H\), then Gershgorin's theorem gives \(\lambda_{\max}(A)\le L_P\). The guard uses \(L_P\) for a preconditioned step and \(L_H\) for an ordinary step:

\[
\alpha_{\mathrm{used}}
=\min\left(\alpha_{\mathrm{requested}},\gamma\frac{2}{L_{\mathrm{update}}}\right),
\qquad 0<\gamma<1,
\]

where \(L_{\mathrm{update}}=L_P\) under diagonal preconditioning and \(L_{\mathrm{update}}=L_H\) otherwise.

## Gradient realization and finite differences

The primary solver uses analytic component derivatives when available. Its fallback differentiates the full objective in every coordinate, including local objectives on isolated nodes in a partially coupled graph. A coupling-neighborhood optimization cannot safely omit those coordinates.

All maintained solver paths use the same box-aware scalar stencil: centered second order in the interior and three-point second-order one-sided at either boundary. No fallback probe leaves \([0,1]\). The stencils recover quadratic derivatives at the boundary up to floating-point error, so the quadratic maps in Theorems 1 and 2 are realized by either analytic gradients or this fallback in the quadratic regime. On a general nonlinear objective, finite differences remain approximate; the theorem is not a claim about an arbitrary approximate-gradient map.

## Theorem 1: ordinary box-projected gradient descent

Let

\[
F(x)=\tfrac12 x^\top Hx-h^\top x
\]

with \(H\succ0\), and let \(C=[0,1]^n\). The constrained minimizer \(x_C^\star\) is unique. Consider

\[
T(x)=\Pi_C\left(x-\alpha\nabla F(x)\right).
\]

If \(L_H\ge\lambda_{\max}(H)\) and \(0<\alpha<2/L_H\), then

\[
\lVert T(x)-x_C^\star\rVert_2
\le q_H\lVert x-x_C^\star\rVert_2,
\qquad
q_H=\max_{\lambda\in\sigma(H)}|1-\alpha\lambda|<1.
\]

Reason: \(x_C^\star\) is a fixed point of the projected-gradient map, Euclidean projection onto the box is nonexpansive, and the unprojected error map is \(I-\alpha H\). This theorem describes the path with preconditioning disabled.

## Theorem 2: box-projected diagonal preconditioning

Under the same quadratic and box assumptions, let \(P\succ0\) be diagonal and define

\[
T_P(x)=\Pi_C^P\left(x-\alpha P^{-1}\nabla F(x)\right),
\]

where \(\Pi_C^P\) is projection in the norm \(\lVert v\rVert_P^2=v^\top Pv\). For a diagonal \(P\) and an axis-aligned box, this projection is the coordinate clipping performed by the coordinator.

If \(L_P\ge\lambda_{\max}(P^{-1/2}HP^{-1/2})\) and \(0<\alpha<2/L_P\), then

\[
\lVert T_P(x)-x_C^\star\rVert_P
\le q_P\lVert x-x_C^\star\rVert_P,
\]

with

\[
q_P=\max_{\lambda\in\sigma(P^{-1/2}HP^{-1/2})}|1-\alpha\lambda|<1.
\]

Proof. The constrained minimizer satisfies the variational inequality for \(F\), so it is a fixed point of \(T_P\). Projection onto a closed convex set is nonexpansive in the projection norm. With \(y=P^{1/2}(x-x_C^\star)\), the difference between the two unprojected affine maps is

\[
P^{1/2}\left[(x-\alpha P^{-1}\nabla F(x))
-(x_C^\star-\alpha P^{-1}\nabla F(x_C^\star))\right]
=\left(I-\alpha A\right)y.
\]

The SPD matrix \(A=P^{-1/2}HP^{-1/2}\) has positive eigenvalues. The step condition places every eigenvalue of \(I-\alpha A\) strictly inside the unit interval in magnitude. Nonexpansiveness of \(\Pi_C^P\) then gives the stated \(P\)-norm contraction.

This theorem covers both diagonal-preconditioned modes only when the bound and update use the same \(P\). Bounding \(H\) while executing \(P^{-1}H\) does not establish this result.

## Descent and projected Armijo

For an unprojected preconditioned step \(x^+=x-\alpha P^{-1}g\), an \(L_P\)-smooth objective in the \(P\) geometry satisfies

\[
F(x^+)\le F(x)
-\alpha\left(1-\frac{\alpha L_P}{2}\right)
\lVert g\rVert_{P^{-1}}^2.
\]

The coefficient is positive when \(0<\alpha<2/L_P\), so the subtracted quantity is nonnegative. For a projected displacement \(s=\Pi_C^P(x-\alpha P^{-1}g)-x\), projection optimality gives \(g^\top s\le-\lVert s\rVert_P^2/\alpha\). Therefore,

\[
F(x+s)\le F(x)
-\left(\frac1\alpha-\frac{L_P}{2}\right)\lVert s\rVert_P^2
\]

under the same segment-valid smoothness bound.

Projected Armijo evaluates the realized box-constrained displacement rather than the norm of the preconditioned direction. For direction \(d\), trial step \(\alpha\), objective gradient \(g\), and

\[
s=\Pi_C(x-\alpha d)-x,
\]

the implemented acceptance condition is

\[
g^\top s\le0,
\qquad
F(x+s)\le F(x)+c\,g^\top s,
\qquad 0<c<1.
\]

The preconditioned direction is \(d=P^{-1}g\). If all backtracking trials fail, then the line search returns the unchanged state. Noise is applied separately and remains subject to the accepted-step energy guard.

Gradient normalization changes the iteration map covered by the spectral theorem. The configuration therefore requires projected Armijo when gradient normalization and the stability guard are enabled together.

## Active sets and state-dependent curvature

Built-in quadratic curvature is global. Built-in directed and asymmetric hinge bounds use the maximum curvature across their active and inactive regions. This worst-case treatment covers a proposal that crosses a hinge boundary; using only the starting active set would not.

Gate-benefit values are external inputs to the objective. Before any solver dispatch, the coordinator converts every `GateBenefitCoupling` and `DampedGateBenefitCoupling` delta to a finite float inside a read-only top-level constraint snapshot. The snapshot remains fixed for the complete solver call, and the caller's original constraint mapping is restored afterward. Energy, gradient, curvature, and acceptance checks therefore observe the same benefit value. Adapter-owned term weights remain an explicit versioned source of between-step objective changes.

For a plain gate-benefit term, and for a damped term with power \(p=1\), the gate energy is linear and its Hessian contribution is zero. For a damped term

\[
F_{\mathrm{damped}}(\eta)=-a\eta^p,
\]

the coefficient \(a\) contains the weight, damping, signed delta scaling, and frozen benefit. When \(p\ge2\), the implementation reports the exact box-wide absolute diagonal bound

\[
\left|\frac{d^2F_{\mathrm{damped}}}{d\eta^2}\right|
\le |a|p(p-1),
\qquad \eta\in[0,1].
\]

When \(1<p<2\) and \(a\ne0\), the second derivative diverges as \(\eta\) approaches zero, so no finite gradient-Lipschitz bound exists on the closed box. A fixed-step guarded run fails closed unless projected Armijo is enabled. When the frozen coefficient \(a\) is zero, the term and its curvature report are zero. Powers below one are rejected at construction.

The reported quantity is an absolute curvature bound. It does not assert positive curvature. In particular, \(-a\eta^p\) is concave when \(a>0\). The contraction theorems above require the total quadratic Hessian to be SPD; an absolute component bound cannot supply that premise.

For nonlinear modules, state-dependent curvature, and custom couplings, a value computed only at the starting point does not automatically bound the entire proposal. The fixed-step theorem applies when the reported component bounds remain valid along the segment from the accepted state to the projected proposal. Otherwise, use projected Armijo to test progressively shorter realized displacements. The final monotonicity check can reject and restore an uphill proposal, but rejection alone does not prove contraction or progress.

The sampled curvature auditor detects observed underreporting at sampled states. It does not prove segment validity between those states.

## Weighted Jacobi relationship

For the unconstrained quadratic system \(Hx=h\), choose \(P=D=\operatorname{diag}(H)\). The implemented preconditioned update is

\[
x^{(t+1)}=x^{(t)}-\alpha D^{-1}(Hx^{(t)}-h).
\]

This is weighted Jacobi with relaxation parameter \(\alpha\). It reduces to classical Jacobi when \(\alpha=1\). If box clipping activates, then the path is projected weighted Jacobi. If the epsilon floor changes \(P\) from \(D\), then it remains a diagonally preconditioned iteration rather than exact Jacobi.

## Small-Gain adapter

`SmallGainWeightAdapter` ranks coupling families by a smoothed value-to-cost score. Value is approximated by gradient norm squared. Cost is the estimated increase in the row-sum bound in the active update geometry. The adapter applies bounded weight changes while staying inside a global predicted-spend cap.

Current implementation boundary:

- It enforces a global predicted-spend cap.
- It records row margin telemetry.
- It does not enforce per-edge row-incidence booking.
- Its benefit depends on the graph and objective. Fewer steps and lower final energy remain empirical outcomes.

The allocator does not replace either theorem or projected Armijo.

## Telemetry

`inspect_state()` separates objective curvature from update curvature:

- `lipschitz_bound` is the raw, unpreconditioned \(L_H\) estimate.
- `update_lipschitz_bound` is \(L_P\) when a preconditioned mode is active and \(L_H\) otherwise.
- `precision_diagonal` is the cached curvature vector before the positive epsilon floor.
- `preconditioner_diagonal` is the exact positive \(P\) used by the update and normalized bound.

When `log_step_cap_slack=True`, the coordinator records

\[
\text{step_cap_slack}=\frac{2}{L_{\mathrm{update}}}-\alpha_{\mathrm{used}}.
\]

This quantity measures distance from the uncushioned \(2/L\) step boundary. It is not the spectral contraction factor \(q_H\) or \(q_P\). The configuration name `log_contraction_margin` and metric names `contraction_margin` and `contraction_margins` remain deprecated compatibility aliases for the same step-cap slack value.

Detailed Lipschitz telemetry reports row sums, row margins, global margin, family costs, edge costs, geometry, and the preconditioner used for those normalized rows. `EnergyBudgetTracker` also records backtracks, acceptance provenance, precision summaries, and Small-Gain allocation fields.

## Tuning guidance

If the update Lipschitz bound is large, then the cap can make steps small. Diagnose the geometry before changing the mechanism:

1. Compare `lipschitz_bound` with `update_lipschitz_bound`.
2. Inspect `preconditioner_diagonal` and normalized row contributions.
3. Check that every custom curvature report covers the proposed segment.
4. Use projected Armijo for state-dependent or uncertain bounds.
5. Reduce `stability_cap_fraction` when more distance from the \(2/L\) boundary is required.
6. Use `SmallGainWeightAdapter` only when adaptive coupling weights are part of the experiment.

For a minimal deterministic check, use `noise_mode="none"`, `weight_adapter=None`, `stability_guard=True`, and `assert_monotonic_energy=True`.

## Curvature-contract audit

Run the sampled finite-difference auditor with:

```powershell
uv run python -m experiments.audit_curvature_contract --samples 32 --strict
```

Strict mode exits nonzero when a reported module or coupling bound falls below an observed Hessian entry. The recorded seven-family run covered 6,080 component-state records with no observed underreporting. This audit detects sampled violations; it does not prove a global or segment-valid bound between sampled states.

## Test coverage

Current stability-related coverage includes:

- `tests/test_precision_conditioning.py`: raw and normalized Gershgorin bounds, randomized SPD spectral-radius checks for \(I-\alpha P^{-1}H\), the one-dimensional small-precision counterexample, scale invariance, box-projected contraction, active-set crossing, custom-bound requirements, and projected Armijo exhaustion.
- `tests/test_end_to_end_relaxation.py`: accepted energy monotonicity, gradient behavior, box-edge and isolated-node finite differences, PSON gradient coverage, step-cap telemetry, large-noise rejection, and sparse Small-Gain behavior.
- `tests/test_small_gain_weight_adapter.py`: allocation ranking, bounds, fallback behavior, coordinator cost-key integration, and monotone accepted energy on a small graph.
- `tests/test_stiffness_updates.py`: stiffness update descent, coupling curvature in the precision cache, weighted-Jacobi behavior, and rejected-step restoration.

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Use `uv run -m pytest tests -v` when the local `uv` cache is accessible in the active environment.

## Summary

The quadratic guarantee follows the implemented map. Ordinary projected gradient descent is controlled by \(H\). Diagonal-preconditioned projected relaxation is controlled by \(P^{-1/2}HP^{-1/2}\) and contracts in the \(P\)-norm when the normalized bound is valid. Nonlinear and custom compositions require segment-valid curvature reports or projected Armijo, followed by the accepted-step restoration check.

## References

- Boyd, S., Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. Use: smooth-objective descent conditions and projected-gradient analysis. Local implication: the cap must bound curvature in the geometry of the executed update. Limits: custom state-dependent compositions still need a segment-valid bound or line search.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*. SIAM. Use: weighted Jacobi and stationary-iteration convergence. Local implication: diagonal stiffness scaling is weighted Jacobi for the stated quadratic system and classical Jacobi at \(\alpha=1\). Limits: the repository does not implement a dedicated Gauss-Seidel stiffness schedule.
- Varga, R. S. (2004). *Gersgorin and His Circles*. Springer. Use: row-sum eigenvalue bounds. Local implication: normalized component bounds can upper-bound \(\lambda_{\max}(P^{-1/2}HP^{-1/2})\). Limits: a truthful component bound can still be conservative.
- Vidyasagar, M. (1993). *Nonlinear Systems Analysis*. Prentice Hall. Use: small-gain framing for bounded feedback interactions. Local implication: update-geometry margin telemetry can constrain adaptive coupling spend. Limits: the current allocator is not a full nonlinear stability certificate.
