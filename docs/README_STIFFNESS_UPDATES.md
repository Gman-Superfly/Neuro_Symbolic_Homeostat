# Stiffness-based updates and the precision layer

Status: implemented and tested with `use_stiffness_updates=True`.

Scope: diagonal curvature aggregation, preconditioned updates, normalized stability bounds, and the weighted-Jacobi relationship.

## Update rule

The coordinator aggregates a nonnegative diagonal curvature cache \(\Lambda\). Before a stiffness update it constructs one positive diagonal

\[
P=\operatorname{diag}(p_1,\ldots,p_n),
\qquad
p_i=\max(\varepsilon_{\mathrm{stiff}},\Lambda_{ii}).
\]

It then applies

\[
\eta_i^+=\Pi_{[0,1]}\left(
\eta_i-\alpha\frac{(\nabla F)_i}{p_i}
\right).
\]

Here \(\alpha\) is the guarded scalar step and \(\Pi_{[0,1]}\) clips one coordinate to the unit interval. The positive floor prevents division by zero. `use_precision_preconditioning=True` follows the same rule with `precision_epsilon` as the floor. Both paths obtain \(P\) from the same helper, and the stability estimator consumes that exact diagonal.

This is a diagonal-preconditioned direction in a general nonlinear problem. It is not a full Newton step because the update does not invert the full Hessian.

## Curvature aggregation

Modules implementing `SupportsPrecision` contribute `curvature(eta)` to the diagonal cache. Supported couplings add diagonal curvature as follows:

- `QuadraticCoupling` contributes its constant diagonal curvature to both endpoints.
- `DirectedHingeCoupling` and `AsymmetricHingeCoupling` contribute active-state diagonal curvature to \(P\).
- A custom `SupportsCouplingCurvature` implementation contributes its reported diagonal bounds.
- `GateBenefitCoupling` is linear in the gate for a frozen caller-supplied benefit, so it contributes force and zero curvature.
- `DampedGateBenefitCoupling` with power \(p=1\) is also linear. For \(p\ge2\), writing its gate energy as \(-a\eta^p\), it contributes the exact box-wide absolute diagonal bound \(|a|p(p-1)\). This magnitude bound does not imply positive curvature; the term is concave when \(a>0\), and contraction still requires an SPD total Hessian. For nonzero \(a\) and \(1<p<2\), no finite closed-box Lipschitz bound exists at zero; a fixed-step guarded run fails closed unless projected Armijo is enabled. A zero frozen coefficient gives a zero term and curvature report. Powers below one are rejected.

The step bound uses a different active-set rule where required. Built-in hinges report their worst-case curvature across active and inactive regions to the stability estimator. This covers a proposal that crosses a hinge boundary even when the starting-state precision cache has no active hinge contribution.

The local Hessian terms in the raw and normalized Gershgorin bounds are independently finite-differenced from local gradients. They are not copied from `SupportsPrecision.curvature`. Coupling curvature feeds both the cache and Hessian-bound paths where supported. The stability estimator normalizes its separately composed Hessian bounds by the exact \(P\) from the cache path.

Before solver dispatch, gate-benefit deltas are converted to finite floats inside a read-only top-level constraint snapshot. The snapshot remains fixed for the complete solver call, and the original caller mapping is restored afterward. The energy, force, curvature, line search, and acceptance guard therefore consume the same benefit value.

## Stability in the executed geometry

For a quadratic objective with SPD Hessian \(H\), define

\[
A=P^{-1/2}HP^{-1/2}.
\]

The normalized Gershgorin bound is

\[
L_P=\max_i\left(
\frac{\bar h_{ii}}{p_i}
+\sum_{j\ne i}\frac{\bar h_{ij}}{\sqrt{p_i p_j}}
\right),
\]

where the barred values bound the corresponding absolute Hessian entries. A valid component contract gives \(\lambda_{\max}(A)\le L_P\). The guard caps the step by

\[
\alpha_{\mathrm{used}}
=\min\left(\alpha_{\mathrm{requested}},\gamma\frac{2}{L_P}\right),
\qquad 0<\gamma<1.
\]

For \(0<\alpha_{\mathrm{used}}<2/L_P\), the box-projected preconditioned map contracts toward the unique constrained minimizer in the \(P\)-norm. The contraction factor is

\[
q_P=\max_{\lambda\in\sigma(A)}|1-\alpha_{\mathrm{used}}\lambda|<1.
\]

This result depends on using the same \(P\) in the direction and bound. An unpreconditioned bound on \(H\) alone does not certify a \(P^{-1}H\) update.

Nonlinear, state-dependent, and custom curvature reports must cover the realized proposal segment for the fixed-step theorem to apply. Projected Armijo is the checked fallback. With objective gradient \(g\), descent direction \(d\), and realized displacement \(s=\Pi_{[0,1]^n}(\eta-\alpha d)-\eta\), it accepts only when

\[
g^\top s\le0,
\qquad
F(\eta+s)\le F(\eta)+c\,g^\top s.
\]

Exhausted backtracking returns the unchanged state.

## Weighted Jacobi relationship

For an unconstrained quadratic system

\[
F(x)=\tfrac12x^\top Hx-h^\top x,
\qquad D=\operatorname{diag}(H),
\]

choose \(P=D\). The update becomes

\[
x^{(t+1)}=x^{(t)}-\alpha D^{-1}(Hx^{(t)}-h).
\]

This is weighted Jacobi with relaxation parameter \(\alpha\). It is classical Jacobi at \(\alpha=1\). Box clipping gives projected weighted Jacobi. An active epsilon floor that changes \(P\) from \(D\) gives a diagonally preconditioned iteration rather than exact Jacobi.

Gaussian belief propagation addresses the same Gaussian mean problem through messages and cavity precisions. The repository does not implement GaBP message updates.

## PSON interaction

Precision-orthogonal noise uses the curvature cache to bias a tangent vector toward coordinates with lower reported curvature. Inverse-precision weighting generally breaks the first projection, so the implementation projects again after weighting. It then normalizes the vector and applies one uniform box-feasible scalar. The uniform scalar preserves first-order tangency; per-coordinate clipping generally would not.

When the ordinary gradient norm is below \(10^{-8}\), the projection helper returns the draw unchanged because the gradient does not define a reliable normal. This branch is stationary-point exploration rather than a Hessian-null-space claim.

## Configuration

```python
coord = EnergyCoordinator(
    modules=modules,
    couplings=couplings,
    use_stiffness_updates=True,
    stiffness_epsilon=1e-8,
    stability_guard=True,
    stability_cap_fraction=0.9,
    log_step_cap_slack=True,
)
```

`step_cap_slack` equals \((2/L_P)-\alpha_{\mathrm{used}}\). It is not the spectral contraction factor \(q_P\). Names containing `contraction_margin` remain deprecated compatibility aliases.

## Diagnostics and verification

`coord.inspect_state(etas)` reports:

- `precision_diagonal`: the cached curvature before the positive floor.
- `preconditioner_diagonal`: the exact \(P\) used by the update.
- `lipschitz_bound`: the raw unpreconditioned Hessian bound.
- `update_lipschitz_bound`: the normalized \(L_P\) bound for an active preconditioned mode.

Relevant tests cover stiffness descent, agreement between the bounded and executed diagonal, weighted-Jacobi behavior, randomized SPD spectral radius, box-projected contraction, the small-precision counterexample, hinge active-set crossing, and state restoration after rejection or exhausted Armijo search.
