# Quadratic stiffness, weighted Jacobi, and Gaussian BP

Status: the synchronous diagonal-preconditioned update is implemented and trajectory-tested. Gaussian belief propagation is literature context.

Scope: exact local algebra for the quadratic update, its contraction condition, and the boundary of the message-passing analogy.

## Implemented quadratic update

Consider

\[
F(x)=\tfrac12x^\top Jx-h^\top x,
\qquad J\succ0.
\]

The gradient is \(\nabla F(x)=Jx-h\), and minimizing \(F\) is equivalent to solving \(Jx=h\). Let \(D=\operatorname{diag}(J)\). When the coordinator's positive update diagonal is exactly \(P=D\), its unconstrained stiffness update is

\[
x^{(t+1)}=x^{(t)}-\alpha D^{-1}(Jx^{(t)}-h).
\]

This is weighted Jacobi with relaxation parameter \(\alpha\). It reduces to classical Jacobi when \(\alpha=1\). The repository default does not imply \(\alpha=1\), so the general implementation must be described as weighted Jacobi.

The coordinator also projects every coordinate into \([0,1]\). If clipping changes a proposal, then the realized path is projected weighted Jacobi. If the configured epsilon floor changes \(P\) from \(D\), then the path remains a diagonal-preconditioned iteration but is not exact Jacobi.

## Algebra

Write

\[
J=D+L+U,
\]

where \(L\) and \(U\) are the strictly lower and upper parts. Classical Jacobi solves

\[
Dx^{(t+1)}=h-(L+U)x^{(t)},
\]

which is equivalent to

\[
x^{(t+1)}=x^{(t)}-D^{-1}(Jx^{(t)}-h).
\]

Adding the scalar \(\alpha\) gives weighted Jacobi. Every coordinate reads neighbors from the previous iterate, so the scheduler remains synchronous.

Gauss-Seidel instead uses

\[
(D+L)x^{(t+1)}=h-Ux^{(t)}.
\]

This triangular solve consumes fresh lower-index values during the same sweep. The current coordinator does not expose a dedicated Gauss-Seidel stiffness mode.

## Convergence in the update geometry

Weighted Jacobi has iteration matrix

\[
I-\alpha D^{-1}J.
\]

Although this matrix need not be symmetric in Euclidean coordinates, it is similar to

\[
I-\alpha D^{-1/2}JD^{-1/2}.
\]

The normalized matrix \(A=D^{-1/2}JD^{-1/2}\) is SPD. If a bound \(L_D\) satisfies \(\lambda_{\max}(A)\le L_D\) and \(0<\alpha<2/L_D\), then

\[
\rho(I-\alpha D^{-1}J)
=\rho(I-\alpha A)<1.
\]

The coordinator forms the normalized Gershgorin bound

\[
L_D=\max_i\left(
\frac{\bar j_{ii}}{d_i}
+\sum_{k\ne i}\frac{\bar j_{ik}}{\sqrt{d_i d_k}}
\right).
\]

The barred entries bound the corresponding absolute Hessian entries. This is the relevant bound for the implemented preconditioned matrix. A row-sum bound on \(J\) alone does not certify \(D^{-1}J\).

For the box-projected update, coordinate clipping is also projection in the \(D\)-norm because \(D\) is diagonal. Projection is nonexpansive in that norm. If \(x_C^\star\) is the unique constrained quadratic minimizer, then

\[
\lVert x^{(t+1)}-x_C^\star\rVert_D
\le q_D\lVert x^{(t)}-x_C^\star\rVert_D,
\]

where

\[
q_D=\max_{\lambda\in\sigma(A)}|1-\alpha\lambda|<1.
\]

The implementation generalizes this result from \(D\) to its exact positive update diagonal \(P\). See `docs/STABILITY_GUARANTEES.md` for the full statement.

## Gaussian BP boundary

Gaussian belief propagation uses edge messages and evolving cavity precisions. When GaBP converges, its mean solves the same linear system \(Jx=h\). General GaBP is not a stationary weighted-Jacobi iteration, and this repository does not implement or test GaBP message updates.

The shared solution identifies the common Gaussian mean problem. It does not make the transient algorithms stepwise equivalent. Walk-summability and other GaBP convergence conditions belong to the message-passing algorithm; the coordinator tests the spectral condition for its own preconditioned iteration.

## Code path

The quadratic stiffness path in `core/coordinator.py` performs the equivalent operations

```python
gradient = compose_gradient(state)              # J @ x - h
preconditioner = get_update_preconditioner()   # exact positive P
update_bound = normalized_gershgorin(P)         # bounds P^-1/2 J P^-1/2
alpha = guarded_step(update_bound)
state = clip_to_box(state - alpha * gradient / preconditioner)
```

The update, normalized bound, and diagnostic `preconditioner_diagonal` refer to the same \(P\).

## References

1. Weiss, Y., Freeman, W. T. (2001). *Correctness of Belief Propagation in Gaussian Graphical Models of Arbitrary Topology.* Neural Computation. Use: correctness of Gaussian BP means when the message procedure converges. Local implication: GaBP and the quadratic coordinator can target the same linear-system solution. Limits: the result does not identify their transient updates.
2. Malioutov, D., Johnson, J. K., Willsky, A. S. (2006). *Walk-sums and belief propagation in Gaussian graphical models.* Journal of Machine Learning Research. Use: walk-sum interpretation and sufficient convergence conditions for GaBP. Local implication: message-passing convergence has its own matrix conditions. Limits: those conditions do not replace the coordinator's weighted-Jacobi condition.
3. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems.* SIAM. Use: Jacobi, weighted Jacobi, and Gauss-Seidel stationary iterations. Local implication: the coordinator's quadratic diagonal-preconditioned path is weighted Jacobi under the conditions stated above. Limits: the repository does not implement a dedicated Gauss-Seidel scheduler.
