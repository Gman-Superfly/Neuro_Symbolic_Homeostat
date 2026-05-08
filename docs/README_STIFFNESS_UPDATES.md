# Stiffness-based updates and precision layer

Status: available in this repository (enabled via `use_stiffness_updates=True`)
Scope: Per-coordinate Newton/Jacobi-style updates, curvature aggregation, and precision scaling.

---

## 1. Core concept

Instead of gradient descent with a heuristic learning rate (`η ← η - α·∇F`), we perform per-coordinate updates based on local stiffness (curvature):

\[
\eta_i \leftarrow \eta_i - \frac{(\nabla F)_i}{\Lambda_{ii} + \varepsilon}
\]

where \(\Lambda_{ii}\) is the aggregated diagonal curvature (precision) for coordinate \(i\).

This corresponds to:
- **Newton's Method** (diagonal approximation)
- **Jacobi Iteration** (solving \(Jx = h\) where \(J \approx \Lambda\))
- **Gaussian Belief Propagation** (message passing equivalent for Gaussian models)

---

## 2. Curvature aggregation

The coordinator aggregates positive diagonal curvature from two sources:

### 2.1 Local Modules
Modules implementing `SupportsPrecision` provide `curvature(η)`.
- Stiff modules (high certainty) contribute large values to \(\Lambda\).
- Slack/uncertain modules contribute near-zero values.

### 2.2 Couplings
Convex couplings contribute to the diagonal stiffness of connected nodes:
- **QuadraticCoupling**: Adds \(2w\) to both \(\Lambda_i\) and \(\Lambda_j\).
- **HingeCoupling** (Directed/Asymmetric): Adds curvature only when the hinge is active (gap > 0) or in the smoothing region.
- **GateBenefitCoupling**: Linear (force-only); contributes zero curvature.

This aggregation happens automatically in `_update_precision_cache()`.

---

## 3. Why it matters

1.  **Auto-Tuning Step Size**: The step \(\Delta \eta \approx \text{Force} / \text{Stiffness}\). Stiff variables take small, precise steps; slack variables take large steps to find equilibrium. No manual learning rate tuning required for well-modeled problems.
2.  **Geometry-Aware PSON**: Precision-Scaled Orthogonal Noise uses this same \(\Lambda\) to scale exploration noise (\(\xi \propto \Lambda^{-1/2}\)), focusing search on flat directions.
3.  **GaBP Equivalence**: For quadratic problems, this update matches the exact algebraic steps of Gaussian Belief Propagation.

---

## 4. Usage

Enable stiffness-based updates in the coordinator:

```python
coord = EnergyCoordinator(
    modules=...,
    couplings=...,
    use_stiffness_updates=True,  # Enable force/stiffness steps
    stiffness_epsilon=1e-8,      # Regularization for zero-curvature modes
    stability_guard=True,        # Keep stability projection active
)
```

### Compatibility
- **Small-Gain**: Still protects stability by capping coupling weights or global step size if off-diagonal interactions are too strong.
- **Adapters**: Weight adapters (SmallGain, GradNorm) still work by scaling the effective force and stiffness terms.
- **CGBC/wormhole terms**: CGBC forces are linear and divided by the node's total stiffness, ensuring non-local signals respect local constraints.

---

## 5. Implementation details

- **Precision Cache**: `_precision_cache` stores \(\Lambda_{ii}\) values, updated before each relaxation pass.
- **Update Kernel**: When `use_stiffness_updates=True`, the gradient descent kernel divides the gradient by `max(stiffness, eps)`.
- **Preconditioning**: If `use_stiffness_updates=False` but `use_precision_preconditioning=True`, the gradient is scaled by curvature, but a global `step_size` is still applied.

---

## 6. Verification

See `tests/test_stiffness_updates.py` for:
- Exact convergence on quadratic problems in 1 step.
- Equivalence to preconditioned gradient descent.
- Verification that CGBC/wormhole linear forces are preserved.

