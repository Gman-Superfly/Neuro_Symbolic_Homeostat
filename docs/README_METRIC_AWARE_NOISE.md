# Metric-aware orthogonal noise: why it is omitted here (and where it fits)

Status: Clarification note (this repo)
Scope: PSON noise, metric awareness, and controller design

---

## Summary

- We inject orthogonal noise (PSON) and optionally use an SPD metric to choose the projection correction.

- In this repo, we do NOT use a separate `MetricAwareNoiseController` class. Magnitude scheduling is handled by the standard controllers (Orthogonal/Precision) and the stability guards.

- In the main “Complexity from Constraints” repository, a metric‑aware controller is available because the system integrates richer metrics (e.g., Fisher/curvature‑derived) and may wish to adapt noise magnitude directly from that geometry.

---

## What we do here

### 1) Euclidean or metric-consistent tangent projection

- Default: Euclidean orthogonal projection of noise to the gradient tangent plane.
- Metric-aware option: When a problem metric \(M\) is provided, we compute the metric gradient \(g_M=M^{-1}g\) and project onto the energy tangent plane along that direction.

Mathematically (when \(M\) is provided):

\[
z_{\perp_M} \;=\; z \;-\; \frac{g^\top z}{g^\top M^{-1}g}\, M^{-1}g \,.
\]

This construction gives \(g^\top z_{\perp_M}=0\). The coordinator accepts either a dense `metric_matrix` or a matrix-free `metric_solve` callback that applies \(M^{-1}\).

### 2) Magnitude scheduling via existing controllers

- OrthogonalNoiseController: Adapts noise based on descent rate, backtracks, and gradient rotation.
- PrecisionNoiseController: Additionally redistributes noise along low‑curvature directions (precision‑aware) and re‑projects to maintain orthogonality.

These controllers are metric‑agnostic; the geometry enters at the projection layer. This keeps the design simple and predictable.

---

## Why we do not use MetricAwareNoiseController here

1) Redundancy with projection
   Tangency is enforced by the projection. The metric selects the correction direction through \(M^{-1}g\), while the controller only schedules magnitude. A separate controller class would duplicate that responsibility without a measured benefit in the included experiments.

2) Compact API
   This repository emphasizes a compact path: stiffness-based updates, a curvature-based step cap, and PSON with optional metric projection. Adding another controller increases surface area without a measured benefit in the included demos and tests.

3) Telemetry and guard behavior remain the same
   Stability and acceptance are handled by the coordinator's step cap and accepted-step guard. Introducing an extra controller does not strengthen those scoped bounds here.

---

## When a metric-aware controller is useful (main repo)

In the “Complexity from Constraints” main repo, a metric‑aware controller can be valuable when:

- The metric \(M\) is principled (e.g., Fisher information, Gauss‑Newton, or a validated application‑specific metric) and changes meaningfully across steps.
- You want the noise magnitude policy itself to depend on metric quantities such as \(g^\top M^{-1}g\), contraction margins computed under \(M\), or anisotropic exploration budgets derived from \(M\).
- You are working in a Riemannian or natural‑gradient setting (Riemannian Langevin), where both direction (projection) and scale (controller) should be consistent with the manifold geometry.

In those cases, a dedicated `MetricAwareNoiseController` can encode M‑conditioned signals and budgets explicitly (and still call the same projection function for direction).

---

## Design guidance (if you add it later)

- Keep projection independent: continue using `project_noise_metric_orthogonal` for direction.
- Have the metric‑aware controller decide only the magnitude schedule (and optional per‑coordinate redistribution), using signals such as:
  - \(g^\top M^{-1}g\) (squared norm of the metric gradient)
  - Contraction margins computed under M‑aware bounds
  - Rotation/curvature proxies compatible with \(M\)
- Preserve stability: keep the curvature-based step cap and accepted-step guard unchanged.
- Preserve first-order tangency: after any redistribution, re-project so \(g^\top\delta=0\) under the selected geometry.

Implementation sketch (conceptual):

```python
class MetricAwareNoiseController(OrthogonalNoiseController):
    def step(self, grad, energy_drop_ratio, backtracks, iter_idx, *, metric_solve):
        # Build metric-aware signals, for example grad.T @ metric_solve(grad).
        # Map to a [0, 1] schedule like the base class, but modulated by M-signals
        # Return magnitude only; projection still determines direction.
        ...
```

---

## Bottom line

- This repo: metric-consistent tangent projection supplies the geometric behavior used by the current demo and tests; existing controllers handle magnitude scheduling. A separate `MetricAwareNoiseController` has no measured benefit here.
- Main repo: If you rely on a principled metric and want the noise policy to adapt to it directly, keep or use the metric-aware controller there.
