# Metric-aware orthogonal noise: why it is omitted here (and where it fits)

Status: Clarification note (this repo)
Scope: PSON noise, metric awareness, and controller design

---

## Summary

- We inject orthogonal noise (PSON) and optionally make it metric‑aware via M‑orthogonal projection.

- In this repo, we do NOT use a separate `MetricAwareNoiseController` class. Magnitude scheduling is handled by the standard controllers (Orthogonal/Precision) and the stability guards.

- In the main “Complexity from Constraints” repository, a metric‑aware controller is available because the system integrates richer metrics (e.g., Fisher/curvature‑derived) and may wish to adapt noise magnitude directly from that geometry.

---

## What we do here

### 1) Orthogonal or M-orthogonal projection

- Default: Euclidean orthogonal projection of noise to the gradient tangent plane.
- Metric‑aware option: When a problem metric \(M\) is provided, we project with respect to \(M\) and re‑project after precision weighting to preserve \(M\)‑orthogonality.

Mathematically (when \(M\) is provided):

\[
z_{\perp_M} \;=\; z \;-\; \frac{z^\top M g}{g^\top M g}\, g \,.
\]

This is already implemented by `project_noise_metric_orthogonal` and used by the coordinator when metric inputs are supplied.

### 2) Magnitude scheduling via existing controllers

- OrthogonalNoiseController: Adapts noise based on descent rate, backtracks, and gradient rotation.
- PrecisionNoiseController: Additionally redistributes noise along low‑curvature directions (precision‑aware) and re‑projects to maintain orthogonality.

These controllers are metric‑agnostic; the geometry enters at the projection layer. This keeps the design simple and predictable.

---

## Why we do not use MetricAwareNoiseController here

1) Redundancy with projection
   Metric awareness that matters most, staying tangent to level sets, is already captured by M-orthogonal projection and re-projection after curvature weighting. A separate controller class would duplicate purpose without clear benefit for this repo’s scope.

2) Compact API
   This repository emphasizes a compact path: stiffness-based updates, Small-Gain stability, and PSON with optional M-projection. Adding another controller increases surface area and cognitive load without improving core outcomes for included demos and tests.

3) Telemetry and guard behavior remain the same
   Stability and acceptance are handled by the same guards (monotone energy, Small‑Gain projection). Introducing an extra controller does not strengthen those scoped bounds here.

---

## When a metric-aware controller is useful (main repo)

In the “Complexity from Constraints” main repo, a metric‑aware controller can be valuable when:

- The metric \(M\) is principled (e.g., Fisher information, Gauss‑Newton, or a validated application‑specific metric) and changes meaningfully across steps.
- You want the noise magnitude policy itself to depend on metric quantities (e.g., \(g^\top M g\), contraction margins computed under \(M\), or anisotropic exploration budgets derived from \(M\)).
- You are working in a Riemannian or natural‑gradient setting (Riemannian Langevin), where both direction (projection) and scale (controller) should be consistent with the manifold geometry.

In those cases, a dedicated `MetricAwareNoiseController` can encode M‑conditioned signals and budgets explicitly (and still call the same projection function for direction).

---

## Design guidance (if you add it later)

- Keep projection independent: continue using `project_noise_metric_orthogonal` for direction.
- Have the metric‑aware controller decide only the magnitude schedule (and optional per‑coordinate redistribution), using signals such as:
  - \(g^\top M g\) (metric energy in the gradient direction)
  - Contraction margins computed under M‑aware bounds
  - Rotation/curvature proxies compatible with \(M\)
- Preserve stability: keep Small‑Gain projection and monotone acceptance unchanged.
- Preserve PSON’s spirit: after any redistribution, re‑project to remain M‑orthogonal.

Implementation sketch (conceptual):

```python
class MetricAwareNoiseController(OrthogonalNoiseController):
    def step(self, grad, energy_drop_ratio, backtracks, iter_idx, *, M=None, Mv=None):
        # Build metric-aware signals (e.g., gT_M_g = g·(M g))
        # Map to a [0, 1] schedule like the base class, but modulated by M-signals
        # Return magnitude only; direction remains handled by M-orthogonal projection
        ...
```

---

## Bottom line

- This repo: M-orthogonal projection gives the geometric behavior used by the current demos and tests; existing controllers handle magnitude scheduling. A separate `MetricAwareNoiseController` would add complexity without a tested benefit here.
- Main repo: If you rely on a principled metric and want the noise policy to adapt to it directly, keep or use the metric-aware controller there.


