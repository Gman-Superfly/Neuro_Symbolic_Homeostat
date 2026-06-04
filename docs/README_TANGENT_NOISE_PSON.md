# Precision-scaled orthogonal noise (PSON): design and usage

Status: available in this repository (metric-aware projection supported)
Scope: What tangent (orthogonal) noise is, what it controls, how we implement and use it (with optional metric awareness), and how to validate it.

---

## 1. What and why

Standard isotropic noise perturbs parameters in arbitrary directions, which can easily inject an uphill component against the gradient, causing non‑monotone steps and fragile acceptance logic.

PSON instead injects noise in the tangent plane orthogonal to the gradient of the energy:
- First-order neutral perturbation: first‑order energy change vanishes (noise does not “fight” descent).
- Targeted search: exploration is directed into flat or uncertain directions (null‑space).
- Curvature‑aware: we scale noise inversely with diagonal curvature so slack variables explore more, stiff variables less.

Optionally, when a problem metric \(M\) is provided, we project noise to be orthogonal under that metric (M‑orthogonal), which aligns exploration with the true geometry (e.g., Fisher/Gauss‑Newton).

---

## 2. Mathematics

### 2.1 Euclidean orthogonal projection
Given gradient \(g\) and a raw noise vector \(z\), the orthogonal component is:

\[
z_{\perp} \;=\; z \;-\; \frac{z^\top g}{g^\top g}\, g \,.
\]

Property: \(g^\top z_{\perp} = 0\) (first‑order energy change from the noise term is zero).

### 2.2 Metric-orthogonal projection (optional)
Given a symmetric positive definite metric \(M\), we project with respect to \(M\):

\[
z_{\perp_M} \;=\; z \;-\; \frac{z^\top M g}{g^\top M g}\, g \,,
\]

so \(g^\top M z_{\perp_M} = 0\). We also re‑project after curvature (precision) re‑weighting to preserve \(M\)‑orthogonality.

### 2.3 Precision-aware scaling
Let \(\Lambda\) be the diagonal curvature (precision) aggregated from modules and convex couplings. We re‑weight noise components by \(w_i \propto 1 / (\Lambda_{ii} + \varepsilon)\) before re‑normalization and orthogonal re‑projection. This allocates exploration budget toward low‑curvature directions.

---

## 3. Energy effect and guards (intuition)

Let the energy be \(F\), current point \(x\), gradient \(g = \nabla F(x)\), and injected orthogonal noise \(\delta\) with \(g^\top \delta = 0\). A second‑order Taylor approximation gives:

\[
F(x + \delta) \approx F(x) + \tfrac{1}{2}\delta^\top H \delta \,,
\]

so the first‑order increase vanishes. Second-order terms can still increase energy. The implementation manages this with (i) down‑only acceptance, (ii) Small‑Gain projection limiting loop gains, and (iii) small curvature‑aware magnitude.

In practice we keep:
- Monotone acceptance: reject steps with \( \Delta F > 0 \) (or use the free‑energy guard, if enabled).
- Stability projection (Small‑Gain): cap couplings to preserve contraction.
- Precision‑scaled noise magnitude: keep second‑order effects small in stiff directions.

---

## 4. Implementation in this repo

### 4.1 Core projection utilities
We implement both Euclidean and metric‑aware projections:

```97:154:core/energy.py
def project_noise_orthogonal(
    noise: np.ndarray,
    grad: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    # ... returns z - ((z·g)/||g||^2) g

def project_noise_metric_orthogonal(
    noise: np.ndarray,
    grad: np.ndarray,
    *,
    M: np.ndarray | None = None,
    Mv: callable | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    # ... returns z - ((z^T M g)/(g^T M g)) g (using Mv(g) when provided)
```

### 4.2 Coordinator integration
During each relaxation step, if noise is enabled, we:
1) Draw raw noise.
2) Project orthogonally (Euclidean or metric‑aware).
3) Optionally re‑weight by inverse curvature (precision) and re‑project to preserve orthogonality.
4) Normalize to the desired magnitude and add to the update.

```399:451:core/coordinator.py
if self.enable_orthogonal_noise:
    raw_noise = np.random.normal(0, 1, size=grad_vector.shape)
    if self.metric_aware_noise_controller and (self.metric_vector_product is not None or self.metric_matrix is not None):
        noise_vector = project_noise_metric_orthogonal(raw_noise, grad_vector, M=self.metric_matrix, Mv=self.metric_vector_product)
    else:
        noise_vector = project_noise_orthogonal(raw_noise, grad_vector)
    if self.precision_aware_noise_controller:
        # compute curvature weights (∝ 1 / (eps + curvature))
        # re-weight then re-project to keep orthogonality
        ...
    # scale to magnitude and add into the step direction
```

---

## 5. Configuration and usage

### 5.1 Basic Euclidean tangent noise
- `enable_orthogonal_noise=True`
- `noise_magnitude`: base magnitude (e.g., 1e‑2)
- Optional anneal: `noise_schedule_decay` (e.g., 0.99)
- Optional automatic scheduling: `auto_noise_controller=True` (uses OrthogonalNoiseController)

### 5.2 Precision-aware variant
- `precision_aware_noise_controller=True` (uses PrecisionNoiseController under the hood)
- Ensure your modules implement `SupportsPrecision` where possible; the coordinator aggregates curvature (including convex couplings) for re‑weighting.

### 5.3 Metric-aware variant (optional)
- Provide a metric via `coord.metric_matrix = M` (or `metric_vector_product`), and set `metric_aware_noise_controller=True`.
- The coordinator then uses M‑orthogonal projection (and re‑projection after precision re‑weighting).

---

## 6. Demos and validation

### 6.1 Projection Properties and Relaxation
Run:

```powershell
uv run python -m experiments.demo_metric_orthogonal
```

This prints Euclidean and metric‑orthogonal dot‑products (≈ 0) and runs a short relaxation with M‑orthogonal noise enabled.

### 6.2 Stability and acceptance
Our test suite exercises precision‑aware weighting and Small‑Gain stability. For broader stability guidance and proofs, see:

- `docs/STABILITY_GUARANTEES.md`
- `docs/README_SMALLGAIN.md`

---

## 7. Design notes and trade-offs

- We keep metric awareness in the projection step (direction) rather than in a dedicated metric‑aware magnitude controller. This simplifies the API while retaining the key geometric property (orthogonality). See `docs/README_METRIC_AWARE_NOISE.md` for rationale.
- Precision scaling is diagonal by design for lower overhead and simpler composition; it pairs with Small‑Gain in the tested mixed regimes. Full metric‑curvature allocations (dense) are possible when the extra cost is justified.

---

## 8. FAQ

**Q: Will tangent noise ever increase energy?**
A: First‑order contribution is zero by construction. Second‑order effects can increase energy if noise is too large in stiff directions. We mitigate this with precision scaling, Small‑Gain, and monotone acceptance.

**Q: Why re‑project after precision re‑weighting?**
A: The re‑weighting alters direction; re‑projecting restores orthogonality (Euclidean or M‑orthogonal).

**Q: When should I use the metric‑aware variant?**
A: When you have a meaningful SPD metric (e.g., Fisher/Gauss‑Newton) and want exploration aligned with that geometry. Otherwise, Euclidean projection is simpler and has lower overhead.

---

## 9. Quick recipe

```python
coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={},
    enable_orthogonal_noise=True,
    noise_magnitude=1e-2,
    precision_aware_noise_controller=True,   # curvature-aware redistribution
    # Optional metric-aware projection:
    metric_aware_noise_controller=True,
)
# Optional: SPD metric
coord.metric_matrix = np.diag([1.0, 10.0])
```

Run your relaxation and monitor ΔF and contraction margins; accepted steps should remain consistent with the configured guards.


