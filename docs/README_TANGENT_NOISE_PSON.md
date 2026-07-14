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

Optionally, when a validated SPD problem metric \(M\) is provided, we use its metric gradient to choose the projection correction. The resulting noise remains tangent to the ordinary energy level set.

---

## 2. Mathematics

### 2.1 Euclidean orthogonal projection
Given gradient \(g\) and a raw noise vector \(z\), the orthogonal component is:

\[
z_{\perp} \;=\; z \;-\; \frac{z^\top g}{g^\top g}\, g \,.
\]

Property: \(g^\top z_{\perp} = 0\) (first‑order energy change from the noise term is zero).

### 2.2 Metric-orthogonal projection (optional)
Given a symmetric positive definite metric \(M\) and ordinary gradient \(g=\nabla F(x)\), define the metric gradient \(g_M=M^{-1}g\). The metric projection onto the energy tangent plane is:

\[
z_{\perp_M} \;=\; z \;-\; \frac{g^\top z}{g^\top M^{-1}g}\, M^{-1}g \,,
\]

This gives \(g^\top z_{\perp_M}=\langle z_{\perp_M},g_M\rangle_M=0\), so the noise has zero first-order directional derivative. We also re-project after curvature weighting when metric and precision modes are combined.

### 2.3 Precision-aware scaling
Let \(\Lambda\) be the diagonal curvature (precision) aggregated from modules and convex couplings. We re‑weight noise components by \(w_i \propto 1 / (\Lambda_{ii} + \varepsilon)\) before re‑normalization and orthogonal re‑projection. This allocates exploration budget toward low‑curvature directions.

---

## 3. Energy effect and guards (intuition)

Let the energy be \(F\), current point \(x\), gradient \(g = \nabla F(x)\), and injected orthogonal noise \(\delta\) with \(g^\top \delta = 0\). A second‑order Taylor approximation gives:

\[
F(x + \delta) \approx F(x) + \tfrac{1}{2}\delta^\top H \delta \,,
\]

so the first-order increase vanishes. Second-order terms can still increase energy. The implementation manages this with down-only acceptance, the Gershgorin-style step cap, and a small curvature-aware magnitude.

In practice we keep:
- Monotone acceptance: reject steps with \( \Delta F > 0 \) (or use the free‑energy guard, if enabled).
- Stability guard: cap the update step below the supported quadratic/SPD bound.
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
    metric_solve: Callable[[np.ndarray], np.ndarray] | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    # ... projects along solve(M, g) while preserving g.T @ noise == 0
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
    if self.metric_aware_noise_controller:
        noise_vector = project_noise_metric_orthogonal(
            raw_noise,
            grad_vector,
            M=self.metric_matrix,
            metric_solve=self.metric_solve,
        )
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
- Provide a metric through `metric_matrix=M` or a matrix-free solve through `metric_solve=lambda vector: solve_M(vector)`, and set `metric_aware_noise_controller=True`.
- When precision-aware noise is also enabled, the coordinator re-projects after inverse-curvature weighting.

---

## 6. Demos and validation

### 6.1 Projection Properties and Relaxation
Run:

```powershell
uv run python -m experiments.demo_metric_orthogonal
```

This prints the Euclidean first-order dot product and the equivalent metric inner product, then runs a short relaxation with combined metric and precision-aware noise.

### 6.2 Stability and acceptance
Our test suite exercises precision-aware weighting, the curvature-based step cap, and the optional Small-Gain allocator. For broader stability guidance and proofs, see:

- `docs/STABILITY_GUARANTEES.md`
- `docs/README_SMALLGAIN.md`

### 6.3 Paired problem-family ablation

Run the recorded synthetic ablation with:

```powershell
uv run python -m experiments.ablate_pson_noise --trials 30 --steps 80 --noise-cost-samples 32 --bootstrap-samples 10000
```

The experiment covers quadratic chain, mixed-gate chain, star, dense, ill-conditioned ring, state-dependent quartic, and active-hinge families at sizes 6, 12, and 24. Noise modes are paired by generated seed and raw Gaussian draw. The command writes raw trials and individual draw costs to `logs/pson_noise_ablation.csv`, then writes paired hierarchical bootstrap effects over seeds and draw indices to `logs/pson_noise_ablation_summary.csv`.

Across the recorded 30-seed run, precision-orthogonal noise had lower initial-state curvature cost in all seven families. This supports a mechanism-level curvature-cost claim for the listed generators. It does not establish task-level benefit or performance on real model outputs.

The closed-form reference can be checked with:

```powershell
uv run python -m experiments.validate_pson_reference --samples 100000
```

The controlled nonconvex escape construction can be reproduced with:

```powershell
uv run python -m experiments.benchmark_pson_escape --trials 200 --steps 40 --bootstrap-samples 10000
```

The escape construction deliberately places one lower-curvature double-well coordinate beside seven stiff distractors. Its result tests precision allocation in that regime and is not a general nonconvex benchmark.

---

## 7. Design notes and trade-offs

- We keep metric awareness in the projection step (direction) rather than in a dedicated metric‑aware magnitude controller. This simplifies the API while retaining the key geometric property (orthogonality). See `docs/README_METRIC_AWARE_NOISE.md` for rationale.
- Precision scaling is diagonal by design for lower overhead and simpler composition. Full metric-curvature scaling is outside the current implementation.

---

## 8. FAQ

**Q: Will tangent noise ever increase energy?**
A: First-order contribution is zero by construction. Second-order effects can increase energy if noise is too large in stiff directions. The implementation limits this risk with precision scaling, bounded magnitude, and rejection with state restoration.

**Q: Why re‑project after precision re‑weighting?**
A: The re-weighting alters direction; re-projecting restores \(g^\top\delta=0\) under the selected Euclidean or metric geometry.

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
