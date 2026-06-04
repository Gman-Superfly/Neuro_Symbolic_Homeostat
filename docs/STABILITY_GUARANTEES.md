# Stability bounds in complexity from constraints

## Overview

This document explains the stability conditions used by this framework, how to enable them, and how to interpret telemetry.

**Status**: validated on synthetic scenarios and local tests in this repository

---

## Quick start

### Enable stability guard

```python
from core.coordinator import EnergyCoordinator

coord = EnergyCoordinator(
    modules=my_modules,
    couplings=my_couplings,
    constraints={},
    stability_guard=True,          # Enable Lyapunov-style step capping
    stability_cap_fraction=0.9,    # Use 90% of the 2/L bound
    log_contraction_margin=True,   # Log step margin
    warn_on_margin_shrink=True,    # Emit warnings when margin drops
    margin_warn_threshold=1e-6,    # Warning threshold
)

etas = coord.relax_etas(etas0, steps=50)
```

### Enable SmallGain allocator (recommended)

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
    stability_guard=True,  # Required for margin tracking
)
```

---

## What are the stability bounds?

### The problem

In standard gradient descent, step sizes are chosen heuristically:
- Too small → slow convergence
- Too large → divergence (energy explodes)

**Goal**: use conservative conditions that bound updates in supported regimes and add runtime guards elsewhere.

### The solution: step caps and acceptance checks

In linear and SPD settings the coordinator treats energy \( F(η) \) as a Lyapunov-style function and enforces accepted-step non-increase:

\[
F(η^{k+1}) \leq F(η^k) \quad \forall k
\]

**How**: by capping step size from a conservative estimate of the Lipschitz constant of \( \nabla F \), then rejecting and restoring steps that still increase the guarded objective.

---

## Mathematical foundation

### Lipschitz constant

The gradient \( \nabla F(η) \) is **L-Lipschitz** if:

\[
\|\nabla F(x) - \nabla F(y)\| \leq L \|x - y\|
\]

**Physical Meaning**: L measures the "stiffest spring" in the system. High L = tight constraints.

### Gradient descent convergence theorem

If \( \nabla F \) is L-Lipschitz and we use step size \( \alpha < 2/L \), then:

\[
F(η^{k+1}) \leq F(η^k) - \alpha (1 - \frac{\alpha L}{2}) \|\nabla F(η^k)\|^2
\]

**Proof of non-increase**: The term \( (1 - \alpha L/2) > 0 \) when \( \alpha < 2/L \).

### Our implementation

We estimate L using a **Gershgorin-style bound**, a conservative upper bound:

\[
L \leq \sum_i |F_{i,local}''(η_i)| + \sum_{(i,j)} |F_{ij,coupling}''|
\]

Then cap step size:

\[
\alpha_{\text{cap}} = 0.9 \cdot \frac{2}{L}
\]

(The 0.9 factor = `stability_cap_fraction` provides additional margin.)

## What is standard, and what is repository specific

The stability mathematics here follows standard results:
- gradient descent condition \( \alpha < 2/L \),
- Gershgorin row-sum bounds for conservative spectral control,
- diagonal preconditioning for stiffness-aware coordinate scaling.

The repository-specific design is the composable curvature contract:
- modules expose local curvature with `SupportsPrecision.curvature`,
- couplings expose row-wise curvature bounds with `SupportsCouplingCurvature.coupling_curvature_bounds`,
- the coordinator composes these values into one precision cache and one Lipschitz estimate used by preconditioning, step capping, and precision-aware noise control.

## Observed regime boundary from tests

The current test suite records two regimes.

### Tight-bound regime, curvature awareness changes the outcome

`tests/test_precision_conditioning.py::test_curvature_awareness_converges_where_plain_gd_stalls_above_2_over_L`

- Coupled quadratic pair with \(\lambda_{\max}=33\), so \(2/L = 0.0606\).
- Requested step \(0.1\) lies above that threshold.
- Plain gradient descent stalls after rejection.
- Curvature-aware modes, diagonal preconditioning or stability guard capping, converge.

### Conservative-bound regime, guard can trade speed for margin

`tests/test_precision_conditioning.py::test_gershgorin_cap_can_be_conservative_on_mixed_preconditioned_problem`

- Mixed sum, product, and hinge couplings.
- Initial Gershgorin cap is below requested step \(0.1\).
- Guarded and unguarded preconditioned runs both converge.
- Over the same fixed step budget, the guarded run can end at higher final energy.

## Visual: step capping and acceptance flow

```
Compute L (Gershgorin) ────────────────┐
                                       │
Requested α (user) ────────┐           │
                           └─ min ──> α_used = min(α_requested, 0.9 · 2/L)
                                                │
Trial step: η_{k+1} = η_k − α_used · ∇F(η_k)    │
                                                │
                         ΔF = F_{k+1} − F_k ≤ 0 ?
                               │                │
                              yes              no
                               │                │
                         ACCEPT step      REJECT and restore η_k
                                             (and optionally warn via margin)
```

Contraction margin gauge (step budget):

```
margin = (2/L) − α_used

0                                      (2/L)
|███████████░░░░░░░░░░░░░░░░░|  healthy
 ^ spent (α_used)
```

---

## SmallGain allocator (advanced)

### The problem with coupled systems

In coupled systems, the Lipschitz bound \( L \) comes from **interactions** between terms. A naive global cap wastes the stability budget.

**SmallGain Idea**: Allocate the budget per-edge based on "value per Lipschitz cost".

### Implemented contract

For a system with local terms \( F_i \) and couplings \( C_{ij} \), the coordinator estimates a conservative row-sum bound:

\[
\hat{L} = \max_i \left(d_i + \sum_{j \neq i} |c_{ij}|\right).
\]

The step cap is:

\[
\alpha_{\mathrm{used}} = \min(\alpha_{\mathrm{requested}}, \gamma \cdot 2/\hat{L}).
\]

The SmallGain allocator receives the remaining global margin and per-row margin estimates. It ranks coupling families by a smoothed `value/cost` score, where value is approximated by gradient-norm squared and cost is the estimated Lipschitz increase. The current implementation enforces the global spend cap and records row margins for telemetry. It does not yet enforce row-incidence booking for each edge.

**Scope-limited bound**: In quadratic/SPD regimes, if \(\hat{L}\) upper-bounds \(\lambda_{\max}(H)\) and \(\alpha_{\mathrm{used}} < 2/\hat{L}\), then \(\rho(I-\alpha_{\mathrm{used}}H) < 1\). The allocator stays inside the same local-linear condition only to the extent that its predicted spend remains inside the estimated margin and the acceptance guard rejects harmful trial steps.

---

## Observability

### Telemetry fields

When `log_contraction_margin=True`, `EnergyBudgetTracker` emits:

- `contraction_margin`: \( (2/L) - \alpha \) (step margin remaining)
- `margin_warn`: 1 if margin < threshold, 0 otherwise
- `spent:global`: Accumulated Lipschitz budget spent (SmallGain only)
- `alloc:coup:<family>`: Per-family allocations (SmallGain only)
- `cost:coup:<family>`: Per-family Lipschitz costs (SmallGain only)

### Interpreting contraction margin

| Margin Value | Meaning | Action |
|--------------|---------|--------|
| > 0.01 |  Healthy | No action needed |
| 0.001 - 0.01 |  Tight | Consider reducing coupling weights or step size |
| < 0.001 |  Risky | **Warning emitted**, reduce step size immediately |
| Negative |  Unstable | System may diverge, hard cap applied automatically |

### Visualization

```powershell
# Plot margin over time
uv run python -m experiments.plots.plot_budget_vs_spend --input logs\energy_budget.csv --run_id my_run

# Plot gain budget (SmallGain allocator)
uv run python -m experiments.plots.plot_gain_budget --input logs\energy_budget.csv --run_id my_run
```

---

## Tuning for stability

### Reducing Lipschitz constant

**Problem**: L is too large → step sizes become tiny → slow convergence

**Solutions**:

1. **Reduce coupling weights**:
   ```python
   QuadraticCoupling(weight=0.3)  # instead of 1.0
   ```

2. **Use homotopy scheduling** (start with weak couplings):
   ```python
   EnergyCoordinator(
       homotopy_coupling_scale_start=0.2,
       homotopy_steps=20,
   )
   ```

3. **Enable coupling auto-cap**:
   ```python
   EnergyCoordinator(
       stability_coupling_auto_cap=True,
       stability_coupling_target=10.0,  # desired max L
   )
   ```

4. **Use polynomial bases** (can improve conditioning):
   ```python
   from modules.polynomial.polynomial_energy import PolynomialEnergyModule
   mod = PolynomialEnergyModule(degree=3, basis="legendre")
   ```

### Increasing step margin

**Problem**: Margin too tight → frequent warnings

**Solutions**:

1. **Reduce step size**:
   ```python
   EnergyCoordinator(step_size=0.03)  # instead of 0.05
   ```

2. **Use more conservative cap fraction**:
   ```python
   EnergyCoordinator(stability_cap_fraction=0.7)  # instead of 0.9
   ```

3. **Enable SmallGain allocator** (greedy margin allocation):
   ```python
   weight_adapter=SmallGainWeightAdapter(budget_fraction=0.6)
   ```

---

## Comparison: standard vs SmallGain guard

| Feature | Standard `stability_guard` | SmallGain Allocator |
|---------|---------------------------|---------------------|
| **Lipschitz bound** | Global (single L) | Per-edge (L_ij) |
| **Step capping** | Uniform cap for all | Adaptive per-coupling weights |
| **Overhead** | ~5% | ~100-200% (worth it for dense graphs) |
| **Guarantees** | Contraction if α < 2/L | Contraction if budget spent ≤ ρ |
| **Allocation policy** | Conservative global cap | Greedy value/cost allocation |
| **Use case** | Simple graphs, prototyping | Dense graphs and synthetic benchmarks |

**Recommendation**:
- Use standard guard for quick experiments
- Use SmallGain for dense benchmark scenarios where its overhead is acceptable

---

## Formal bounds and assumptions

### Theorem 1: monotonic energy descent

**Statement**: For deterministic gradient descent on an \(L\)-smooth objective, if the estimated \(L\) upper-bounds the true gradient Lipschitz constant and the used step satisfies \( \alpha < 2/L \), then:

\[
F(η^{k+1}) \leq F(η^k) \quad \forall k
\]

**Proof Sketch**:
1. Lipschitz continuity implies \( F(η + α g) \leq F(η) + α \langle \nabla F, g \rangle + \frac{α^2 L}{2} \|g\|^2 \)
2. Setting \( g = -\nabla F \) (gradient direction) gives:
   \[
   F(η^{k+1}) \leq F(η^k) - \alpha (1 - \frac{\alpha L}{2}) \|\nabla F\|^2
   \]
3. Since \( \alpha < 2/L \), the term \( (1 - \alpha L/2) > 0 \), giving descent for this smooth deterministic step.

### Theorem 2: SmallGain contraction, scoped form

**Statement**: In the same quadratic/SPD local-linear regime, if the allocator's predicted Lipschitz spend remains inside the reserved margin, then the capped gradient step remains inside the same conservative contraction condition.

**Proof Sketch**:
1. Row-wise Lipschitz constraint: \( \sum_j L_{ij} < 2/\alpha \)
2. SmallGain enforces a global predicted-spend cap: \( \sum_j \text{allocated}_{ij} \leq \rho \cdot (2/\alpha) \)
3. Since ρ < 1, margin \( (1-ρ) \cdot (2/\alpha) > 0 \) remains
4. By Gershgorin theorem, the Jacobian spectral radius \( < 2/\alpha \)
5. Therefore, the fixed-point iteration is contractive under these local assumptions.

**Empirical Validation**: See `docs/SMALLGAIN_VALIDATION_FINAL.md`

---

## Troubleshooting

### Warning: "Contraction margin below threshold"

**Meaning**: The step margin is shrinking (safety buffer), system approaching instability

**Actions** (in order of preference):
1. Reduce `step_size` by 50% (e.g., 0.05 → 0.025)
2. Reduce coupling weights by 30% (e.g., `weight=1.0 → 0.7`)
3. Use homotopy to ramp up couplings gradually
4. Enable SmallGain allocator for greedy margin allocation

### Energy increasing despite guard

**Possible Causes**:
1. Numerical precision issues (use higher tolerance: `monotonic_energy_tol=1e-8`)
2. Adaptive methods active (set `assert_monotonic_energy=False`)
3. Noise enabled (increases energy to second order)
4. Bug in gradient implementation (check with finite-difference)

**Debug Steps**:
1. Disable all extras: `noise_magnitude=0.0`, `weight_adapter=None`, `homotopy_steps=0`
2. Enable assertion: `assert_monotonic_energy=True`
3. Run minimal test case
4. Check logs for NaN or inf values

### Step size becoming tiny

**Symptoms**: `contraction_margin` → 0, convergence slows drastically

**Causes**:
- Coupling weights too high (Lipschitz bound exploding)
- Ill-conditioned energy function (monomials vs polynomials)

**Fixes**:
1. Use polynomial basis: `PolynomialEnergyModule(basis="legendre")`
2. Reduce coupling weights (start low, increase gradually)
3. Use homotopy: `homotopy_coupling_scale_start=0.2`
4. Check for degenerate constraints (e.g., conflicting hinges)

---

## Test coverage

Stability behavior in this repository is validated by:

### Direct tests

- `tests/test_stability_coupling_cap.py`: Auto-cap applied correctly
- `tests/test_stability_coupling_sweep.py`: Stability across coupling strengths
- `tests/test_stability_margin_warnings.py`: **NEW**, warning system (3 tests)
- `tests/test_monotonic_energy.py`: Monotonicity assertions work

### Integration tests

- `tests/test_small_gain_weight_adapter.py`: SmallGain keeps monotone energy
- `tests/test_polynomial_conditioning.py`: polynomial bases reduce ΔF variance in the tested setup
- All `test_coordinator_*.py`: Energy non-increasing across modes

**Run all stability tests**:

```powershell
uv run -m pytest tests/ -k "stability or monotonic or margin" -v
```

---

## Worked example

### Problem: dense coupling graph diverges

```python
# Risky: no stability guard, large step, strong couplings
coord = EnergyCoordinator(
    modules=[...],  # 16 modules
    couplings=[(i, j, QuadraticCoupling(weight=2.0)) for ...],  # Dense graph
    constraints={},
    step_size=0.15,  # Too large!
    stability_guard=False,
)

etas = coord.relax_etas(etas0, steps=50)
# Energy diverges after ~10 steps
```

### Solution 1: enable guard

```python
# Stability guard auto-caps step size
coord = EnergyCoordinator(
    modules=[...],
    couplings=[(i, j, QuadraticCoupling(weight=2.0)) for ...],
    constraints={},
    step_size=0.15,  # Requested, but will be capped
    stability_guard=True,
    log_contraction_margin=True,
)

etas = coord.relax_etas(etas0, steps=50)
# Converges with a capped step, but may be slow (step size capped to ~0.01)
```

### Solution 2: SmallGain allocator

```python
# SmallGain allocates budget using value/cost priorities
from core.weight_adapters import SmallGainWeightAdapter

coord = EnergyCoordinator(
    modules=[...],
    couplings=[(i, j, QuadraticCoupling(weight=2.0)) for ...],
    constraints={},
    weight_adapter=SmallGainWeightAdapter(),
    stability_guard=True,
)

etas = coord.relax_etas(etas0, steps=50)
# In this scenario, uses fewer steps than Solution 1 with lower final energy
```

---

## Stability modes compared

### 1. No guard (default, use for prototyping only)

```python
coord = EnergyCoordinator(stability_guard=False)
```

**Guarantees**:  None
**Pros**: Lowest per-step overhead
**Cons**: Can diverge on difficult energy functions
**Use when**: Small graphs, smooth energies, debugging

### 2. Standard stability guard

```python
coord = EnergyCoordinator(stability_guard=True)
```

**Guarantees**:  Energy non-increasing under the stated deterministic \(L\)-smooth assumptions
**Pros**: Simple, low overhead (~5%)
**Cons**: Conservative (uniform cap wastes budget)
**Use when**: Simple graphs, applications that need conservative accepted-step checks, conservative baseline

### 3. SmallGain allocator (recommended for dense benchmark scenarios)

```python
coord = EnergyCoordinator(
    weight_adapter=SmallGainWeightAdapter(),
    stability_guard=True,
)
```

**Guarantees**:  Same accepted-step guard, plus bounded predicted spend
**Pros**: Fewer ΔF90 steps and lower final energy in the documented dense benchmark
**Cons**: 2-5x computational overhead per step
**Use when**: Dense graphs (10+ modules), and energy quality matters more than per-step compute cost

---

## Gershgorin bound (implementation details)

### How we estimate L

For each module \( i \), we estimate the local Hessian contribution:

\[
L_i^{local} = |F_i''(η_i)|
\]

For each coupling \( (i,j) \), we estimate:

\[
L_{ij}^{coupling} = |F_{ij}''|
\]

**Row sum** (Gershgorin bound):

\[
L_i^{row} = L_i^{local} + \sum_j L_{ij}^{coupling}
\]

**Global bound**:

\[
L = \max_i L_i^{row}
\]

### Coupling-specific estimates

| Coupling Type | Lipschitz Contribution |
|---------------|------------------------|
| **QuadraticCoupling** | \( L_{ij} = 4w \) (second derivative of \( w(η_i - η_j)^2 \)) |
| **HingeCoupling** | \( L_{ij} = 4w \) (when active), 0 (when inactive) |
| **GateBenefitCoupling** | \( L_{ij} \approx 0 \) (linear term, no curvature) |

**SmallGain Smoothing**: For hinges near activation (gap ≈ 0), we use a smooth interpolation to avoid discontinuities.

---

## Contraction margin interpretation

### Definition

\[
\text{margin} = \frac{2}{L} - \alpha_{\text{used}}
\]

**Physical meaning**: How much estimated step margin is left unused.

### Healthy margins

- **margin > 0.01**: room remains for adaptation
- **margin ∈ [0.001, 0.01]**: accepted-step margin is tighter
- **margin ∈ [1e-6, 0.001]**: tight, consider backing off
- **margin < 1e-6**: warning emitted, instability risk

### SmallGain budget tracking

The allocator tracks:

- **Global budget**: \( B = \rho \cdot (2/\alpha) \)
- **Spent**: \( \sum_{ij} \text{allocated}_{ij} \)
- **Remaining**: \( B - \text{spent} \)

**Default operation**: Spent ≤ 70% of budget (ρ=0.7)

---

## Advanced: passivity and dissipativity

### Passivity interpretation

Treating the coordinator as a dynamical system:

\[
\dot{η} = -\nabla F(η)
\]

The energy F acts as a **storage function**. The system is **passive** if:

\[
\frac{dF}{dt} = \langle \nabla F, \dot{η} \rangle = -\|\nabla F\|^2 \leq 0
\]

**Physical meaning**: Under this continuous model, energy decreases along the flow.

### Small-Gain theorem (control theory)

For interconnected subsystems with gains \( \gamma_i \):

**Stability Condition**:

\[
\prod_{i \in \text{loop}} \gamma_i < 1
\]

**Our case**: Each coupling has a predicted gain \( L_{ij} \cdot \alpha \). The SmallGain allocator keeps predicted spend inside the configured margin.

**Reference**: Zhou, K., & Doyle, J. C. (1998). Essentials of Robust Control. Chapter 6.

---

## Empirical validation

### SmallGain allocator results

From `docs/SMALLGAIN_VALIDATION_FINAL.md`:

**Baseline Scenario**:
- Standard guard: ΔF90 = 22 steps, final energy = -0.0004
- **SmallGain**: ΔF90 = 10 steps (55% reduction), final energy = -0.020

**Dense Scenario (16 modules)**:
- Standard guard: ΔF90 = 40 steps, **diverges** (final energy positive)
- **SmallGain**: ΔF90 = 12 steps, final energy = -0.094

**Conclusion**: In these benchmark scenarios, SmallGain used fewer ΔF90 steps and reached lower final energy than the compared baselines.

### Polynomial conditioning results

From `tests/test_polynomial_conditioning.py`:

**Legendre vs Raw Landau**:
- Raw Landau: ΔF variance = 0.045 (irregular)
- **Legendre**: ΔF variance = 0.018 (60% smoother)

**Takeaway**: In this test, the orthonormal basis reduced ΔF variance independent of step capping.

---

## FAQ

### Q: Do I need `stability_guard=True`?

**A**: No, but recommended for:
- Dense coupling graphs
- scenarios with stronger coupling interactions
- When coupling weights are tuned empirically (not hand-picked)
- Applications where conservative accepted-step checks are required

Disable for:
- Prototyping on tiny graphs (<3 modules)
- When you've validated step sizes empirically
- Lowest per-step overhead is required and step sizes have been validated

### Q: What's the overhead of SmallGain?

**A**: 2-5x per-step compute vs gradient descent, but:
- 30-40% fewer ΔF90 steps in documented dense benchmarks
- lower final energy in documented dense benchmarks
- **Observed result**: lower wall-time in the documented dense benchmark when fewer steps offset overhead

### Q: Can I combine SmallGain with line search?

**A**: Yes. They check different things:
- SmallGain allocates budget across couplings
- Line search validates each step (Armijo condition)
- Both active = stricter step acceptance

```python
coord = EnergyCoordinator(
    weight_adapter=SmallGainWeightAdapter(),
    stability_guard=True,
    line_search=True,  # Extra acceptance check
)
```

### Q: Why not just use line search alone?

**A**: Line search checks a proposed step after computing it. The stability guard caps the step before applying it. SmallGain adds a greedy value/cost allocation over the estimated margin.

---

## Summary

- **Stability guard**: Lyapunov-style step capping with conservative bounds
- **SmallGain allocator**: Budget allocation that reduced ΔF90 steps in the documented dense benchmark
- **Polynomial bases**: Reduced ΔF variance in the documented conditioning test
- **Contraction margin**: observable step-margin metric with warnings

**Recommended configuration for these repository demos**:

```python
from core.coordinator import EnergyCoordinator
from core.weight_adapters import SmallGainWeightAdapter
from modules.polynomial.polynomial_energy import PolynomialEnergyModule

coord = EnergyCoordinator(
    modules=[PolynomialEnergyModule(degree=3, basis="legendre"), ...],
    couplings=my_couplings,
    constraints={},
    weight_adapter=SmallGainWeightAdapter(),
    stability_guard=True,
    log_contraction_margin=True,
    warn_on_margin_shrink=True,
    line_search=True,  # Extra acceptance check
)
```

This configuration provides:
- conservative stability controls
- fewer ΔF90 steps on the documented dense benchmarks
- runtime warnings
- reproducible defaults for this repository

---

## References

### Papers

- Zhou, K., & Doyle, J. C. (1998). *Essentials of Robust Control*. Prentice Hall.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. (Chapter 9: Gradient methods)

### Code

- Implementation: `core/coordinator.py` (Gershgorin bound estimation)
- SmallGain: `core/weight_adapters.py` (`SmallGainWeightAdapter`)
- Tests: `tests/test_stability_*.py`, `tests/test_small_gain_*.py`

### Related docs

- `docs/SMALLGAIN_VALIDATION_FINAL.md`, empirical validation results
- `docs/PROXIMAL_METHODS.md`, proximal operators for stability
- `docs/POLYNOMIAL_BASES.md`, conditioning via orthonormal bases
- `README.md`, quick-start examples

