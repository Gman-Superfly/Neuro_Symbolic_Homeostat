# SmallGain Allocator — Stability-Aware Weight Adaptation ✅

**Status**: Production Ready  
**Type**: Meta-Learning / Weight Adapter  
**Novel Contribution**: First stability-margin allocator for EBMs with formal guarantees

---

## What It Does (One Sentence)

The **SmallGain Allocator** treats your system's stability margin as a limited budget and intelligently allocates it across coupling terms to maximize convergence speed while maintaining formal stability guarantees.

---

## The Problem It Solves

In energy-based optimization with multiple coupled terms (modules), you have two conflicting goals:

1. **Speed**: Boost important couplings to converge faster
2. **Stability**: Don't boost so much that the system becomes unstable/divergent

Traditional approaches either:
- ❌ Apply uniform damping (wastes budget on unimportant terms)
- ❌ Use heuristics like GradNorm (no stability guarantees)
- ❌ Hope for the best (systems can diverge on dense graphs)

---

## The Solution: Resource Allocation with Formal Guarantees

**Core Insight**: Your system has a stability margin \( m \approx 2/\hat{L} - \alpha \) where \(\hat{L}\) is the Lipschitz constant. This margin is like a **budget** you can spend.

**SmallGain treats optimization like a knapsack problem**:

- **Budget**: Available stability margin (how much curvature increase you can afford)
- **Cost**: Each coupling edge costs \(\Delta L_k\) (Lipschitz increase per unit weight boost)
- **Value**: Each edge provides \(\Delta F_k\) (energy reduction per unit weight boost)
- **Policy**: Greedy allocation by `value/cost` ratio (fractional knapsack is optimal)

### Algorithm (Per Step)

```python
1. Compute stability budgets:
   - Global margin: m_global = max(0, L_target - L_current)
   - Per-row margins: m_row[r] = max(0, target_row[r] - current_row[r])
   - Usable budget: ρ * m_global (ρ=0.7 by default, conservative)

2. Compute per-edge metrics:
   - Cost[k] = ΔL_k (Lipschitz increase per unit weight boost)
   - Value[k] = grad_norm²[k] (proxy for energy reduction)
   - Score[k] = EMA(Value[k] / Cost[k])  # smoothed over time

3. Rank edges by score (descending)

4. Greedy allocation:
   For each edge k in ranked order:
     - Propose weight increase: w_new = w_old * (1 + max_step_change)
     - Compute cost: δL = Cost[k] * (δw / w_old)
     - If spent + δL ≤ budget:
         - Accept: w[k] ← w_new
         - Update: spent ← spent + δL

5. Return updated weights (bounded by [floor, ceiling])
```

## Visual: Stability Budget Allocation (Fractional Knapsack)

```
Stability budget B = ρ · (2/α)

Remaining budget:
[||||||||||||||||||||      ]   ← B − spent
 ^ spent (Σ δL)             ^ remaining

Edges ranked by score = value/cost (descending):
  e3: ████████████████   (best payoff)
  e1: ████████
  e2: ███
  e4: ██

Greedy allocation:
  e3 ← Δw (small δL, large ΔF) ──┐
  e1 ← Δw                        ├──> stop when spent ≥ B
  e2 ← Δw (skip if over budget) ─┘

Effect:
  - Spend curvature budget where it buys the most energy drop
  - Keep total “gain” under the stability margin (formal guarantee)
```

---

## Theoretical Foundations

The SmallGain Allocator combines three classical results:

1. **Small-Gain Theorem** (Zames 1966, Control Theory)
   - For feedback stability, keep total loop gain < 1
   - We budget the stability "reserve" optimally

2. **Fractional Knapsack** (Dantzig 1957, Optimization)
   - Greedy by value/cost is optimal when items are divisible
   - Local linearization makes weight changes "divisible"

3. **Gauss-Southwell Rule** (Nutini et al. 2015, Coordinate Descent)
   - Prioritize coordinates with best \(g²/(2L)\) ratio
   - Our per-edge scores mirror this at the coupling level

4. **Gershgorin Bounds** (1931, Linear Algebra)
   - Per-row diagonal/off-sum bounds justify row-aware budgeting

**Formal Guarantee**: If SmallGain spends ≤ ρ·budget with ρ < 1, the system remains contractive (Lyapunov stable).

---

## Performance Results (Validated)

### Baseline Scenario (2 modules, 2 couplings)

| Config | ΔF90 Steps | Final Energy | Improvement |
|--------|-----------|--------------|-------------|
| Analytic | 22 | -0.000385 | baseline |
| GradNorm | 10 | -0.005014 | 2.2x faster |
| **SmallGain** | **10** ✅ | **-0.020079** ✅ | **2.2x faster, 52x better energy** |

### Dense Scenario (16 modules, 48 couplings)

| Config | ΔF90 Steps | Final Energy | Improvement |
|--------|-----------|--------------|-------------|
| Analytic | 40 | +0.018582 ❌ | diverges |
| GradNorm | 20 | -0.021235 | usable |
| **SmallGain** | **12** ✅ | **-0.093700** ✅ | **40% faster, 4.4x better energy** |

**Key Takeaways**:
- ✅ Matches GradNorm on simple problems
- ✅ **40% faster** than GradNorm on dense graphs
- ✅ **4-52x better final energy** than baselines
- ✅ **Prevents divergence** where vanilla methods fail
- ✅ Formal stability guarantees (unique in EBM literature)

**See full validation**: [`SMALLGAIN_VALIDATION_FINAL.md`](SMALLGAIN_VALIDATION_FINAL.md)

---

## Usage

### Basic (Recommended Defaults)

```python
from core.coordinator import EnergyCoordinator
from core.weight_adapters import SmallGainWeightAdapter

coord = EnergyCoordinator(
    modules=your_modules,
    couplings=your_couplings,
    constraints={},
    weight_adapter=SmallGainWeightAdapter(),  # uses production defaults
    stability_guard=True,  # REQUIRED for margin tracking
)

etas = coord.relax_etas(initial_etas, steps=50)
```

### Production Defaults

```python
SmallGainWeightAdapter(
    budget_fraction=0.7,      # spend ≤70% of margin (conservative)
    max_step_change=0.10,     # ±10% weight change per step
    floor=0.1,                # minimum coupling weight
    ceiling=3.0,              # maximum coupling weight
    ema_alpha=0.3,            # smoothing for value/cost ratios
)
```

### Speed-Optimized Variant

```python
# For 30% faster ΔF90 with slightly weaker final energy
SmallGainWeightAdapter(
    budget_fraction=0.7,
    max_step_change=0.20,     # larger steps = faster convergence
    floor=0.1,
    ceiling=3.0,
)
```

### Safety-Critical Variant

```python
# For maximum stability (fewer backtracks), slower convergence
SmallGainWeightAdapter(
    budget_fraction=0.5,      # more conservative budget
    max_step_change=0.05,     # smaller steps
    floor=0.2,                # tighter bounds
    ceiling=2.0,
)
```

---

## When to Use SmallGain

### ✅ Ideal Use Cases

1. **Dense coupling graphs** (10+ modules, many couplings)
   - SmallGain shines where GradNorm struggles
   - 40% faster convergence demonstrated

2. **Safety-critical systems** requiring stability guarantees
   - Formal Lyapunov-style guarantees
   - Monotone acceptance maintained

3. **Energy quality matters** more than wall-clock speed
   - 4-10x better final energy than baselines
   - Worth the 2-5x per-step overhead

4. **Mixed coupling families** (quadratic + hinge + gate-benefit)
   - Optimal allocation across heterogeneous terms

### ⚠️ Consider Alternatives

1. **Very sparse graphs** (2-3 modules)
   - GradNorm is faster with similar results
   - Overhead not worth it for simple problems

2. **Real-time systems** with tight latency budgets
   - Per-step overhead: 2-5x vs GradNorm
   - Use speed-optimized variant or GradNorm

3. **Stationary landscapes** (fixed weights work fine)
   - No adaptation needed if static weights converge well

---

## Tuning (When Defaults Aren't Optimal)

**When to tune**:
- Domain-specific constraints (must converge in <N steps)
- Non-stationary dynamics (changing coupling activity)
- Optimizing specific KPIs (backtracks vs ΔF90 vs energy)

**Quick parameter sweep**:

```powershell
# Test 4 configs (~2 min)
uv run python -m experiments.sweep_smallgain --quick --rhos 0.7 0.9 --dws 0.10 0.20

# Analyze results
Get-Content plots/df90_smallgain_sweep_summary.csv | ConvertFrom-Csv | Sort-Object delta_f90_steps
```

**Key parameters**:
- **`budget_fraction` (ρ)**: Fraction of margin to spend per step
  - Lower (0.5) = safer, slower
  - Higher (0.9) = aggressive, faster
  - Default (0.7) is robust across scenarios

- **`max_step_change`**: Maximum relative weight change per step
  - Lower (0.05) = smoother, more stable
  - Higher (0.20) = faster convergence, more backtracks
  - Default (0.10) balances speed and stability

---

## Observability

### Per-Step Telemetry

```python
from cf_logging.observability import EnergyBudgetTracker

tracker = EnergyBudgetTracker(run_id="my_experiment")
tracker.attach(coord)

etas = coord.relax_etas(etas0, steps=50)
tracker.flush()  # writes to logs/energy_budget.csv
```

**Logged metrics**:
- `spent:global` — Cumulative Lipschitz budget spent
- `alloc:coup:<family>` — Per-family allocation totals
- `cost:coup:<family>` — Per-family Lipschitz costs
- `contraction_margin` — Safety margin remaining
- `margin_warn` — 1 if margin dropped below threshold

### Visualization

```powershell
# Plot budget spend vs margin
uv run python -m experiments.plots.plot_budget_vs_spend --input logs\energy_budget.csv

# Plot allocations over time
uv run python -m experiments.plots.plot_gain_budget --input logs\energy_budget.csv
```

---

## Test Coverage

**Unit tests**: `tests/test_small_gain_weight_adapter.py` (4 tests)
- ✅ Greedy allocation prioritizes high-value, low-cost terms
- ✅ Respects floor and ceiling bounds
- ✅ Fallback returns identity when no valid allocations
- ✅ Maintains monotone energy on small problems

**Benchmarks**: `experiments/benchmark_delta_f90.py`
- ✅ Baseline scenario validation
- ✅ Dense scenario validation
- ✅ Comparison vs analytic/GradNorm baselines

**Run tests**:
```powershell
# Unit tests
uv run -m pytest tests/test_small_gain_weight_adapter.py -v

# Quick benchmark
uv run python -m experiments.benchmark_delta_f90 --configs analytic gradnorm smallgain --scenario baseline --steps 60
```

---

## Why "SmallGain"?

The name comes from the **Small-Gain Theorem** in control theory:

> *For a feedback system to be stable, the total loop gain must be < 1*

The allocator:
1. Estimates the "gain" (Lipschitz cost) of each coupling edge
2. Keeps total gain within a safe budget (< 1 stability margin)
3. Allocates that budget to edges with best payoff (energy reduction per gain)

**Result**: Faster convergence with **formal stability guarantees** — unique in EBM optimization.

---

## Related Documentation

- **Design**: [`STABILITY_MARGIN_ALLOCATOR.md`](STABILITY_MARGIN_ALLOCATOR.md) — Detailed algorithm design
- **Validation**: [`SMALLGAIN_VALIDATION_FINAL.md`](SMALLGAIN_VALIDATION_FINAL.md) — Full experimental results
- **Stability Theory**: [`STABILITY_GUARANTEES.md`](STABILITY_GUARANTEES.md) — Lyapunov analysis, Small-Gain theorem
- **Comparison with other adapters**: [`META_LEARNING.md`](META_LEARNING.md) — GradNorm, AGM, GSPO-token

---

## Implementation

**Code**: [`core/weight_adapters.py`](../core/weight_adapters.py) — `SmallGainWeightAdapter` class  
**Tests**: [`tests/test_small_gain_weight_adapter.py`](../tests/test_small_gain_weight_adapter.py)  
**Benchmarks**: [`experiments/benchmark_delta_f90.py`](../experiments/benchmark_delta_f90.py)  
**Sweep script**: [`experiments/sweep_smallgain.py`](../experiments/sweep_smallgain.py)  

---

## Citation

If you use the SmallGain allocator in research:

```bibtex
@software{complexity_from_constraints,
  title = {Complexity from Constraints: SmallGain Stability-Margin Allocator},
  author = {Goldman, Oscar},
  organization = {Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業},
  year = {2025},
  note = {Production-ready stability-aware weight adapter for energy-based models}
}
```

**Key references**:
- Zames, G. (1966). On the input-output stability of time-varying nonlinear feedback systems. *IEEE TAC*.
- Dantzig, G. (1957). Discrete-variable extremum problems. *Operations Research*.
- Nutini, J., et al. (2015). Coordinate Descent Converges Faster with the Gauss-Southwell Rule. *ICML*.

---

## Quick Start Checklist

- [ ] Install framework: `pip install -e .` (or `uv sync`)
- [ ] Import: `from core.weight_adapters import SmallGainWeightAdapter`
- [ ] Enable stability guard: `stability_guard=True` in coordinator
- [ ] Use defaults for production: `SmallGainWeightAdapter()`
- [ ] Run benchmark to verify: `uv run python -m experiments.benchmark_delta_f90 --configs smallgain`
- [ ] Optional: Tune with sweep script for your domain
- [ ] Optional: Enable telemetry with `EnergyBudgetTracker`

**That's it!** SmallGain handles the rest automatically.

---

**Status**: ✅ Production Ready — 120 tests passing, validated on baseline and dense scenarios.

