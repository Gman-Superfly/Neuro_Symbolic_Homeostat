# SmallGain allocator: stability-aware weight adaptation

**Status**: validated on synthetic scenarios in this repository
**Type**: Meta-Learning / Weight Adapter
**Novel contribution**: stability-margin allocation integrated with this repository's coordinator and observability stack

---

## What it does (one sentence)

The **SmallGain Allocator** treats the stability margin as a budget and allocates it across coupling terms to improve convergence while preserving conservative stability bounds in supported regimes.

---

## The problem it solves

In energy-based optimization with multiple coupled terms (modules), you have two conflicting goals:

1. **Speed**: Boost important couplings to converge faster
2. **Stability**: Don't boost so much that the system becomes unstable/divergent

Traditional approaches either:
- Apply uniform damping (wastes budget on unimportant terms)
- Use heuristics like GradNorm (no stability guarantees)
- Hope for the best (systems can diverge on dense graphs)

---

## The solution: resource allocation with conservative bounds

**Core Insight**: Your system has a stability margin \( m \approx 2/\hat{L} - \alpha \) where \(\hat{L}\) is the Lipschitz constant. This margin is like a **budget** you can spend.

**SmallGain treats optimization like a knapsack problem**:

- **Budget**: Available stability margin (how much curvature increase you can afford)
- **Cost**: Each coupling edge costs \(\Delta L_k\) (Lipschitz increase per unit weight boost)
- **Value**: Each edge provides \(\Delta F_k\) (energy reduction per unit weight boost)
- **Policy**: Greedy allocation by `value/cost` ratio (fractional knapsack is optimal)

### Algorithm (per step)

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

## Visual: stability budget allocation (fractional knapsack)

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
  - Keep total “gain” under the stability margin (conservative bound)
```

---

## Theoretical foundations

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

**Bounded-case guarantee**: If SmallGain spends ≤ ρ·budget with ρ < 1, linearized/SPD regimes remain contractive under the stated assumptions.

---

## Performance results (validated)

### Baseline scenario (2 modules, 2 couplings)

| Config | ΔF90 Steps | Final Energy | Improvement |
|--------|-----------|--------------|-------------|
| Analytic | 22 | -0.000385 | baseline |
| GradNorm | 10 | -0.005014 | 2.2x faster |
| **SmallGain** | **10**  | **-0.020079**  | **2.2x faster, 52x better energy** |

### Dense scenario (16 modules, 48 couplings)

| Config | ΔF90 Steps | Final Energy | Improvement |
|--------|-----------|--------------|-------------|
| Analytic | 40 | +0.018582  | diverges |
| GradNorm | 20 | -0.021235 | usable |
| **SmallGain** | **12**  | **-0.093700**  | **40% faster, 4.4x better energy** |

**Key Takeaways**:
- Matches GradNorm on simple problems
- **40% faster** than GradNorm on dense graphs
- **4-52x better final energy** than baselines
- **Prevents divergence** where vanilla methods fail
- Conservative stability controls, plus monotone acceptance checks

**See full validation**: [`SMALLGAIN_VALIDATION_FINAL.md`](SMALLGAIN_VALIDATION_FINAL.md)

---

## Usage

### Basic (recommended defaults)

```python
from core.coordinator import EnergyCoordinator
from core.weight_adapters import SmallGainWeightAdapter

coord = EnergyCoordinator(
    modules=your_modules,
    couplings=your_couplings,
    constraints={},
    weight_adapter=SmallGainWeightAdapter(),  # uses default settings
    stability_guard=True,  # REQUIRED for margin tracking
)

etas = coord.relax_etas(initial_etas, steps=50)
```

### Default settings

```python
SmallGainWeightAdapter(
    budget_fraction=0.7,      # spend ≤70% of margin (conservative)
    max_step_change=0.10,     # ±10% weight change per step
    floor=0.1,                # minimum coupling weight
    ceiling=3.0,              # maximum coupling weight
    ema_alpha=0.3,            # smoothing for value/cost ratios
)
```

### Speed-optimized variant

```python
# For 30% faster ΔF90 with slightly weaker final energy
SmallGainWeightAdapter(
    budget_fraction=0.7,
    max_step_change=0.20,     # larger steps = faster convergence
    floor=0.1,
    ceiling=3.0,
)
```

### Conservative variant

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

## When to use SmallGain

### Ideal use cases

1. **Dense coupling graphs** (10+ modules, many couplings)
   - SmallGain shines where GradNorm struggles
   - 40% faster convergence demonstrated

2. **Systems where conservative stability guards are required**
   - Linearized/SPD guarantees with explicit assumptions
   - Monotone acceptance maintained

3. **Energy quality matters** more than wall-clock speed
   - 4-10x better final energy than baselines
   - Worth the 2-5x per-step overhead

4. **Mixed coupling families** (quadratic + hinge + gate-benefit)
   - Optimal allocation across heterogeneous terms

### Consider alternatives

1. **Sparse graphs** (2-3 modules)
   - GradNorm is faster with similar results
   - Overhead not worth it for simple problems

2. **Real-time systems** with tight latency budgets
   - Per-step overhead: 2-5x vs GradNorm
   - Use speed-optimized variant or GradNorm

3. **Stationary landscapes** (fixed weights work fine)
   - No adaptation needed if static weights converge well

---

## Tuning (when defaults are not optimal)

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

### Per-step telemetry

```python
from cf_logging.observability import EnergyBudgetTracker

tracker = EnergyBudgetTracker(run_id="my_experiment")
tracker.attach(coord)

etas = coord.relax_etas(etas0, steps=50)
tracker.flush()  # writes to logs/energy_budget.csv
```

**Logged metrics**:
- `spent:global`: Cumulative Lipschitz budget spent
- `alloc:coup:<family>`: Per-family allocation totals
- `cost:coup:<family>`: Per-family Lipschitz costs
- `contraction_margin`: Safety margin remaining
- `margin_warn`: 1 if margin dropped below threshold

### Visualization

```powershell
# Plot budget spend vs margin
uv run python -m experiments.plots.plot_budget_vs_spend --input logs\energy_budget.csv

# Plot allocations over time
uv run python -m experiments.plots.plot_gain_budget --input logs\energy_budget.csv
```

---

## Test coverage

**Unit tests**: `tests/test_small_gain_weight_adapter.py` (4 tests)
- Greedy allocation prioritizes high-value, low-cost terms
- Respects floor and ceiling bounds
- Fallback returns identity when no valid allocations
- Maintains monotone energy on small problems

**Benchmarks**: `experiments/benchmark_delta_f90.py`
- Baseline scenario validation
- Dense scenario validation
- Comparison vs analytic/GradNorm baselines

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

**Result**: Faster convergence in the documented dense scenarios, with conservative stability controls.

---

## Related documentation

- **Design**: [`STABILITY_MARGIN_ALLOCATOR.md`](STABILITY_MARGIN_ALLOCATOR.md), detailed algorithm design
- **Validation**: [`SMALLGAIN_VALIDATION_FINAL.md`](SMALLGAIN_VALIDATION_FINAL.md), full experimental results
- **Stability theory**: [`STABILITY_GUARANTEES.md`](STABILITY_GUARANTEES.md), Lyapunov analysis and Small-Gain theorem
- **Comparison with other adapters**: [`META_LEARNING.md`](META_LEARNING.md), GradNorm, AGM, GSPO-token

---

## Implementation

**Code**: [`core/weight_adapters.py`](../core/weight_adapters.py), `SmallGainWeightAdapter` class
**Tests**: [`tests/test_small_gain_weight_adapter.py`](../tests/test_small_gain_weight_adapter.py)
**Benchmarks**: [`experiments/benchmark_delta_f90.py`](../experiments/benchmark_delta_f90.py)
**Sweep script**: [`experiments/sweep_smallgain.py`](../experiments/sweep_smallgain.py)

---

## Citation

If you use this repository in your research, please cite it. This is ongoing work; we would like to know your opinions and experiments. Thank you.

**Authors:** Oscar Goldman, Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業).

**Reference (author-year format):** Goldman, O. (2025). *Complexity from Constraints: SmallGain stability-margin allocator*. Software repository. Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業). Stability-aware weight adaptation for energy-based models in this repository.

**Key references**:
- Zames, G. (1966). On the input-output stability of time-varying nonlinear feedback systems. *IEEE TAC*.
- Dantzig, G. (1957). Discrete-variable extremum problems. *Operations Research*.
- Nutini, J., et al. (2015). Coordinate Descent Converges Faster with the Gauss-Southwell Rule. *ICML*.

---

## Quick start checklist

- [ ] Install framework: `pip install -e .` (or `uv sync`)
- [ ] Import: `from core.weight_adapters import SmallGainWeightAdapter`
- [ ] Enable stability guard: `stability_guard=True` in coordinator
- [ ] Use defaults: `SmallGainWeightAdapter()`
- [ ] Run benchmark to verify: `uv run python -m experiments.benchmark_delta_f90 --configs smallgain`
- [ ] Optional: Tune with sweep script for your domain
- [ ] Optional: Enable telemetry with `EnergyBudgetTracker`

After setup, SmallGain applies allocation updates automatically.

---

**Status**: validated on repository scenarios (`baseline`, `dense`) and covered by local tests listed above.

