# Counterfactual gate-benefit coupling (CGBC), wormhole nickname

**Status**: validated core mechanism in repository demos
**Type**: coupling mechanism for non-local gate updates
**Scope**: addresses zero-gradient deadlock when a caller provides a downstream benefit estimate
**Nickname**: wormhole coupling

---

## What it is (one sentence)

**Counterfactual gate-benefit coupling (CGBC)** applies a gate gradient proportional to a caller-supplied estimate of downstream benefit. The implementation and older docs use **wormhole coupling** as a nickname for CGBC.

---

## The problem: zero-gradient deadlock

Imagine you have a module that's currently **completely inactive** (η = 0, like a "closed door"):

```
Standard Energy-Based Model:
  Module State: η_gate = 0  (OFF)
  ↓
  No energy flows through closed connection
  ↓
  No gradient force can act on the gate
  ↓
  Gate stays closed forever (LOCAL MINIMUM TRAP)
```

The basic question: how does a closed gate receive a learning signal when its current path carries no signal?

In standard physics or neural networks:
- If a connection is closed (weight = 0 or gate = 0), **no information flows**
- If no information flows, **no gradient exists** to tell the system "opening this would help"
- The system is **stuck in a local minimum** with no way out except random noise

---

## The solution: counterfactual gate gradient

We use a coupling type called `GateBenefitCoupling` that implements CGBC. It creates a non-local gate gradient from a benefit estimate that the caller provides through `constraints[delta_key]`. The mechanism does not discover the benefit by itself. The useful abstraction is that the benefit estimate becomes a typed energy term that participates in the same relaxation, stability guards, and telemetry as the other couplings.

### The mathematical trick

**Energy function**:
```
F = -w * η_gate * Δη_domain
```

Where:
- `η_gate` ∈ [0,1] is the gate activation (0 = closed, 1 = open)
- `Δη_domain` is the caller-supplied benefit estimate if the gate were to open
- `w` is the coupling weight

**Gradient with respect to gate**:
```python
dF/dη_gate = -w * Δη_domain  # NO η_gate IN THE GRADIENT!
```

Key property: the gradient does not depend on `η_gate`.

- Even when η_gate = 0 (completely closed)
- The gate receives a force proportional to the supplied `Δη_domain`
- This force acts even though the connection is closed, which motivates the wormhole nickname

---

## Where does Delta_benefit come from?

CGBC requires a benefit estimate from outside the coupling. That estimate can come from several realistic sources:

- A learned benefit estimator trained to predict downstream improvement.
- A downstream loss differential, for example `loss_before - loss_after` from a cheap local rollout.
- A planning rollout that estimates the value of opening a branch.
- External supervision or a hand-written domain heuristic.

The demo uses a hard-coded positive value to isolate the gradient property. That demo shows that a supplied positive benefit creates a descent force on a closed gate. It does not show internal benefit estimation.

---

## Visual analogy

### Standard energy function view (no CGBC)
```
Energy
  ↑
  |     ╱╲              ╱╲
  |    ╱  ╲            ╱  ╲
  |   ╱    ╲__________╱    ╲  ← stuck in left well
  |  ╱      ^                    (no gradient to escape)
  | ╱       └─ η_gate=0
  └──────────────────────────→ η_gate
```

### With CGBC (wormhole nickname)
```
Energy
  ↑
  |     ╱╲    ~~~~~~~~>  ╱╲
  |    ╱  ╲    CGBC     ╱  ╲
  |   ╱    ╲__________╱    ╲  ← gradient signal acts
  |  ╱      ^          ^        from right well to left
  | ╱       └──────────┘
  └──────────────────────────→ η_gate
         η=0         η=1

A caller-supplied benefit estimate creates a force
on η_gate=0 even though no active connection exists.
```

---

## Code implementation

### GateBenefitCoupling

Located in `core/couplings.py`:

```python
@dataclass(frozen=True)
class GateBenefitCoupling(EnergyCoupling):
    """Coupling that rewards opening a gate when domain improvement exists.

    Energy: F = -w * η_gate * Δη_domain
    """
    weight: float = 1.0
    delta_key: str = "delta_eta_domain"

    def coupling_energy(
        self,
        eta_i: OrderParameter,  # gate
        eta_j: OrderParameter,  # domain (unused directly)
        constraints: Mapping[str, Any],
    ) -> float:
        delta = float(constraints.get(self.delta_key, 0.0))
        eta_gate = float(eta_i)
        return float(-self.weight * eta_gate * delta)

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        delta = float(constraints.get(self.delta_key, 0.0))
        # THE WORMHOLE: gradient independent of η_gate!
        gi = float(-self.weight * delta)
        gj = 0.0
        return gi, gj
```

**Key line**: `gi = -self.weight * delta`, no η_gate dependency.

---

## Demo: see it in action

Run the demonstration:

```powershell
uv run python -m experiments.demo_wormhole
```

### Results (actual run)

**Scenario 1: without CGBC** (Standard Quadratic Coupling)
- Initial gate: η = 0.000 (completely CLOSED)
- After 30 steps: η = 0.112 (barely opens)
- Final energy: +0.006
- Behavior: Gate opens slowly and only receives the local mismatch signal

**Scenario 2: with CGBC** (`GateBenefitCoupling`, wormhole nickname)
- Initial gate: η = 0.000 (same, completely CLOSED)
- After 30 steps: η = 0.535 (4.8x more open)
- Final energy: -0.199
- Behavior: Gate opens more over the same step budget when the supplied benefit signal is positive.

| Metric | Without CGBC | With CGBC | Improvement |
|--------|-----------------|---------------|-------------|
| Final η_gate | 0.112 | 0.535 | **4.8x more open** |
| Energy Drop | 0.332 | 0.411 | **24% larger in this run** |
| Final Energy | +0.006 | **-0.199** | **lower in this run** |

CGBC lets the gate receive a gradient signal from the supplied benefit estimate even when the gate is closed.

---

## Why this matters

### For neuro-symbolic AI

Traditional neural networks struggle with **sparse, structured reasoning** because:
- They need dense connections everywhere (expensive)
- They can't efficiently represent "potential but inactive" paths
- They waste compute on unlikely branches

**CGBC supports**:
- **Sparse activation**, open gates when the supplied benefit estimate supports opening
- **Non-local updates**, downstream estimates can influence gate opening
- **Targeted exploration**, test hypothetical benefits without full execution
- **Escape from some local traps**, when benefit estimates are informative

### Real-world applications

#### Code synthesis
```python
# Baseline: generate and test every candidate
for candidate in all_possible_functions:  # expensive!
    if test_suite(candidate):
        return candidate

# With CGBC: supplied benefit estimates guide generation
gate_benefits = {
    func: estimate_test_improvement(func)  # cheap lookahead
    for func in candidate_pool
}
# CGBC gradient pulls gates open only if benefit is high
activated = [f for f in candidates if cgbc_grad(f) > threshold]
# Now test only selected candidates.
```

#### Sequence processing
```python
# Baseline: process the whole sequence
for token in long_sequence:
    output = process(token)  # all tokens treated equally

# With CGBC: Gate important tokens only
for token in long_sequence:
    benefit = estimate_information_gain(token)  # lookahead
    if cgbc_gradient(benefit) > threshold:
        output = process(token)  # only process if valuable
    else:
        output = skip(token)  # skip low-value tokens
```

#### Planning / search
```python
# Baseline: breadth-first or random expansion
def explore(state):
    for action in all_actions:  # explore everything
        new_state = apply(action, state)
        ...

# With CGBC: Value-guided expansion
def explore(state):
    action_values = {a: estimate_future_reward(a, state) for a in actions}
    # CGBC pulls high-value branches open first
    for action in sorted(actions, key=lambda a: cgbc_grad(action_values[a])):
        if action_values[action] > threshold:
            new_state = apply(action, state)
            ...
```

---

## Theoretical foundations

CGBC is mathematically grounded in:

### 1. Small-Gain theorem (control theory)
- **Zames 1966**: feedback stability
- Couplings create "feedback loops" across the system
- CGBC allows beneficial loops even when paths are inactive

### 2. Turbo codes (information theory)
- **Berrou et al. 1993**: turbo-code error correction
- Key insight: **Extrinsic information exchange** between decoders
- CGBC is the energy-based analog: later context sends a soft update signal to an earlier gate

### 3. Hindsight experience replay (reinforcement learning)
- **Andrychowicz et al. 2017**: reference algorithm for relabeling trajectories
- Failed trajectories can be relabeled as success for what they achieved
- CGBC is an energy-based analogy, not the same algorithm

### 4. Noisy channel coding
- **Shannon 1948**: Channel capacity requires redundancy
- Later context can act like an update signal for earlier uncertain decisions
- CGBC sends this update signal to the gate variable

**Key Papers**:
- Zames, G. (1966). "On the input-output stability of time-varying nonlinear feedback systems." *IEEE TAC*.
- Berrou, C., Glavieux, A., & Thitimajshima, P. (1993). "Near Shannon limit error-correcting coding and decoding: Turbo-codes." *ICC*.
- Andrychowicz, M., et al. (2017). "Hindsight Experience Replay." *NeurIPS*.

---

## Usage

### Basic example

```python
from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling, QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule

# Create modules
gate_module = EnergyGatingModule(gain_fn=lambda _: 0.1, a=0.2, b=0.1)
domain_module = EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.3, b=0.2)

# Estimate downstream benefit (your application logic)
potential_benefit = 0.3  # supplied estimate of downstream energy reduction

# Create coordinator with CGBC (wormhole nickname)
coord = EnergyCoordinator(
    modules=[gate_module, domain_module],
    couplings=[
        (0, 1, QuadraticCoupling(weight=0.5)),  # standard spring
        (0, 1, GateBenefitCoupling(weight=2.0, delta_key="delta_benefit")),  # CGBC
    ],
    constraints={"delta_benefit": potential_benefit},
    step_size=0.05,
)

# Start with gate CLOSED
etas = [0.0, 0.5]  # gate=0 (closed), domain=0.5

# Relax: CGBC gradient can pull the gate open.
etas_final = coord.relax_etas(etas, steps=30)

print(f"Gate opened to: {etas_final[0]:.3f}")  # e.g., 0.535!
```

### Computing the benefit signal

The caller must compute `delta_benefit`, the estimated improvement if the gate were to open:

```python
def compute_benefit(domain_module, current_eta, inputs):
    """Estimate benefit of activating this gate."""
    # Option 1: Lookahead heuristic
    if gate_activates:
        eta_new = 1.0
    else:
        eta_new = 0.0
    delta = eta_new - current_eta

    # Option 2: Finite difference (accurate)
    E_off = domain_module.local_energy(0.0, inputs)
    E_on = domain_module.local_energy(1.0, inputs)
    delta = E_off - E_on  # benefit = how much energy we save

    return delta
```

### Advanced: damped CGBC

For smoother activation curves:

```python
from core.couplings import DampedGateBenefitCoupling

# Energy: F = -w * (η_gate ** eta_power) * damping * delta
coupling = DampedGateBenefitCoupling(
    weight=2.0,
    delta_key="delta_benefit",
    damping=0.8,              # soften the effect
    eta_power=1.5,            # non-linear gate response
    positive_scale=1.0,       # boost positive benefits
    negative_scale=0.5,       # dampen negative signals
)
```

---

## Architecture pattern: redemption

CGBC is the local mechanism behind the "Redemption" architecture pattern used throughout the framework:

**Redemption** = later context supplies an update signal for earlier provisional decisions

### Pattern instances

| Repository | η represents | CGBC manifests as | Result |
|------------|-------------|----------------------|---------|
| **This framework** | Generic order parameter | `GateBenefitCoupling` | Inactive modules can receive an opening force |
| `Inverse_ND_Reconstruction` | Loop trajectory parameters | Refinement stage corrects provisional loops | Explainable closed-loop reconstruction |
| `Normalized_Dynamic_OPT` | Cluster centers | Later points reassign provisional assignments | compact compression with geometric relations kept |
| `Hallucinations_Noisy_Channels` | Latent sequence state | Later tokens update earlier provisional state (when allowed) | Noisy-channel analysis |
| `Spaced_Repetition_Learning` | Replay priority | Hard/diverse samples force correction | Inference-time self-improvement |

In this repository, CGBC is the primitive mechanism for applying the supplied update signal to a gate.

---

## Comparison with alternatives

| Approach | Handles Closed Gates? | Needs Noise? | Sparse-Friendly? | Formal Guarantees? |
|----------|---------------------|--------------|------------------|-------------------|
| **Standard Physics** |  No gradient |  Yes |  No |  No |
| **Dense Neural Nets** | N/A (fixed dense connectivity) |  No |  No (dense) |  No |
| **Sparse Neural Nets** |  Dead ReLU problem |  Yes |  Partial |  No |
| **RL Exploration** |  With exploration |  Yes (ε-greedy) |  Partial |  No |
| **CGBC (wormhole nickname)** |  Non-local gradient from supplied estimate |  No |  Yes |  Scoped to stated stability assumptions |

---

## Testing

**Demo script**: [`experiments/demo_wormhole.py`](../experiments/demo_wormhole.py)

```powershell
# Run interactive demonstration
uv run python -m experiments.demo_wormhole

# Expected output:
# - Without CGBC: Final eta_gate = 0.112 (local signal only)
# - With CGBC:    Final eta_gate = 0.535 (non-local supplied benefit)
```

**Unit tests**: Covered in `tests/test_couplings.py` and `tests/test_gate_benefit_*.py`

```powershell
# Test GateBenefitCoupling
uv run -m pytest tests/test_couplings.py -k gate_benefit -v

# Test ADMM proximal operators for gate couplings
uv run -m pytest tests/test_admm_damped_gate_benefit.py -v
```

---

## Limitations and future work

### Current status

The core `GateBenefitCoupling` is implemented and exercised in this repository's demos and tests.

### Known limitations

1. **Benefit estimation**: Computing `delta_benefit` accurately requires domain knowledge
   - Heuristics and finite differences are approximate
   - True benefit may depend on complex downstream effects
   - **Mitigation**: Use conservative estimates and rely on acceptance guards for harmful steps

2. **Scaling**: On large systems (1000+ modules), benefit computation can be expensive
   - Each gate needs a lookahead or finite-difference estimate
   - **Mitigation**: Amortize with caching, use sparse active sets

3. **Theoretical gap**: The local gradient formula is defined, but this document does not prove global optimality
   - Does CGBC converge to global minimum?
   - Under what conditions does it reduce step count or final energy relative to noise-based exploration?
   - **Status**: Tested in synthetic gating cases; broader benefit claims require task-level ablations

### Future directions

- **Learned benefit estimators**: Train a small network to predict `delta_benefit`
- **Hierarchical CGBCs**: Multi-level gates with nested benefit signals
- **Adaptive damping**: Learn `damping` and `eta_power` parameters online
- **Formal analysis**: Derive convergence conditions for CGBC dynamics where possible

---

## FAQ

### Q: Is this just a fancy way of saying "skip connections"?

**A**: No. Skip connections are architectural and fixed in topology. CGBCs apply a dynamic gate gradient based on a supplied benefit estimate.

Skip connections: "Always add input to output"
CGBC: "Let estimated downstream value pull inactive paths open"

### Q: Doesn't this violate causality?

**A**: No. The gradient is **causal** (computed from current state). It's the **topology** that's non-local: gradient flows through disconnected regions.

Practical interpretation: this is a designed coupling term, not a literal physical tunneling process.

### Q: Can I use this with PyTorch/JAX/Numpy?

**A**: Yes. CGBC is implemented as a coupling function. We provide:
- Pure Python: `core/couplings.py`
- JAX backend: `core/jax_backend.py`
- PyTorch backend: `core/torch_backend.py` (experimental)

### Q: What if my benefit estimate is wrong?

**A**: The system is guarded. Overestimates can open gates unnecessarily, but monotonic energy acceptance can reject harmful steps. Underestimates can slow opening.

### Q: Can I combine multiple CGBC terms?

**A**: Yes. You can have multiple `GateBenefitCoupling` terms with different `delta_key` constraints. They compose naturally in the energy sum.

---

## Citation

If you use this repository in your research, please cite it. This is ongoing work; we would like to know your opinions and experiments. Thank you.

**Authors:** Oscar Goldman, Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業).

**Reference (author-year format):** Goldman, O. (2025). *Complexity from Constraints: counterfactual gate-benefit coupling*. Software repository. Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業). Component of the Neuro-Symbolic Homeostat; addresses zero-gradient deadlock in sparse energy-based models when downstream benefit is supplied to `GateBenefitCoupling`. Wormhole coupling is the implementation nickname.

---

## Related documentation

- **Core philosophy**: [`Complexity_from_Constraints.md`](../Complexity_from_Constraints.md), the "five equations" framework
- **Couplings**: [`core/couplings.py`](../core/couplings.py), implementation of `GateBenefitCoupling`
- **Gating modules**: [`README_GATING.md`](README_GATING.md), energy gating and gate modules
- **Stability**: [`STABILITY_GUARANTEES.md`](STABILITY_GUARANTEES.md), CGBC interaction with stability guard
- **Redemption Architecture**: [`Complexity_from_Constraints.md`](../Complexity_from_Constraints.md) (lines 122-158)

---

## Quick start

```python
from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling

# 1. Estimate benefit
benefit = estimate_domain_improvement()  # your logic here

# 2. Add CGBC (wormhole nickname)
coord = EnergyCoordinator(
    modules=[gate_mod, domain_mod],
    couplings=[
        (0, 1, GateBenefitCoupling(weight=1.0, delta_key="benefit")),
    ],
    constraints={"benefit": benefit},
)

# 3. Observe gate updates from the benefit signal
etas = coord.relax_etas([0.0, 0.5], steps=50)
```

After setup, CGBC contributes this non-local gradient term during relaxation.

---

**Status**: Core mechanism validated in this repository's demos and tests.

The mechanism is direct: a coupling applies gate gradients from a supplied downstream benefit estimate.

