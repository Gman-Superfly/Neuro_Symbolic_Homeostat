# The Neuro-Symbolic Homeostat
### Logic as Physics. Inference as Relaxation.

> **"A modular System-2 relaxation layer that applies explicit constraints to System-1 outputs using energy-based updates."**

---

## Context and origins

**Note:** This repository is a focused validation lab for the **Neuro-Symbolic Homeostat**, a specific component of the broader **Datamutant** library.

It serves as a validation lab for ideas from *Complexity from Constraints*. The core mechanisms in this repository include precision-scaled orthogonal noise, counterfactual gate-benefit couplings (CGBCs, nicknamed wormhole couplings), and stability projectors. The repository documents synthetic demos and implementation details for these mechanisms, and it feeds design and validation results back into the broader Datamutant ecosystem.

---

## Proof status

This repository separates mechanism validity from empirical benefit. A mechanism can be mathematically well-defined without showing that it reduces step count, lowers final energy, or improves behavior across problem classes.

What is analytically supported:

- **CGBC** gives a nonzero gate gradient at \(\eta_{\text{gate}} = 0\) when \(\Delta_{\text{benefit}} \neq 0\).
- **PSON** has zero first-order energy change for perturbations orthogonal to the gradient.
- **Small-Gain** gives a contraction condition under linear/SPD assumptions and sufficient Gershgorin bounds.
- **Stiffness/GaBP updates** match known linear solvers under quadratic/SPD assumptions.

What remains empirical:

- Whether these mechanisms reduce step count.
- Whether final energy decreases more than baseline methods across problem classes.
- Whether CGBC helps when benefit estimates are noisy or wrong.
- Whether the combined system beats simpler baselines.

The current demos test specific synthetic cases. They do not show universal benefit.

The optimization mathematics used here is standard: diagonal preconditioning, Gershgorin row-sum bounds, and the gradient descent condition \(\alpha < 2/L\). The repository-specific design is the composable curvature contract: modules expose local stiffness, couplings expose curvature bounds, and the coordinator composes those reports into precision-aware updates and stability caps. Current tests show both sides of this contract: in a tight quadratic regime, curvature awareness lets the system converge where plain gradient descent stalls above \(2/L\); in a mixed constraint regime, the conservative cap can trade speed for margin while preconditioning still converges.

---

## The big idea

This repository studies a specific coordination problem:
*   Neural scorers can produce useful continuous outputs while violating explicit constraints.
*   Symbolic rule systems can enforce constraints but often require crisp inputs and discrete choices.

The **Neuro-Symbolic Homeostat** treats some logical constraints as energy terms.
Instead of running a discrete proof procedure, we define an energy function where violating a rule increases cost. The system then relaxes toward lower energy, seeking a state that satisfies more constraints while staying close to the data.

### The philosophy: don’t just descend, shape the objective

Standard optimization asks: *"Given a fixed energy function, how do I find the minimum?"*
**Our approach: shape and move**

We do not rely on a single optimizer. The system combines objective shaping, stability projection, preconditioned relaxation, and tangent exploration. Some parts change the effective objective, while others change the update geometry or trajectory.

See **[Objective shaping](docs/README_MOVING_THE_LANDSCAPE.md)** for the unified strategy.

---

### Core concepts (simply explained)

#### 1. Relaxation
Modules expose variables called order parameters. Couplings connect those variables through energy terms. When new data arrives, the coordinator updates the order parameters to reduce total energy while respecting the configured guards.

#### 2. Precision-scaled orthogonal noise (PSON)
*The Problem:* Exploration can help avoid stalls, but random noise can add an uphill component against the gradient.
*The Solution:* **Tangent Noise**. We inject noise in directions that do not increase energy to first order. We scale this exploration with precision so uncertain coordinates receive larger perturbations than stiff coordinates.

#### 3. The stability projector (Small-Gain)
*The Problem:* If components interact too strongly, feedback loops can cause divergence or oscillation.
*The Solution:* A stability guard that caps interactions using Gershgorin-based bounds and step-size checks. In linear and SPD regimes, this gives a conservative contraction condition. In mixed regimes, it acts as a practical guard.

#### 4. Counterfactual gate-benefit coupling (CGBC)
*The Problem:* In sparse logic gates, if a gate is closed, the local gradient can be zero. The gate may not receive the downstream signal that opening it would reduce energy.
*The Solution:* A **counterfactual gate-benefit coupling (CGBC)**, nicknamed a wormhole coupling. We add a coupling term that applies a gradient on a gate proportional to a caller-supplied estimate of downstream benefit. This lets closed gates receive a non-local update signal.

#### 5. Distributed linear algebra processor
*The Insight:* This system does not run symbolic proofs step by step. It uses matrix updates, including Jacobi-style iterations on quadratic blocks, to perform relaxation.

---

## Documentation map

We believe in documenting *why*, not just *how*.

### Theory and mechanisms
*   **[Thesis](The_Neuro_Symbolic_Homeostat_Paper_V1.md)**: Draft paper describing the current system and scope limits.
*   **[Moving the Landscape](docs/README_MOVING_THE_LANDSCAPE.md)**: High-level overview of how the four mechanisms combine.
*   **[Tangent Noise / PSON](docs/README_TANGENT_NOISE_PSON.md)**: How tangent noise controls first-order energy change.
*   **[Stability Guarantees](docs/STABILITY_GUARANTEES.md)**: The control theory math and scoped assumptions behind the guards.
*   **[Counterfactual gate-benefit coupling](docs/README_WORMHOLE.md)**: Non-local gate updates from caller-supplied benefit estimates. “Wormhole” is the implementation nickname.
*   **[Stiffness Updates](docs/README_STIFFNESS_UPDATES.md)**: How precision acts as stiffness (Newton-like scaling).

### Implementation details
*   **[Auto Scheduling](docs/README_AUTO_SCHEDULING.md)**: How the system tunes its own noise and step sizes.
*   **[Observability](docs/README_OBSERVABILITY.md)**: How we measure what's happening (metrics, logs).
*   **[Safe Defaults](docs/README_SAFE_DEFAULTS.md)**: The recommended configuration for reproducible results.

---

## Experiments and demos

We assert early, fail fast, and validate empirically.

### Ready-to-Run (Windows PowerShell)

**1. Counterfactual gate-benefit coupling (CGBC), wormhole nickname**
Demonstrates how a caller-supplied benefit estimate creates a non-local gradient on a closed gate.
```powershell
python -m experiments.demo_wormhole
```

**2. Metric-orthogonal projection**
Shows how PSON respects the geometry of the problem (noise dot gradient ≈ 0).
```powershell
python -m experiments.demo_metric_orthogonal
```

**3. Operator splitting**
Compare standard relaxation with advanced operator splitting techniques.
```powershell
python -m experiments.demo_operator_splitting
```

**4. Small-Gain sweep**
Sweeps interaction strengths to test conservative stability boundaries in the included scenarios.
```powershell
python -m experiments.sweep_smallgain --quick
```

**5. Constraint correction**
Shows a small System-1 output before and after stability-guarded relaxation against explicit constraints: sum-to-one, mutual exclusion, and monotonicity.
```powershell
python -m experiments.demo_constraint_correction
```

---

## Testing

Use this command as the canonical local test run:

```powershell
uv run -m pytest tests -v
```

The pytest defaults are recorded in `pyproject.toml`.

---

## Citation

If you use this repository in your research, please cite it. This is ongoing work; we would like to know your opinions and experiments. Thank you.

**Authors:** Oscar Goldman, Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業).

**Reference (author-year format):** Goldman, O. (2025). *Complexity from Constraints: The Neuro-Symbolic Homeostat*. Software repository. Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業). Matrix-message relaxation with precision-scaled orthogonal noise and stability projection.

---

Built by the Shogu Research Group.
