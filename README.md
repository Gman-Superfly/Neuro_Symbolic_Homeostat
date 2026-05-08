# The Neuro-Symbolic Homeostat
### Logic as Physics. Inference as Relaxation.

> **"A modular System-2 relaxation layer that applies explicit constraints to System-1 outputs using energy-based updates."**

---

## Context and origins

**Note:** This repository is a focused validation lab for the **Neuro-Symbolic Homeostat**, a specific component of the broader **Datamutant** library.

It serves as a validation lab for ideas from *Complexity from Constraints*. The core mechanisms in this repository include precision-scaled orthogonal noise, counterfactual gate-benefit couplings (CGBCs, nicknamed wormhole couplings), and stability projectors. The repository documents synthetic demos and implementation details for these mechanisms, and it feeds design and validation results back into the broader Datamutant ecosystem.

---

## Proof status

This repository separates mechanism validity from empirical benefit. A mechanism can be mathematically well-defined without proving that it improves convergence speed, final energy, or robustness across problem classes.

What we can prove:

- **CGBC** gives a nonzero gate gradient at \(\eta_{\text{gate}} = 0\) when \(\Delta_{\text{benefit}} \neq 0\).
- **PSON** has zero first-order energy change when noise is orthogonal to the gradient.
- **Small-Gain** gives contraction under linear/SPD assumptions and sufficient Gershgorin bounds.
- **Stiffness/GaBP updates** match known linear solvers under quadratic/SPD assumptions.

What remains empirical:

- Whether these mechanisms improve convergence speed.
- Whether final energy improves across problem classes.
- Whether CGBC helps when benefit estimates are noisy or wrong.
- Whether the combined system beats simpler baselines.

The current demos test specific synthetic cases. They do not prove universal benefit.

---

## The big idea

Modern AI faces a dilemma:
*   **System 1 (Neural Nets):** Fast and intuitive, but prone to "hallucinating" and breaking rules.
*   **System 2 (Symbolic Logic):** Correct and rigorous, but brittle and slow.

The **Neuro-Symbolic Homeostat** treats some logical constraints as energy terms.
Instead of running brittle logical proofs, we define an energy function where violating a rule increases cost. The system then relaxes toward lower energy, seeking a state that satisfies more constraints while staying close to the data.

### The philosophy: don’t just descend, shape

Standard optimization asks: *"Given a fixed energy function, how do I find the minimum?"*
**Our approach": Shape and Move**

We do not rely on a single optimizer. The system combines objective shaping, stability projection, preconditioned relaxation, and tangent exploration. Some parts change the effective objective, while others change the update geometry or trajectory.

See **[Moving the Landscape](docs/README_MOVING_THE_LANDSCAPE.md)** for the unified strategy.

---

### Core concepts (simply explained)

#### 1. Relaxation (the "settling" process)
Imagine a bedspring mattress. When you lie down, the springs adjust to support you. Our system works similarly: "Modules" (variables) are connected by "Couplings" (springs/constraints). When new data arrives, the system vibrates and settles into a new stable shape. This is **inference**.

#### 2. Precision-scaled orthogonal noise (PSON)
*The Problem:* Exploration can help avoid stalls, but random noise can add an uphill component against the gradient.
*The Solution:* **Tangent Noise**. We inject noise only in directions that do not increase energy to first order. We scale this exploration with precision so uncertain coordinates receive larger perturbations than stiff coordinates.

#### 3. The stability projector (Small-Gain)
*The Problem:* If components interact too strongly, feedback loops can cause the system to explode or oscillate wildly.
*The Solution:* A stability guard that caps interactions using Gershgorin-based bounds and step-size checks. In linear and SPD regimes, this gives a conservative contraction condition. In mixed regimes, it acts as a practical guard.

#### 4. Counterfactual gate-benefit coupling (CGBC)
*The Problem:* In sparse logic gates, if a gate is closed, no information flows. The system can't learn that opening the gate would be a good idea because the gradient is zero.
*The Solution:* A **counterfactual gate-benefit coupling (CGBC)**, nicknamed a wormhole coupling. We add a coupling term that applies a gradient on a gate proportional to a caller-supplied estimate of downstream benefit. This lets closed gates receive a non-local update signal.

#### 5. Distributed linear algebra processor
*The Insight:* This system does not run symbolic proofs step by step. It uses matrix updates, including Jacobi-style iterations on quadratic blocks, to perform fast relaxation.

---

## Documentation map

We believe in documenting *why*, not just *how*.

### Theory and mechanisms
*   **[Thesis](The_Neuro_Symbolic_Homeostat_Paper_V1.md)**: Draft paper describing the current system and scope limits.
*   **[Moving the Landscape](docs/README_MOVING_THE_LANDSCAPE.md)**: High-level overview of how the four mechanisms combine.
*   **[Tangent Noise / PSON](docs/README_TANGENT_NOISE_PSON.md)**: Why standard noise fails and how we fix it.
*   **[Stability Guarantees](docs/STABILITY_GUARANTEES.md)**: The control theory math (Small-Gain) that keeps us safe.
*   **[Counterfactual gate-benefit coupling](docs/README_WORMHOLE.md)**: Non-local credit assignment for gating and planning. “Wormhole” is the implementation nickname.
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

---

## Citation

If you use this repository in your research, please cite it. This is ongoing work; we would like to know your opinions and experiments. Thank you.

**Authors:** Oscar Goldman, Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業).

**Reference (author-year format):** Goldman, O. (2025). *Complexity from Constraints: The Neuro-Symbolic Homeostat*. Software repository. Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業). Fast matrix-message relaxation with precision-scaled orthogonal noise and stability projection.

---

Built by the Shogu Research Group.
