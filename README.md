# The Neuro-Symbolic Homeostat
### Logic as Physics. Inference as Relaxation.

> **"A fast, modular 'System-2' layer that grounds the outputs of fast 'System-1' models in strict constraints using physics grounded self-regulation."**

---

##  Context & Origins

**Note:** This repository is a focused validation lab for the **Neuro-Symbolic Homeostat**, a specific component of the broader **Datamutant** library. 

It serves as the "playground of ideas" and rigorous testing ground for the claims made in our thesis (*Complexity from Constraints*). The core technologies developed here—Precision-Scaled Orthogonal Noise, Wormhole Couplings, and Stability Projectors—are destined for the production-grade Datamutant ecosystem being developed in the *Complexity from Constraints* repository.

---

##  The Big Idea

Modern AI faces a dilemma:
*   **System 1 (Neural Nets):** Fast and intuitive, but prone to "hallucinating" and breaking rules.
*   **System 2 (Symbolic Logic):** Correct and rigorous, but brittle and slow.

The **Neuro-Symbolic Homeostat** builds a bridge. It treats **logic as physics**. 
Instead of running fragile logical proofs, we define a "landscape" where breaking a rule costs "energy." The system then "relaxes" (like a ball rolling downhill) to find the state of lowest energy—a state that satisfies as many constraints as possible while staying close to the data.

### The Philosophy: Don’t Just Descend, Shape

Standard optimization asks: *"Given a fixed energy landscape, how do I find the bottom?"*  
**Our approach": Shape and Move**

We do not rely on a single "super-optimizer." Instead, we combine four focused mechanisms that manipulate the energy surface and the agent's trajectory in complementary ways. By exploiting algebraic equivalences, we replace expensive global operations with fast, local updates, yielding a system that behaves like a sophisticated global solver but runs with the speed of a local one.

See **[Moving the Landscape](docs/README_MOVING_THE_LANDSCAPE.md)** for the unified strategy.

---

###  Core Concepts (Simply Explained)

#### 1. Relaxation (The "Settling" Process)
Imagine a bedspring mattress. When you lie down, the springs adjust to support you. Our system works similarly: "Modules" (variables) are connected by "Couplings" (springs/constraints). When new data arrives, the system vibrates and settles into a new stable shape. This is **inference**.

#### 2. Precision-Scaled Orthogonal Noise (PSON)
*The Problem:* To find the best answer, you need to explore. But shaking the system randomly (standard noise) often destroys the progress you've already made (pushing you back uphill).
*The Solution:* **Tangent Noise**. We inject noise *only* in the directions that don't increase energy. It's like exploring a flat valley floor without climbing the walls. We scale this exploration based on "precision"—we explore the uncertain stuff (slack) more than the certain stuff (stiff).

#### 3. The Stability Projector (Small-Gain)
*The Problem:* If components interact too strongly, feedback loops can cause the system to explode or oscillate wildly.
*The Solution:* A "Governor" that watches the system's energy. If things get too heated, it mathematically clamps the interactions (using Gershgorin bounds) to guarantee the system always settles down.

#### 4. Wormhole Couplings
*The Problem:* In sparse logic gates, if a gate is closed, no information flows. The system can't learn that opening the gate would be a good idea because the gradient is zero.
*The Solution:* A **Wormhole**. We create a "virtual" connection that teleports credit through the closed gate. It tells the gate: "If you *were* open, this would help." This allows the system to plan and switch strategies dynamically.

#### 5. Distributed Linear Algebra Processor
*The Insight:* Crucially, this system doesn't run logical proofs step-by-step. It acts as a **distributed linear algebra processor**. By converting logic into energy landscapes, we solve "reasoning" problems using fast, parallel matrix operations (like Jacobi iterations)—running inference at the speed of numerical physics.

---

##  Documentation Map

We believe in documenting *why*, not just *how*.

### Theory & Mechanisms
*   **[Thesis](The_Neuro_Symbolic_Homeostat_Paper_V1.md)**: The canonical academic paper describing the full system.
*   **[Moving the Landscape](docs/README_MOVING_THE_LANDSCAPE.md)**: High-level overview of how our four pillars combine.
*   **[Tangent Noise / PSON](docs/README_TANGENT_NOISE_PSON.md)**: Why standard noise fails and how we fix it.
*   **[Stability Guarantees](docs/STABILITY_GUARANTEES.md)**: The control theory math (Small-Gain) that keeps us safe.
*   **[Wormholes](docs/README_WORMHOLE.md)**: Non-local credit assignment for gating and planning.
*   **[Stiffness Updates](docs/README_STIFFNESS_UPDATES.md)**: How precision acts as stiffness (Newton-like scaling).

### Implementation Details
*   **[Auto Scheduling](docs/README_AUTO_SCHEDULING.md)**: How the system tunes its own noise and step sizes.
*   **[Observability](docs/README_OBSERVABILITY.md)**: How we measure what's happening (metrics, logs).
*   **[Safe Defaults](docs/README_SAFE_DEFAULTS.md)**: The recommended configuration for reproducible results.

---

##  Experiments & Demos

We assert early, fail fast, and validate empirically.

### Ready-to-Run (Windows PowerShell)

**1. The Wormhole Effect**
Demonstrates how the system learns to open a closed gate using non-local gradients.
```powershell
python -m experiments.demo_wormhole
```

**2. Metric-Orthogonal Projection**
Shows how PSON respects the geometry of the problem (noise dot gradient ≈ 0).
```powershell
python -m experiments.demo_metric_orthogonal
```

**3. Operator Splitting**
Compare standard relaxation with advanced operator splitting techniques.
```powershell
python -m experiments.demo_operator_splitting
```

**4. Small-Gain Sweep**
Validate the stability boundaries by sweeping interaction strengths.
```powershell
python -m experiments.sweep_smallgain --quick
```

---

##  Citation

If you use this repository in your research, please cite it as below:

```bibtex
@software{complexity_from_constraints_homeostat_2025,
  title        = {Complexity from Constraints: The Neuro‑Symbolic Homeostat},
  author       = {Goldman, Oscar},
  organization = {Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業},
  year         = {2025},
  note         = {Fast matrix–message relaxation with precision‑scaled orthogonal noise and stability projection}
}
```

---

*Built with ❤️ by the Shogu Research Group.*
