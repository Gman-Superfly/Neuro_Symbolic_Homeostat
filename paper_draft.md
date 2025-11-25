# Complexity from Constraints: The Neuro‑Symbolic Homeostat
## Fast Matrix–Message Relaxation with Precision‑Scaled Orthogonal Noise and Stability Projection

**Authors:** Oscar Goldman (Gman‑Superfly), Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業  
**Date:** November 2025  
**Status:** Draft with working code and demos


##NOTE:
no need to introduce a general bilinear off-diagonal coupling right now. I can wait until migrated tests from CFC are in place
---

### Abstract

We present a neuro‑symbolic coordination framework that treats logic as physics and inference as relaxation. The system composes typed modules with explicit energy terms (locals and couplings) and regulates them with three orthogonal mechanisms: (i) Precision‑Scaled Orthogonal Noise (PSON) that explores the null‑space of uncertainty without breaking energy monotonicity, (ii) a Stability Projector (Small‑Gain allocator) that enforces contraction via Gershgorin‑style bounds, and (iii) Wormhole couplings that provide non‑local credit assignment akin to Equilibrium Propagation nudges. For quadratic/Gaussian sub‑problems we exploit a tight equivalence between Gaussian Belief Propagation (GaBP) and classical linear iterations (Jacobi/Gauss–Seidel), showing that message passing and (preconditioned) gradient descent become the same computation under standard conditions (SPD precision, appropriate scheduling). The result is a fast, modular “System‑2” layer that corrects hallucinations of fast “System‑1” models using thermodynamically grounded self‑regulation, with production‑grade implementation, observability, and stability guarantees.

---

## 1. Introduction

Modern “System‑1” models are fast but brittle under hard constraints; symbolic solvers are correct but brittle under noise. We develop a middle path: a Neuro‑Symbolic Homeostat that
- maintains internal stability (constraints) while adapting to external stress (data),
- explores safely without violating invariants, and
- evolves structure as uncertainty changes.

Our core design principle is entity‑first, physics‑grounded coordination: modules expose order parameters and energies; the coordinator relaxes the global free energy with null‑space exploration, stability guards, and non‑local corrections.

---

## 2. Theoretical Framework

### 2.1 Four Lenses
- Physics (Energy Minimization): Relax to a ground state that minimizes total energy, locals + couplings.
- Control Theory (H∞ robustness): Small‑gain constraints ensure loop gains < 1, guaranteeing contraction.
- Statistics (Gaussian Graphical Models): Couplings act as messages; precision (inverse variance) is stiffness.
- Information Theory (Channel Capacity): Manage bandwidth vs error; adapt to SNR via precision‑aware updates.

### 2.2 Precision‑Scaled Orthogonal Noise (PSON)
Standard Langevin noise breaks monotonicity. We inject noise in the tangent plane orthogonal to the gradient and scale it by inverse precision (local curvature):

\[ \xi_{\mathrm{injection}} \propto \Lambda^{-1}\,\mathrm{proj}_{\nabla \mathcal{F}^\perp}\big(\mathcal{N}(0, I)\big) \]

PSON explores flat directions (null‑space) without fighting descent, providing robust exploration and smoothing (dithering) that suppresses high‑frequency potholes while tracking deep valleys. PSON replaces the prior “Golden Spike” name for clarity.

### 2.3 Wormhole Effect (Non‑Local Gradient Teleportation)
Closed gates receive forces proportional to downstream potential benefit. With gate–benefit energy
\[ F_{\text{gate}} = -w\, \eta_{\text{gate}}\, \Delta_{\text{benefit}}, \]
the gradient w.r.t. the gate is independent of the current gate value:
\[ \frac{\partial F}{\partial \eta_{\text{gate}}} = -w\, \Delta_{\text{benefit}}. \]
This provides a non‑local correction akin to the “nudge” in Equilibrium Propagation—enabling credit assignment without backprop through inactive paths.

### 2.4 Stability and the GaBP Link
We keep the iteration contractive with a Small‑Gain projector: Gershgorin row/global bounds cap couplings so the Jacobian spectral radius stays < 1. For quadratic regimes, relaxation with precision tracking is equivalent to Gaussian Belief Propagation (GaBP)/Gauss–Seidel; precision acts as stiffness, producing Newton‑like scaling for fast convergence.

---

## 3. Message Passing ↔ Gradient Descent: When Are They the Same?

Consider quadratic energy \( f(x) = \tfrac{1}{2} x^\top J x - h^\top x \) with SPD precision matrix \(J\). Solving \(Jx = h\) via iterative methods yields the following equivalences:

- GaBP (means) with a synchronous schedule matches Jacobi; with a sequential schedule matches Gauss–Seidel (GS).
- Gradient descent with diagonal preconditioning (\(\alpha = D^{-1}\)) reproduces Jacobi; with triangular preconditioning (\((D+L)^{-1}\)) reproduces GS.

Therefore, for Gaussian/quadratic sub‑problems under SPD and standard scheduling, “message passing” and “(preconditioned) gradient descent” are the same computation up to ordering. Convergence requires the spectral radius of the iteration matrix < 1; for GaBP this is “walk‑summability,” closely related to diagonal dominance. This dovetails with the Small‑Gain constraint (loop gains < 1).

**Accuracy Note (paper‑safe phrasing):**

> For quadratic energies with SPD precision \(J\), minimizing \(f(x)\) is equivalent to solving \(Jx=h\). In this setting, loopy Gaussian belief propagation (GaBP) for the means is algebraically equivalent to classical linear solvers: with a synchronous schedule it reduces to Jacobi; with a sequential schedule it reduces to Gauss–Seidel. Likewise, gradient descent with diagonal (resp. triangular) preconditioning reproduces Jacobi (resp. Gauss–Seidel). Under walk‑summability/spectral‑radius < 1 (i.e., loop gains < 1), all of these iterations converge to the unique minimizer.

**Implementation Note (scope and realization):**

We scope GaBP claims strictly to SPD/quadratic blocks and the standard scheduling equivalence (Jacobi/GS). Our implementation realizes the same linear iterations via gradient descent with diagonal/triangular preconditioning (and optional coordinate updates), not via explicit message objects. This preserves the algebraic equivalence and convergence conditions without introducing a separate message type.

---

## 4. Architecture & Mechanisms

### 4.1 Entities, Energies, and Precision
Modules expose order parameters and implement local energies. Couplings encode interactions (springs, hinges, wormholes). A `SupportsPrecision` interface elevates curvature (precision) to a first‑class signal that scales steps and modulates PSON. The coordinator caches diagonal precision and composes vectorized relaxations over the graph with compile‑time graph caching to avoid Python overhead.

### 4.2 Stability Projector (Small‑Gain Allocator)
The Small‑Gain allocator enforces contraction by budgeting Gershgorin‑estimated Lipschitz margins. In strictly Gaussian sub‑problems it acts as a stability projector/monitor (down‑only scaling to keep \(\rho(J) < 1\)); in mixed regimes (gates/hinges) it remains a conservative allocator. Observability records global and per‑row margins and spend, aligning control‑theory guarantees with practical tuning.

### 4.3 Wormhole Couplings
`GateBenefitCoupling` injects non‑local gradients even for closed connections, solving the zero‑gradient deadlock in sparse topologies. Damped variants provide smoother activation curves. This mechanism generalizes across planning, sequence, and gating tasks as the core “Redemption” pattern (future context corrects earlier decisions).

### 4.4 Precision‑Scaled Orthogonal Noise (PSON)
Null‑space exploration without monotonicity violations. Precision‑aware scaling makes slack variables shoulder exploration while stiff variables take safer, smaller steps. This stabilizes near convergence and accelerates escape from spurious minima.

---

## 5. Algorithms (Composing Fast Matrix Math and Messages)

We retain Dynamic Gradient‑Based Energy Minimization as the default inner solver and augment it with optional kernels per factor family:

- Quadratic/Gaussian blocks: GaBP‑style updates (Jacobi/GS schedule) = precision‑weighted linear solves with per‑iteration cost \(O(\mathrm{nnz}(J))\); no global factorization required.
- Non‑Gaussian/gated/hinge terms: gradient with line search, proximal updates, or ADMM blocks (production‑ready) with acceptance guards.
- Mixed graphs: hybrid passes—GaBP on quadratic stars, prox/gradient on others—under a common stability projector.

**Penalty vs Primal–Dual Clarification:**

> The system is a penalty/augmented‑Lagrangian–style scheme by default (primal variables with adaptive penalty weights). When ADMM mode is enabled, the updates become true primal–dual iterations with explicit multipliers.

**Pseudocode Sketch (Conceptual):**

```python
# One relaxation pass
for block in factorization_order:
    if block.is_quadratic_and_spd():
        # GaBP-style Jacobi/GS update (precision-weighted)
        x_block = solve_local_system(block)  # matvec-based, no global factorization
    elif block.has_closed_form_prox():
        x_block = prox_update(block)
    else:
        x_block = gradient_step(block, preconditioner=diag_precision)
    apply_pson_tangent_noise(x_block)         # orthogonal, precision-scaled
    enforce_small_gain_stability_projection() # keep spectral radius < 1
accept_if_monotone_or_guarded()
```

---

## 6. Observability

We ship relaxation trackers and stability telemetry: per‑step ΔF, acceptance provenance, contraction margins (global and row), and “budget vs spend” for Small‑Gain. This enables black‑box‑free debugging of convergence and stability behavior.

---

## 7. Empirical Guidance and Repro

- Orthogonal vs isotropic noise: compare ΔF histograms and sharpness at matched loss.
- Precision‑aware vs uniform noise scaling: escape events, ΔF90, final energy.
- Small‑Gain vs line‑search‑only vs GradNorm: ΔF90, backtracks, final energy on dense graphs.
- Wormhole ablation: activation/opening rates vs energy drop versus hinge/quadratic baselines.

Note on gaps (tracked): a dedicated script for “PSON vs isotropic vs precision‑orthogonal” ablation is planned; current code ships orthogonal and precision‑orthogonal modes (isotropic baseline to be added for in‑paper table).

**Reproducibility Commands (Windows PowerShell):**

```powershell
# Wormhole demo
uv run python -m experiments.demo_wormhole

# Unit tests (subset)
uv run -m pytest tests -k "gate_benefit or couplings" -v

# Adapter comparison sweep (example)
uv run python -m experiments.benchmark_delta_f90 --configs default gradnorm smallgain --steps 60

# SmallGain validation sweep (see docs/SMALLGAIN_VALIDATION_FINAL.md)
uv run python -m experiments.sweep_smallgain --quick
```

---

## 8. Future Work

- Asynchronous/priority scheduling variants for sparse graphs to improve wall‑clock efficiency (beyond Jacobi/GS): implement prioritized updates and compare against synchronous passes.
- Dedicated ablation script for isotropic vs orthogonal vs precision‑orthogonal noise to populate the empirical table in Section 7.

---

## 9. Limitations

- GaBP equivalence applies to Gaussian/quadratic sub‑problems with SPD precision; mixed regimes require hybrid updates and guards.  
- Walk‑summability/diagonal‑dominance violations can stall/oscillate; Small‑Gain projection mitigates but cannot fix poor modeling.  
- Precision tracking uses diagonal approximations by default; full metrics require SPD and careful conditioning.  
- Wormhole benefit estimation quality affects activation dynamics; use conservative estimates with monotone acceptance.

---

## 10. Related Work

- Equilibrium Propagation (Scellier & Bengio): nudge‑based non‑local gradient in energy‑based models.
- Gaussian BP and walk‑sums (Weiss & Freeman; Malioutov et al.): equivalence to classical linear solvers and convergence conditions.
- Small‑gain/passivity (Zames; Vidyasagar): loop‑gain constraints for stability in nonlinear feedback systems.
- Operator‑splitting/ADMM (Boyd et al.): proximal and primal‑dual updates for composite objectives.

---

## 11. Conclusion

By unifying precision‑aware null‑space exploration (PSON), stability projection (Small‑Gain), and non‑local correction (Wormhole), the Homeostat delivers a fast, controllable, and observable “System‑2” layer for neuro‑symbolic systems. In Gaussian regions, message passing and preconditioned gradient descent collapse to the same sparse matrix iteration, reducing tuning to stability budgeting; in mixed regimes, proximal and ADMM updates preserve efficiency and guarantees. The resulting architecture is both special and fast: matrix‑math heavy inner loops, explicit safety mechanisms, and non‑local credit assignment—all in a modular, typed stack.

---

### References

Needs full FURLAT ABSTRACTIONS links, thanks etc.

1. Weiss, Y., & Freeman, W. T. (2001). Correctness of Belief Propagation in Gaussian Graphical Models of Arbitrary Topology. Neural Computation.  
2. Malioutov, D., Johnson, J. K., & Willsky, A. S. (2006). Walk‑sums and belief propagation in Gaussian graphical models. Journal of Machine Learning Research.  
3. Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM.  
4. Scellier, B., & Bengio, Y. (2017). Equilibrium Propagation: Bridging the Gap Between Energy‑Based Models and Backpropagation. Frontiers in Neuroscience.  
5. Zames, G. (1966). On the input‑output stability of time‑varying nonlinear feedback systems. IEEE TAC.  
6. Vidyasagar, M. (1993). Nonlinear Systems Analysis. Prentice Hall.  
7. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends in Machine Learning.  

---

### Citation

If you use this repository in your research, please cite it as below.

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

### Notes on Implementation

- Codebase: Python, protocol‑based architecture; hot‑swappable NumPy/Torch/JAX backends.  
- Vectorization: Compile‑time graph vectorization cache to amortize sparse passes.  
- Observability: Event‑driven telemetry (RelaxationTracker) for energy descent, stability margins, and adapter spend.  
- Precision Layer: Implemented; per‑coordinate steps and noise scaling via diagonal curvature.
