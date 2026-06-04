# Moving the energy function: a unified strategy

Status: conceptual overview
Scope: why we use four techniques in the Neuro-Symbolic Homeostat, and how they interact.

The techniques mentioned here are expanded in *Complexity from Constraints*.

---

## 1. The philosophy: don’t just descend, shape the objective

Standard optimization asks: *"Given a fixed energy function, how do I find the minimum?"*
This assumes the objective is static and the only tool is descent. In complex logical problems, which are often non-convex, discrete, or flat, descent can stall in local minima or plateaus.

**Our approach asks: "How can I modify the objective and updates so optimization keeps useful gradients and explicit step bounds?"**

We view inference as an active process that combines four mechanisms. We do not rely on one optimizer. In quadratic/SPD blocks, algebraic equivalences let us replace some global operations with local updates.

---

## 2. The four mechanism groups

We use four focused systems to coordinate the homeostat. Each addresses a specific failure mode of standard gradient descent (GD).

### A. Precision-scaled orthogonal noise (PSON)
**Role**: Tangent exploration / "shaking the box"
**The problem**: To escape local traps, you need exploration noise. Standard isotropic noise perturbs all directions equally and can oppose descent.
**The Solution**: We inject noise orthogonal to the descent direction to remove the first-order energy change. We then scale this noise by inverse precision (curvature). Stiff directions get little noise; slack directions get more.
**The effect**: The system explores flat null-space directions while removing the first-order uphill component of the perturbation.
*See `docs/README_TANGENT_NOISE_PSON.md` for technical details.*

### B. Small-Gain stability (the projector)
**Role**: Bounded dynamics / "the guard rail"
**The problem**: Strong couplings can create unstable feedback loops and divergence.
**The solution**: We enforce Gershgorin-based conservative bounds on couplings. In linear and SPD settings, these bounds support contraction conditions.
**The effect**: We can adjust coupling strengths with explicit step margins and runtime checks.
*See `docs/STABILITY_GUARANTEES.md`.*

### C. Counterfactual gate-benefit coupling (CGBC)
**Role**: Non-local credit assignment
**The Problem**: In a gated system (like a logic gate or a mixture of experts), if a gate is closed (weight = 0), gradients can vanish. The local update may not receive the downstream signal that opening the gate would reduce energy.
**The Solution**: We add a counterfactual gate-benefit coupling, nicknamed a wormhole coupling, that links the *potential* downstream benefit directly to the gate control. This force is independent of the current gate state.
**The effect**: A supplied downstream estimate can push gate updates even when the gate is currently closed.
*See `docs/README_WORMHOLE.md`.*

### D. Stiffness-based updates (GaBP equivalence)
**Role**: Stiffness-aware updates
**The Problem**: First-order gradient descent is slow in narrow valleys (poor conditioning). Full Newton steps are expensive for large systems.
**The Solution**: We use the algebraic equivalence between **Gaussian Belief Propagation (GaBP)** and **Jacobi/Gauss-Seidel** linear solvers in quadratic/SPD blocks.
**The effect**: By tracking diagonal precision, we approximate Hessian diagonals and improve quadratic sub-problem updates using vectorized operations.
*See `docs/README_GABP_EQUIVALENCE.md`.*

---

## 3. The synergy: why it works

On their own, these methods have limitations:
- **PSON** is just noise; it doesn't guide you to the goal.
- **Small-Gain** restricts expressivity; it bounds updates but does not choose the objective.
- **CGBC/wormhole terms** are caller-driven forces; they can pull gates in wrong directions if the benefit estimate is wrong.
- **GaBP** matches classical linear-solver updates for quadratic/SPD blocks under its assumptions; it does not cover non-convex gates.

**Combined, they provide:**
1.  **CGBC/wormhole terms** provide non-local gate gradients that can increase gate values when the benefit estimate has the right sign.
2.  **GaBP/Stiffness** rescales conditioned quadratic updates once paths are active.
3.  **Small-Gain** keeps updates within conservative stability bounds during reconfiguration.
4.  **PSON** reduces stall risk in shallow minima while updates continue.

The result is a system that can change coupling structure, keep explicit guards active, and continue optimization under mixed dynamics.

---

## 4. Inverse kinematics and homotopy

This objective-shaping philosophy extends to broader control problems in `complexity-from-constraints`:

- **Inverse kinematics**: Instead of directly computing joint angles, we define energy constraints on end-effectors. In this formulation, relaxation over the kinematic chain drives the system toward a feasible pose.
- **Homotopy (continuation)**: We slowly introduce complex constraints into the objective. Starting from an easier objective and morphing toward the harder one reduces early trapping in poor local minima.

While this demo repository focuses on core coordination, these techniques follow the same entity-first, energy-based principles and share the same practical pattern: change the objective shape, keep updates bounded, and preserve observability.

---

## 5. Local equivalents

A key design choice is using local algebraic equivalents where the assumptions hold.
- Implementing full Newton steps is expensive. Implementing GaBP with message objects has Python overhead.
- Implementing the *algebraic equivalent* (stiffness-scaled updates) allows us to use vectorized NumPy kernels.
- Small-Gain checks are O(N) local sums, not O(N³) eigenvalue decompositions.

By choosing techniques that have local equivalents, we use global signals (via CGBC/wormhole terms and stiffness propagation) while keeping the inner loops local.

---

## Summary

We do more than execute plain gradient descent. We:
1.  Shape the objective (CGBC/wormhole terms, Small-Gain).
2.  Scale updates by stiffness (Stiffness/GaBP).
3.  Add tangent perturbations (PSON).

This is the objective-shaping strategy used by this repository.
