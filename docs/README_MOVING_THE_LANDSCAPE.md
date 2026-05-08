# Moving the energy function: a unified strategy

Status: conceptual overview
Scope: why we use four techniques in the Neuro-Symbolic Homeostat, and how they interact.

The techniques mentioned here are expanded in *Complexity from Constraints*.

---

## 1. The philosophy: don’t just descend, shape

Standard optimization asks: *"Given a fixed energy function, how do I find the minimum?"*
This assumes the objective is static and the only tool is descent. In complex logical problems, which are often non-convex, discrete, or flat, descent can stall in local minima or plateaus.

**Our approach asks: "How can I modify the objective and updates so optimization remains stable and informative?"**

We view inference as an active process that combines four mechanisms. We do not rely on one optimizer. By using algebraic equivalences, we replace expensive global operations with faster local updates.

---

## 2. The four mechanism groups

We use four focused systems to coordinate the homeostat. Each solves a specific failure mode of standard gradient descent (GD).

### A. Precision-scaled orthogonal noise (PSON)
**Role**: Safe Exploration / "Shaking the Box"
**The problem**: To escape local traps, you need exploration noise. Standard isotropic noise perturbs all directions equally and can oppose descent.
**The Solution**: We inject noise strictly *orthogonal* to the descent direction (tangent noise). Furthermore, we scale this noise by the inverse precision (curvature). Directions that are "stiff" (high curvature, certain) get little noise; directions that are "slack" (flat, uncertain) get more.
**The effect**: The system explores flat null-space directions while keeping descent behavior more stable.
*See `docs/README_TANGENT_NOISE_PSON.md` for technical details.*

### B. Small-Gain stability (the projector)
**Role**: Bounded dynamics / "the guard rail"
**The problem**: Strong couplings can create unstable feedback loops and divergence.
**The solution**: We enforce Gershgorin-based conservative bounds on couplings. In linear and SPD settings, these bounds support contraction conditions.
**The effect**: We can adjust coupling strengths with explicit safety margins and runtime checks.
*See `docs/STABILITY_GUARANTEES.md`.*

### C. Counterfactual gate-benefit coupling (CGBC)
**Role**: Non-local credit assignment
**The Problem**: In a gated system (like a logic gate or a mixture of experts), if a gate is closed (weight = 0), gradients vanish. The system cannot "see" that opening the gate would reduce energy downstream. This is the classic "vanishing gradient" problem in sparse structures.
**The Solution**: We add a counterfactual gate-benefit coupling, nicknamed a wormhole coupling, that links the *potential* downstream benefit directly to the gate control. This force is independent of the current gate state.
**The effect**: A supplied downstream estimate can push gate updates even when the gate is currently closed.
*See `docs/README_WORMHOLE.md`.*

### D. Stiffness-based updates (GaBP equivalence)
**Role**: Fast Convergence / "The Accelerator"
**The Problem**: First-order gradient descent is slow in narrow valleys (poor conditioning). Second-order methods (Newton's method) fix this but are computationally prohibitive (O(N³)) for large systems.
**The Solution**: We exploit the algebraic equivalence between **Gaussian Belief Propagation (GaBP)** and **Jacobi/Gauss-Seidel** linear solvers.
**The effect**: By tracking diagonal precision, we approximate Hessian diagonals and speed up quadratic sub-problem updates using vectorized operations.
*See `docs/README_GABP_EQUIVALENCE.md`.*

---

## 3. The synergy: why it works

On their own, these methods have limitations:
- **PSON** is just noise; it doesn't guide you to the goal.
- **Small-Gain** restricts expressivity; it keeps you safe but doesn't solve the problem.
- **CGBC/wormhole terms** are heuristic forces; they can pull gates in wrong directions if the benefit estimate is wrong.
- **GaBP** only solves quadratics exactly; it fails on non-convex gates.

**Combined, they provide:**
1.  **CGBC/wormhole terms** create the necessary global gradients to open new paths (solving the "discrete search" problem).
2.  **GaBP/Stiffness** rapidly solves the resulting flow problems once paths are open (solving the "slow convergence" problem).
3.  **Small-Gain** keeps updates within conservative stability bounds during reconfiguration.
4.  **PSON** reduces stall risk in shallow minima while updates continue.

The result is a system that can change coupling structure, preserve safety guards, and continue optimization under mixed dynamics.

---

## 4. Inverse kinematics and homotopy

This objective-shaping philosophy extends to broader control problems in `complexity-from-constraints`:

- **Inverse kinematics**: Instead of solving for joint angles directly, we define energy constraints on end-effectors. In this formulation, relaxation over the kinematic chain drives the system toward a feasible pose.
- **Homotopy (continuation)**: We slowly introduce complex constraints into the objective. Starting from an easier objective and morphing toward the harder one reduces early trapping in poor local minima.

While this demo repository focuses on core coordination, these techniques follow the same entity-first, energy-based principles and share the same practical pattern: change the objective shape, keep updates bounded, and preserve observability.

---

## 5. Speed and equivalences

A key design choice is **speed via equivalence**.
- Implementing full Newton steps is slow. Implementing GaBP with message objects is slow (in Python).
- Implementing the *algebraic equivalent* (stiffness-scaled updates) allows us to use vectorized NumPy kernels.
- Small-Gain checks are O(N) local sums, not O(N³) eigenvalue decompositions.

By choosing techniques that have fast, local equivalents, we build a system that uses global signals (via CGBC/wormhole terms and stiffness propagation) while computing locally and rapidly.

---

## Summary

We don't just execute Gradient Descent. We:
1.  **Shape** the manifold (CGBC/wormhole terms, Small-Gain).
2.  **Accelerate** the flow (Stiffness/GaBP).
3.  **Shake** the state (PSON).

This is the objective-shaping strategy used by this repository.
