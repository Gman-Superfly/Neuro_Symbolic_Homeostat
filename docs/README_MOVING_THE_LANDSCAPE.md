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

### B. Curvature-based stability guard and optional allocator
**Role**: Bounded update steps and budgeted coupling adaptation
**The problem**: Strong couplings can create unstable feedback loops and divergence.
**The solution**: The coordinator composes Gershgorin-style curvature bounds and caps the update step. The optional Small-Gain allocator uses remaining estimated margin as a global budget for coupling-family weight increases.
**The effect**: The step cap supports the scoped quadratic contraction condition. Allocator behavior remains an empirical weighting policy with explicit spend telemetry.
*See `docs/STABILITY_GUARANTEES.md`.*

### C. Counterfactual gate-benefit coupling (CGBC)
**Role**: Non-local credit assignment
**The Problem**: In a gated system (like a logic gate or a mixture of experts), if a gate is closed (weight = 0), gradients can vanish. The local update may not receive the downstream signal that opening the gate would reduce energy.
**The Solution**: We add a counterfactual gate-benefit coupling, nicknamed a wormhole coupling, that links the *potential* downstream benefit directly to the gate control. This force is independent of the current gate state.
**The effect**: A supplied downstream estimate can push gate updates even when the gate is currently closed.
*See `docs/README_WORMHOLE.md`.*

### D. Stiffness-based updates (Jacobi form)
**Role**: Stiffness-aware updates
**The Problem**: First-order gradient descent is slow in narrow valleys (poor conditioning). Full Newton steps are expensive for large systems.
**The Solution**: We use diagonal curvature to implement the Jacobi form exactly on quadratic/SPD systems. Gaussian BP solves the related Gaussian mean problem with explicit messages but is not implemented here.
**The effect**: By tracking diagonal precision, we approximate Hessian diagonals and improve quadratic sub-problem updates using vectorized operations.
*See `docs/README_GABP_EQUIVALENCE.md`.*

---

## 3. The synergy: why it works

On their own, these methods have limitations:
- **PSON** is just noise; it doesn't guide you to the goal.
- **Small-Gain allocator** changes coupling-family weights under a predicted global curvature-spend budget; its task benefit remains empirical.
- **CGBC/wormhole terms** are caller-driven forces; they can pull gates in wrong directions if the benefit estimate is wrong.
- **Stiffness/Jacobi updates** apply directly to quadratic systems and act as diagonal curvature scaling in mixed regimes; they do not implement Gaussian BP.

**Combined, they provide:**
1.  **CGBC/wormhole terms** provide non-local gate gradients that can increase gate values when the benefit estimate has the right sign.
2.  **Stiffness scaling** rescales conditioned quadratic updates once paths are active.
3.  **The step cap and accepted-step guard** keep proposals within the repository's scoped stability contract.
4.  **PSON** reduces stall risk in shallow minima while updates continue.

The result is a system that can adapt effective coupling weights, keep explicit guards active, and continue optimization under mixed dynamics.

---

## 4. Related future directions

The broader `complexity-from-constraints` work considers two related directions that are not implemented in this repository:

- **Inverse kinematics**: Instead of directly computing joint angles, we define energy constraints on end-effectors. In this formulation, relaxation over the kinematic chain drives the system toward a feasible pose.
- **Homotopy (continuation)**: We slowly introduce complex constraints into the objective. Starting from an easier objective and morphing toward the harder one reduces early trapping in poor local minima.

These directions require separate implementations and validation. The active code in this repository covers the four mechanism groups above.

---

## 5. Local equivalents

A key design choice is using local algebraic equivalents where the assumptions hold.
- Implementing full Newton steps is expensive. Implementing GaBP with message objects has Python overhead.
- Implementing the Jacobi form directly as stiffness-scaled updates allows us to use vectorized NumPy kernels in supported paths.
- Small-Gain checks are O(N) local sums, not O(N³) eigenvalue decompositions.

By choosing techniques that have local equivalents, we use global signals (via CGBC/wormhole terms and stiffness propagation) while keeping the inner loops local.

---

## Summary

We do more than execute plain gradient descent. We:
1.  Shape the objective (CGBC/wormhole terms, Small-Gain).
2.  Scale updates by diagonal stiffness (Jacobi form on quadratic systems).
3.  Add tangent perturbations (PSON).

This is the objective-shaping strategy used by this repository.
