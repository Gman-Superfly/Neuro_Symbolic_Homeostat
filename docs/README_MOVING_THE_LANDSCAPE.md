# Shaping the objective and update geometry

Status: conceptual overview
Scope: why we use four techniques in the Neuro-Symbolic Homeostat, and how they interact.

The techniques mentioned here are expanded in *Complexity from Constraints*.

---

## 1. Mechanism boundary

The coordinator combines four operations that affect different parts of a relaxation run. CGBC and the optional weight adapter change objective terms. Diagonal preconditioning changes the update geometry. The stability guard limits the step in that geometry. PSON adds a bounded tangent perturbation. Keeping these roles separate is necessary because a guarantee for one operation does not automatically transfer to the others.

For quadratic objectives with an SPD Hessian, the implemented diagonal-preconditioned map has a scoped contraction result when the guard bounds the normalized Hessian with the exact preconditioner used by the update. Mixed and nonconvex objectives retain monotone rejection and, when configured, projected Armijo checks. They do not inherit the quadratic contraction theorem without its assumptions.

---

## 2. The four mechanism groups

We use four focused systems to coordinate the homeostat. Each addresses a specific failure mode of standard gradient descent (GD).

### A. Precision-scaled orthogonal noise (PSON)
**Role**: Tangent exploration / "shaking the box"
**The problem**: A stalled descent path does not test alternative local directions. Standard isotropic noise perturbs all directions equally and can oppose descent.
**The solution**: PSON projects a random draw against the ordinary gradient, applies inverse-precision weights, and projects again because the weighting generally breaks the first orthogonality condition. It normalizes the result and uses one uniform box-feasible scale for the complete vector.
**The effect**: Above the numerical gradient threshold, the final noise vector satisfies \(g^\top\delta=0\), so its first-order energy change is zero. This is a tangent direction, not a Hessian null vector or a flat direction. Its second-order energy change can be positive, and rejected proposals restore the prior state.
*See `docs/README_TANGENT_NOISE_PSON.md` for technical details.*

### B. Curvature-based stability guard and optional allocator
**Role**: Bounded update steps and budgeted coupling adaptation
**The problem**: Strong couplings can create unstable feedback loops and divergence.
**The solution**: The coordinator composes Gershgorin-style curvature bounds and caps the update step. The optional Small-Gain allocator uses remaining estimated margin as a global budget for coupling-family weight increases.
**The effect**: The step cap supports the scoped quadratic contraction condition. Allocator behavior remains an empirical weighting policy with explicit spend telemetry.
*See `docs/STABILITY_GUARANTEES.md`.*

### C. Counterfactual gate-benefit coupling (CGBC)
**Role**: Caller-supplied non-local gate force
**The Problem**: In a gated system (like a logic gate or a mixture of experts), if a gate is closed (weight = 0), gradients can vanish. The local update may not receive the downstream signal that opening the gate would reduce energy.
**The solution**: CGBC, nicknamed a wormhole coupling, inserts a gate force proportional to a caller-supplied downstream benefit estimate. The coordinator converts that value to a finite float and freezes it for the complete solver call.
**The effect**: The frozen estimate can push a gate even when the gate is currently closed. CGBC consumes this estimate; it does not derive counterfactual credit. A wrong-sign estimate pushes the gate in the wrong direction.
*See `docs/README_WORMHOLE.md`.*

### D. Stiffness-based updates (Jacobi form)
**Role**: Stiffness-aware updates
**The Problem**: First-order gradient descent is slow in narrow valleys (poor conditioning). Full Newton steps are expensive for large systems.
**The solution**: The update divides the gradient by a positive diagonal preconditioner. On a quadratic/SPD system with \(P=\operatorname{diag}(H)\), this is weighted Jacobi. It reduces to classical Jacobi when the step factor is one, the epsilon floor is inactive, and box clipping does not alter the proposal. Gaussian BP solves the related Gaussian mean problem with explicit messages but is not implemented here.
**The effect**: The diagonal rescales coordinates without constructing or inverting the full Hessian. Whether that rescaling reduces iteration count is an empirical question outside the contraction theorem.
*See `docs/README_GABP_EQUIVALENCE.md`.*

---

## 3. Composition and evidence boundary

On their own, these methods have limitations:
- **PSON** is just noise; it doesn't guide you to the goal.
- **Small-Gain allocator** changes coupling-family weights under a predicted global curvature-spend budget; its task benefit remains empirical.
- **CGBC/wormhole terms** are caller-driven forces; they can pull gates in wrong directions if the benefit estimate is wrong.
- **Stiffness/Jacobi updates** apply directly to quadratic systems and act as diagonal curvature scaling in mixed regimes; they do not implement Gaussian BP.

**Combined, they provide:**
1.  **CGBC/wormhole terms** provide non-local gate gradients that can increase gate values when the benefit estimate has the right sign.
2.  **Stiffness scaling** rescales conditioned quadratic updates once paths are active.
3.  **The step cap** enforces the normalized-Hessian condition on quadratic/SPD updates when every curvature report covers the realized segment. Projected Armijo checks a sufficient-decrease fallback, and accepted-step rejection restores uphill proposals.
4.  **PSON** supplies bounded tangent exploration. The current experiments show lower exact synthetic full-Hessian noise cost in seven generated families and controlled escape in one designed anisotropic case.

The combined system can adapt effective coupling weights, apply geometry-matched guards, and test noisy proposals. The repository has mechanism-level synthetic evidence. It does not yet establish a task-level advantage for the combined system.

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
- Implementing the weighted-Jacobi form directly as stiffness-scaled updates allows vectorized NumPy kernels in supported paths.
- Sparse row-sum checks are \(O(\lvert V\rvert+\lvert E\rvert)\), rather than the dense \(O(\lvert V\rvert^3)\) cost of a full eigenvalue decomposition.

These local calculations let the coordinator consume caller-supplied gate signals and diagonal curvature without adding Gaussian message objects or a dense Hessian solve.

---

## Summary

The repository composes four distinct operations:
1.  Shape selected objective terms with frozen CGBC inputs and optional weight adaptation.
2.  Scale updates by a positive diagonal preconditioner, giving weighted Jacobi under the stated quadratic conditions.
3.  Cap the step using the curvature bound for the executed geometry, with projected Armijo available for uncovered curvature.
4.  Add re-projected, uniformly box-scaled tangent perturbations through PSON.

Each formal or empirical claim applies only to the corresponding operation and stated assumptions.
