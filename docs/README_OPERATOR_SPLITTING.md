# Proximal Operator Splitting and ADMM

Status: Experimental / Advanced Feature  
Scope: Solving energy minimization via proximal operators instead of gradient steps.

---

## 1. Overview

For energy terms that are non-smooth or have hard constraints (e.g., hinges, L1 norms, box constraints), gradient descent can be unstable or inaccurate. Proximal algorithms handle these by solving small implicit sub-problems exactly.

The coordinator supports two modes:
1.  **Proximal-Only** (`operator_splitting=True`): Iterative application of proximal operators.
2.  **ADMM** (`use_admm=True`): Alternating Direction Method of Multipliers, introducing dual variables (multipliers) for strict constraint enforcement.

---

## 2. Why Use It?

- **Exact Constraint Handling**: Proximal steps project exactly onto the constraint manifold (e.g., \(\eta \in [0,1]\), hinge satisfied).
- **Stability**: Implicit steps are unconditionally stable for convex terms, allowing larger step sizes than explicit gradient descent.
- **Handling Non-Differentiable Terms**: L1 sparsity, sharp hinges, or discrete-like penalties work naturally without smoothing.

---

## 3. Proximal Operators Implemented

Located in `core/prox_utils.py`:

### 3.1 `prox_quadratic_pair`
Solves the proximal operator for \(w(\eta_i - \eta_j)^2\).
- Used for: `QuadraticCoupling`.
- Effect: Pulls \(\eta_i, \eta_j\) closer together analytically.

### 3.2 `prox_asym_hinge_pair`
Solves the proximal operator for \(w \max(0, \beta \eta_j - \alpha \eta_i)^2\).
- Used for: `DirectedHingeCoupling`, `AsymmetricHingeCoupling`.
- Effect: Enforces the hinge constraint exactly if violated; does nothing if satisfied.

### 3.3 `prox_linear_gate`
Solves the proximal operator for linear terms like \(c \cdot \eta\).
- Used for: `GateBenefitCoupling` (in ADMM mode with `admm_gate_prox=True`).
- Effect: Shifts \(\eta\) by the benefit gradient, respecting bounds.

---

## 4. ADMM Implementation

When `use_admm=True`, the `relax_etas_admm` loop runs:

1.  **s-update**: Update auxiliary splitting variables based on couplings.
2.  **η-update (Primal)**: Update order parameters using local gradients + coupling forces + Lagrange multipliers.
3.  **u-update (Dual)**: Update Lagrange multipliers based on constraint violation (residuals).

This effectively turns the "soft" springs of the physics simulation into "hard" constraints over time as multipliers grow.

---

## 5. Usage

### Proximal-Only Mode
```python
coord = EnergyCoordinator(
    ...,
    operator_splitting=True,
    prox_steps=50,
    prox_tau=0.05,  # Step size parameter
)
```

### ADMM Mode
```python
coord = EnergyCoordinator(
    ...,
    use_admm=True,
    admm_rho=1.0,       # Penalty parameter (stiffness)
    admm_steps=50,
    admm_gate_prox=True # Use prox for gates
)
```

---

## 6. Verification

See `experiments/demo_operator_splitting.py` for a runnable example of proximal relaxation on a small graph with hinges and springs.

