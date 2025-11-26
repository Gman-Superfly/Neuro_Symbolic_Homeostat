# Equivalence: GaBP ↔ Linear Solvers (Jacobi/Gauss-Seidel)

Status: Theoretical Foundation  
Scope: Algebra and proofs linking Message Passing to Gradient Descent/Iterative Linear Solvers.

---

## 1. The Claim

For **quadratic energies** with **positive-definite precision** (SPD stiffness), the following algorithms perform the **exact same computation**:

1.  **Gaussian Belief Propagation (GaBP)** for means (message passing).
2.  **Iterative Linear Solvers**:
    *   Synchronous schedule = **Jacobi Method**.
    *   Sequential schedule = **Gauss-Seidel (GS) Method**.
3.  **Preconditioned Gradient Descent**:
    *   Diagonal preconditioning = Jacobi.
    *   Triangular preconditioning = Gauss-Seidel.

This equivalence allows us to implement "message passing" efficiently using **fast matrix-vector operations** (vectorized stiffness updates) without explicit message objects.

---

## 2. The Algebra

Consider a quadratic energy function (Gaussian random field):

\[
F(x) = \frac{1}{2} x^\top J x - h^\top x
\]

Where:
- \(x\): Vector of variables (order parameters).
- \(J\): Precision matrix (Hessian / Stiffness). \(J_{ii} > 0\) is diagonal stiffness; \(J_{ij}\) is coupling strength.
- \(h\): Potential vector (linear bias / input).

Minimizing \(F(x)\) is equivalent to solving the linear system:
\[
J x = h
\]

Decompose \(J\) into **Diagonal (D)**, **Strictly Lower (L)**, and **Strictly Upper (U)** parts:
\[ J = D + L + U \]

### 2.1 Jacobi Iteration (Synchronous)

The Jacobi update solves for \(x_i\) assuming neighbors are fixed at the *previous* step \(t\):

\[
D x^{(t+1)} = h - (L + U) x^{(t)}
\]
\[
x^{(t+1)} = D^{-1} (h - (J - D) x^{(t)}) = x^{(t)} - D^{-1} (J x^{(t)} - h)
\]

In gradient terms (since \(\nabla F = Jx - h\)):
\[
x^{(t+1)} = x^{(t)} - D^{-1} \nabla F(x^{(t)})
\]

**This is Gradient Descent with Inverse-Diagonal Stiffness Preconditioning.**

In GaBP terms: A node computes its new mean by summing incoming messages (forces) from neighbors at time \(t\) and dividing by its total precision (stiffness).

### 2.2 Gauss-Seidel Iteration (Sequential)

The GS update solves for \(x_i\) using *new* values for neighbors \(j < i\) and *old* values for \(j > i\):

\[
(D + L) x^{(t+1)} = h - U x^{(t)}
\]
\[
x^{(t+1)} = (D + L)^{-1} (h - U x^{(t)})
\]

This corresponds to updating variables one by one in order, immediately using the fresh values for the next variable.

**This is Coordinate Descent with Stiffness Scaling.**

In GaBP terms: Messages are passed sequentially; information flows faster in the direction of the update order.

---

## 3. Why This Matters for the Homeostat

1.  **Efficiency**: We don't need graph pointers or message objects. We just need the diagonal stiffness \(D\) (which we aggregate in `_precision_cache`) and the gradient \(\nabla F\).
2.  **Speed**: The update \(x \leftarrow x - \nabla F / \text{diag}(J)\) is fully vectorizable (Jacobi).
3.  **Stability**: Convergence is guaranteed if the spectral radius \(\rho(I - D^{-1}J) < 1\). This is exactly the condition enforced by our **Small-Gain Stability Projector** (diagonal dominance / Gershgorin bounds).

---

## 4. References

1.  **Weiss, Y., & Freeman, W. T. (2001).** *Correctness of Belief Propagation in Gaussian Graphical Models of Arbitrary Topology.* Neural Computation.
    *   **Key Result**: Proves GaBP means converge to the true means for any topology where the walk-sum series converges (walk-summability).
2.  **Malioutov, D., et al. (2006).** *Walk-sums and belief propagation in Gaussian graphical models.* JMLR.
    *   **Key Result**: Links walk-summability directly to the spectral radius condition of linear solvers.
3.  **Saad, Y. (2003).** *Iterative Methods for Sparse Linear Systems.* SIAM.
    *   **Key Result**: Standard text on Jacobi/GS convergence properties.

---

## 5. In Our Code

- **Where**: `core/coordinator.py` inside `relax_etas`.
- **Flag**: `use_stiffness_updates=True`.
- **Logic**:
  ```python
  # Equivalent to Jacobi / GaBP-Synchronous
  diag_stiffness = self.get_precision_diagonal() # D
  grad = self._grads(etas)                       # Jx - h
  step = grad / diag_stiffness                   # D^-1 (Jx - h)
  etas -= step
  ```

