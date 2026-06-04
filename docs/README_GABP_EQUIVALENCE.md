# Equivalence: GaBP ↔ Linear solvers (Jacobi/Gauss-Seidel)

Status: theoretical foundation, with synchronous Jacobi implemented through stiffness updates.
Scope: algebra and proofs linking message passing to gradient descent and iterative linear solvers.

---

## 1. The claim

For quadratic energies with positive-definite precision (SPD stiffness), the following algorithms perform the same computation under the stated scheduling assumptions:

1.  **Gaussian Belief Propagation (GaBP)** for means (message passing).
2.  **Iterative Linear Solvers**:
    *   Synchronous schedule = **Jacobi Method**.
    *   Sequential schedule = **Gauss-Seidel (GS) Method**.
3.  **Preconditioned Gradient Descent**:
    *   Diagonal preconditioning = Jacobi.
    *   Triangular preconditioning = Gauss-Seidel.

This equivalence lets the repository implement the synchronous Jacobi form with vectorized stiffness updates and no explicit message objects. Sequential Gauss-Seidel remains a theoretical reference and future scheduler target in this version.

---

## 2. The algebra

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

### 2.1 Jacobi iteration (synchronous)

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

This is gradient descent with inverse-diagonal stiffness preconditioning.

In GaBP terms: A node computes its new mean by summing incoming messages (forces) from neighbors at time \(t\) and dividing by its total precision (stiffness).

### 2.2 Gauss-Seidel iteration (sequential)

The GS update solves for \(x_i\) using *new* values for neighbors \(j < i\) and *old* values for \(j > i\):

\[
(D + L) x^{(t+1)} = h - U x^{(t)}
\]
\[
x^{(t+1)} = (D + L)^{-1} (h - U x^{(t)})
\]

This corresponds to updating variables one by one in order, immediately using the fresh values for the next variable.

This is a sequential stiffness-scaled linear solve. The current coordinator has coordinate-descent utilities, but it does not yet expose this Gauss-Seidel stiffness schedule as a dedicated mode.

In GaBP terms: messages are passed sequentially, so the update order changes how information propagates.

---

## 3. Why this matters for the Homeostat

1. Efficiency: the synchronous path needs the diagonal stiffness \(D\), stored in `_precision_cache`, and the gradient \(\nabla F\).
2. Vectorization: the update \(x \leftarrow x - \nabla F / \text{diag}(J)\) maps directly to array operations.
3. Stability: Jacobi convergence requires \(\rho(I - D^{-1}J) < 1\). The repository also applies a separate Gershgorin Lipschitz step cap for gradient iterations, which enforces \(\rho(I-\alpha H)<1\) in quadratic/SPD cases when the bound is valid.

---

## 4. References

1.  **Weiss, Y., & Freeman, W. T. (2001).** *Correctness of Belief Propagation in Gaussian Graphical Models of Arbitrary Topology.* Neural Computation.
    *   **Key Result**: Proves GaBP means converge to the true means for any topology where the walk-sum series converges (walk-summability).
2.  **Malioutov, D., et al. (2006).** *Walk-sums and belief propagation in Gaussian graphical models.* JMLR.
    *   **Key Result**: Links walk-summability directly to the spectral radius condition of linear solvers.
3.  **Saad, Y. (2003).** *Iterative Methods for Sparse Linear Systems.* SIAM.
    *   **Key Result**: Standard text on Jacobi/GS convergence properties.

---

## 5. In the code

- Where: `core/coordinator.py` inside `relax_etas`.
- Flag: `use_stiffness_updates=True`.
- Implemented logic:
  ```python
  # Equivalent to Jacobi / GaBP-Synchronous
  diag_stiffness = self.get_precision_diagonal() # D
  grad = self._grads(etas)                       # Jx - h
  step = grad / diag_stiffness                   # D^-1 (Jx - h)
  etas -= step
  ```

