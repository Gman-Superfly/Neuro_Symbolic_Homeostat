# Quadratic stiffness updates, Jacobi, and Gaussian BP

Status: synchronous Jacobi is implemented and trajectory-tested; Gaussian BP is literature context.
Scope: exact local algebra for the implemented stiffness update and the boundary of the message-passing analogy.

---

## 1. The implemented claim

For a quadratic energy with SPD precision, the coordinator's stiffness update is exactly Jacobi:

\[
x^{(t+1)} = x^{(t)} - D^{-1}(Jx^{(t)}-h).
\]

Gaussian belief propagation addresses the same Gaussian mean problem through edge messages and cavity precisions. When GaBP converges, its mean solves the same system $Jx=h$. General GaBP is not a stationary Jacobi iteration, and this repository does not implement or test GaBP message updates. Sequential Gauss-Seidel remains a future scheduler target.

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

This update uses the previous iterate for every neighbor and therefore has Jacobi scheduling semantics.

### 2.2 Gauss-Seidel iteration (sequential)

The GS update solves for \(x_i\) using *new* values for neighbors \(j < i\) and *old* values for \(j > i\):

\[
(D + L) x^{(t+1)} = h - U x^{(t)}
\]
\[
x^{(t+1)} = (D + L)^{-1} (h - U x^{(t)})
\]

This corresponds to updating variables one by one in order, immediately using the fresh values for the next variable.

This is a sequential stiffness-scaled linear solve. The current coordinator does not expose it as a dedicated mode.

---

## 3. Why this matters for the Homeostat

1. Efficiency: the synchronous path needs the diagonal stiffness \(D\), stored in `_precision_cache`, and the gradient \(\nabla F\).
2. Vectorization: the update \(x \leftarrow x - \nabla F / \text{diag}(J)\) maps directly to array operations.
3. Stability: Jacobi convergence requires \(\rho(I - D^{-1}J) < 1\). The repository also applies a separate Gershgorin Lipschitz step cap for gradient iterations, which enforces \(\rho(I-\alpha H)<1\) in quadratic/SPD cases when the bound is valid.

---

## 4. References

1.  **Weiss, Y., & Freeman, W. T. (2001).** *Correctness of Belief Propagation in Gaussian Graphical Models of Arbitrary Topology.* Neural Computation.
    *   **Local use**: correctness properties for Gaussian BP means when the message procedure converges. It does not prove that the coordinator implements GaBP.
2.  **Malioutov, D., et al. (2006).** *Walk-sums and belief propagation in Gaussian graphical models.* JMLR.
    *   **Local use**: walk-sum interpretation and sufficient convergence conditions for Gaussian BP. These conditions are related to, but distinct from, the tested Jacobi condition.
3.  **Saad, Y. (2003).** *Iterative Methods for Sparse Linear Systems.* SIAM.
    *   **Local use**: Jacobi and Gauss-Seidel convergence properties for linear systems.

---

## 5. In the code

- Where: `core/coordinator.py` inside `relax_etas`.
- Flag: `use_stiffness_updates=True`.
- Implemented logic:
  ```python
  # Exact Jacobi form for a quadratic system
  diag_stiffness = self.get_precision_diagonal() # D
  grad = self._grads(etas)                       # Jx - h
  step = grad / diag_stiffness                   # D^-1 (Jx - h)
  etas -= step
  ```
