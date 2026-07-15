# Bilinear Couplings in the Neuro‑Symbolic Homeostat

Status: Design document (optional feature; not enabled by default, you will have to implement it)
Intent: Explain what a bilinear coupling adds, its impact on stability/solver choices, and a conservative path to implement behind a flag.

---

## 1. Motivation

Existing couplings cover:
- Quadratic springs: encourage equality (η_i − η_j → 0)
- Hinge variants: one‑sided activation/gaps
- CGBC/wormhole: linear non-local gate force

These do not encode multiplicative synergy/competition directly. A bilinear term provides that:

- Expressivity: reward or penalize co‑activation with a single edge
- Compactness: AND‑like “only together” effects without auxiliary variables

---

## 2. Definition

We consider a symmetric bilinear energy between two order parameters η_i, η_j ∈ [0, 1]:

\[
E_{ij}^{\text{bilinear}} = w_{ij}\, \eta_i \eta_j
\]

- w_{ij} > 0: co‑activation penalty (competition)
- w_{ij} < 0: co‑activation reward (synergy)

Gradients:

\[
\frac{\partial E}{\partial \eta_i} = w_{ij}\, \eta_j,
\qquad
\frac{\partial E}{\partial \eta_j} = w_{ij}\, \eta_i.
\]

Hessian (2×2 block for i,j):

\[
H_{ij}^{\text{bilinear}} =
\begin{bmatrix}
0 & w_{ij} \\
w_{ij} & 0
\end{bmatrix}
\quad\Rightarrow\quad
\lambda \in \{+|w_{ij}|, -|w_{ij}|\}.
\]

Implication: the pure bilinear piece is indefinite (zero diagonal curvature, non‑zero off‑diagonals). Overall SPD requires sufficient diagonal curvature from locals and other convex terms.

---

## 3. Curvature accounting and stability

For an ordinary gradient step, let \(L_H\) denote the Gershgorin row-sum bound on the raw Hessian. For a bilinear edge (i, j):

- It contributes |w_{ij}| to each row’s off‑diagonal sum.
- It does not add to diagonal curvature by itself.

For a diagonal-preconditioned step, the relevant matrix is \(P^{-1/2}HP^{-1/2}\), using the exact positive diagonal \(P\) that divides the gradient. The same edge then contributes \(|w_{ij}|/\sqrt{p_i p_j}\) to each endpoint's normalized off-diagonal row sum. The fixed-step condition is based on the resulting \(L_P\), not on raw \(L_H\).

Therefore:
- Diagonal dominance can be lost more easily as |w_{ij}| grows.
- The raw curvature estimate must include |w_{ij}| in both endpoint row sums, and the preconditioned estimate must include its normalized value.
- A contraction statement additionally requires the full local Hessian to be positive definite. An absolute row-sum bound alone does not make an indefinite objective convex.

Recommended guardrails:
- Report `(0, 0, abs(w_ij))` through `SupportsCouplingCurvature` and retain the geometry-matched step cap.
- Require enough positive diagonal curvature to establish the intended local SPD regime.
- If adaptive coupling weights are under study, account for |w_{ij}| as spend in the optional Small-Gain allocator.
- Keep monotone acceptance (reject steps with ΔF > 0).

---

## 4. Solver Integration

### 4.1 Per‑Coordinate Stiffness Update

The stiffness path constructs \(p_i=\max(\varepsilon,\Lambda_{ii})\) and applies \(\Delta\eta_i=-\alpha g_i/p_i\), where \(\Lambda\) is the nonnegative diagonal curvature cache. Pure bilinear curvature belongs only in the off-diagonal bound and adds zero to \(\Lambda_{ii}\). The guard must therefore normalize the bilinear edge with the epsilon-floored \(P\) used by the update. A pure bilinear objective is indefinite, so the SPD contraction theorem does not apply even when the row-sum cap is finite. Monotone rejection can restore an uphill proposal, but it cannot turn the indefinite objective into a contractive quadratic map.

### 4.2 Gauss–Seidel / 2×2 Local Block

For variables i and j with diagonal curvatures \(a_i, a_j \ge 0\) (from locals/other convex terms) and a bilinear cross‑term \(w\), a local Newton step solves the 2×2 system:

\[
\begin{bmatrix}
a_i & w \\
w & a_j
\end{bmatrix}
\begin{bmatrix}
\Delta \eta_i \\
\Delta \eta_j
\end{bmatrix}
=
-\begin{bmatrix}
g_i \\
g_j
\end{bmatrix}.
\]

This 2×2 solve has a positive-curvature Newton interpretation only when the block is SPD. A sufficient two-variable condition is \(a_i>0\), \(a_j>0\), and \(a_i a_j>w^2\). A singular or indefinite block does not provide that stabilization and requires separate regularization or solver analysis. This block solver remains a design option; the current coordinator does not implement it.

### 4.3 Practical Recipe
- Require (or induce) modest diagonal curvature \(a_i, a_j > 0\) where bilinear edges exist (from locals or a tiny diagonal regularizer).
- Prefer a separately validated GS, priority, or SPD block method in sparse graphs; compare it against the implemented synchronous weighted-Jacobi path.
- Keep the curvature-based step cap and monotone acceptance active. Use the Small-Gain allocator only when adaptive coupling weights are part of the experiment.

---

## 5. PSON, CGBC/wormhole, and mixed regimes

- PSON: Keep pure bilinear curvature out of the diagonal precision cache. Project the draw, apply inverse-precision weights, project again, normalize, and use one uniform box-feasible scale. The second projection restores first-order tangency after weighting.
- CGBC/wormhole: Plain CGBC and damped power \(p=1\) remain linear gate forces driven by a caller-supplied benefit value frozen for the solver call. Damped powers above one are nonlinear; for nonzero coefficient and \(1<p<2\), a fixed-step guarded run fails closed unless projected Armijo is enabled. None of these variants derives the benefit estimate.
- Hinges/Springs: Provide diagonal curvature that tames bilinear cross‑terms.

---

## 6. When to Use (and When Not To)

Use when:
- You need explicit synergy/competition that cannot be captured by equality springs, hinges, or gates.
- You want AND‑like co‑activation effects without auxiliary constructs.

Avoid or postpone when:
- The current model already converges and expresses the behavior via gates/hinges; bilinear would mainly tighten conditioning.
- You lack sufficient diagonal curvature near those nodes (risk of slow or oscillatory Jacobi).

---

## 7. Proposed API (Design Sketch)

Note: This is a sketch; not enabled in the codebase yet.

```python
from dataclasses import dataclass
from typing import Mapping, Tuple
from core.interfaces import EnergyCoupling, OrderParameter, SupportsCouplingGrads

@dataclass(frozen=True)
class BilinearCoupling(EnergyCoupling, SupportsCouplingGrads):
    """Bilinear coupling: w * eta_i * eta_j (indefinite; use with guards)."""
    weight: float = 1.0

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, float],
    ) -> float:
        return float(self.weight) * float(eta_i) * float(eta_j)

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, float],
    ) -> Tuple[float, float]:
        w = float(self.weight)
        return w * float(eta_j), w * float(eta_i)
```

Integration points:
- Lipschitz estimator: add |w| to raw off-diagonal row sums and |w| / sqrt(p_i p_j) to normalized rows (no diagonal add).
- Curvature protocol: report `(0, 0, abs(w))` so both Gershgorin rows include the off-diagonal contribution.
- Optional Small-Gain allocator: treat |w| as family cost when weights are adaptive.
- Optional: 2×2 local solver path when `use_stiffness_updates` and `has_bilinear_edges=True`.

---

## 8. Safety Checklist

- Keep the curvature-based step cap enabled and start with small fixed bilinear weights.
- Ensure diagonal curvature (from locals/springs/hinges or a tiny regularizer) near bilinear edges.
- Prefer GS or local 2×2 block solves on bilinear edges.
- Maintain accepted-state monotonicity; reject increases above the configured numerical tolerance.
- Do not feed bilinear into diagonal precision used by PSON scaling.
- Use the optional Small-Gain allocator only in experiments that adapt coupling-family weights.

---

## 9. Tests to Add (once implemented)

1) Gradients: analytic vs finite‑difference on random η pairs and weights
2) Gershgorin: row sum increases by |w| for both rows; diagonal unchanged
3) Stability: the raw row-sum bound covers the exact Hessian spectral radius, and the normalized bound covers \(P^{-1/2}HP^{-1/2}\), on random bilinear plus local quadratic graphs
4) Solver: Compare a validated GS or SPD 2×2 update against weighted Jacobi on the same graph
5) PSON: re-projection restores \(g^\top\delta=0\) above the numerical threshold, uniform box scaling preserves it, and the full-Hessian noise cost is recorded separately from the diagonal proxy

---

## 10. Summary

Adding a bilinear coupling introduces true off-diagonal curvature for concise multiplicative interactions. It increases expressivity, but tightens stability and conditioning. Require explicit curvature reporting, adequate diagonal curvature, and guarded GS or 2×2 local updates. Keep it behind a flag and validate the SPD assumptions, monotone acceptance, and telemetry before wider use.
