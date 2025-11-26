# Bilinear Couplings in the Neuro‑Symbolic Homeostat

Status: Design document (optional feature; not enabled by default, you will have to implement it)  
Intent: Explain what a bilinear coupling adds, its impact on stability/solver choices, and a safe path to implement behind a flag.

---

## 1. Motivation

Existing couplings cover:
- Quadratic springs: encourage equality (η_i − η_j → 0)
- Hinge variants: one‑sided activation/gaps
- Wormhole: linear non‑local force

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

## 3. Stability and Small‑Gain

Let \(L\) denote the Lipschitz bound of ∇F estimated via Gershgorin row sums. For a bilinear edge (i, j):

- It contributes |w_{ij}| to each row’s off‑diagonal sum.
- It does not add to diagonal curvature by itself.

Therefore:
- Diagonal dominance can be lost more easily as |w_{ij}| grows.
- Small‑Gain budgeting should cap the effective |w_{ij}| contribution to keep loop gains < 1.

Recommended guardrails:
- Treat |w_{ij}| as “spend” against the row/global budget in the Small‑Gain allocator.
- Prefer lower budget fractions for bilinear families (e.g., ρ=0.5 vs 0.7) when first enabling.
- Keep monotone acceptance (reject steps with ΔF > 0).

---

## 4. Solver Integration

### 4.1 Per‑Coordinate Stiffness Update

Our stiffness update uses \(\Delta \eta_i \approx -g_i / (\Lambda_{ii} + \varepsilon)\), where \(\Lambda\) is the diagonal curvature aggregated from modules and convex couplings. Pure bilinear adds zero to \(\Lambda_{ii}\). Consequences:
- With bilinear only, denominators are near ε → updates resemble un‑preconditioned GD on those coordinates.
- Convergence can slow/oscillate under Jacobi unless sufficient diagonal curvature is present (from locals, springs, or a tiny diagonal regularizer).

### 4.2 Gauss–Seidel / 2×2 Local Block

For variables i and j with diagonal curvatures \(a_i, a_j \ge 0\) (from locals/other convex terms) and a bilinear cross‑term \(w\), a local Newton step solves:

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

This 2×2 system is cheap and stabilizes updates when \(w \neq 0\). It is recommended to pair bilinear edges with either GS scheduling (sequential updates) or a small 2×2 block solve for the linked pair.

### 4.3 Practical Recipe
- Require (or induce) modest diagonal curvature \(a_i, a_j > 0\) where bilinear edges exist (from locals or a tiny diagonal regularizer).
- Prefer GS/priority scheduling in sparse graphs; avoid purely synchronous Jacobi if many strong bilinear edges exist.
- Keep Small‑Gain active to cap |w| and protect contraction.

---

## 5. PSON, Wormhole, and Mixed Regimes

- PSON: Keep bilinear out of the diagonal precision used for noise scaling. Continue scaling noise with \(\Lambda^{-1}\) (module + convex coupling curvature) and keep orthogonality to the gradient.
- Wormhole: Unchanged. Wormhole is linear (force), not curvature; it pairs well with bilinear but is orthogonal in purpose (non‑local credit vs multiplicative interaction).
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
- Lipschitz estimator: add |w| to off‑diagonal row sums (no diagonal add).
- Small‑Gain allocator: treat |w| as “cost” per family for budget/spend.
- Optional: 2×2 local solver path when `use_stiffness_updates` and `has_bilinear_edges=True`.

---

## 8. Safety Checklist

- Keep Small‑Gain enabled; cap bilinear family scale aggressively at first.
- Ensure diagonal curvature (from locals/springs/hinges or a tiny regularizer) near bilinear edges.
- Prefer GS or local 2×2 block solves on bilinear edges.
- Maintain monotone acceptance; reject non‑decreasing steps.
- Do not feed bilinear into diagonal precision used by PSON scaling.

---

## 9. Tests to Add (once implemented)

1) Gradients: analytic vs finite‑difference on random η pairs and weights  
2) Gershgorin: row sum increases by |w| for both rows; diagonal unchanged  
3) Stability: with Small‑Gain on, ΔF ≤ 0 for a small graph with bilinear + locals  
4) Solver: GS/2×2 update converges faster/cleaner than Jacobi on the same graph  
5) PSON: noise energy contribution remains bounded; orthogonality preserved  

---

## 10. Summary

Adding a bilinear coupling introduces true off‑diagonal curvature that enables concise multiplicative interactions. It increases expressivity, but tightens stability/conditioning; safe deployment requires Small‑Gain budgeting, adequate diagonal curvature, and GS/2×2 local updates. Keep it behind a flag and validate with monotone acceptance and telemetry before wider use.


