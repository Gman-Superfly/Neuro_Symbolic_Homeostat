# Complexity from Constraints: The Neuro‑Symbolic Homeostat
## Matrix–Message Relaxation with Precision‑Scaled Orthogonal Noise and Stability Projection

**Authors:** Oscar Goldman
**Date:** November 2025
**Status:** V1 (ish) with working code and demos

### Abstract

We present a neuro-symbolic coordination framework that represents selected logical constraints as energy terms and computes constraint-conditioned order-parameter states by relaxation. The main mechanism is a composable curvature contract: modules report local stiffness, couplings report curvature bounds, and the coordinator composes those reports into precision-aware updates, conservative step caps, and noise scaling. The optimization mathematics is standard, using diagonal preconditioning, Gershgorin row-sum bounds, and the quadratic/SPD equivalence between Gaussian Belief Propagation and Jacobi or Gauss-Seidel iterations. The repository-specific contribution is the way these pieces are exposed as typed module and coupling interfaces. Tests show a regime boundary: tight quadratic bounds can make curvature-aware relaxation converge where plain gradient descent stalls above \(2/L\), while conservative mixed-regime bounds can trade speed for margin. Counterfactual gate-benefit couplings and precision-scaled orthogonal noise are included as typed energy mechanisms with synthetic validation; broader task-level benefits remain empirical.

---

## 1. Introduction

Modern “System‑1” models can produce useful scores while still violating explicit constraints. Symbolic solvers can enforce rules but often require crisp inputs and discrete decisions. We develop a relaxation layer that
- represents constraints as energy terms,
- adds exploration under acceptance checks, and
- uses curvature reports to scale updates and cap steps.

Our core design principle is modular coordination: modules expose order parameters and energies; the coordinator relaxes the global energy with null‑space exploration, stability guards, and non-local gate updates.

### 1.1 Proof status

This paper separates mechanism validity, guard behavior, and empirical benefit. A mechanism can follow from the stated energy rule and still require empirical tests to show fewer steps, lower final energy, lower constraint violation, or changed behavior on a given task class.

- CGBC validity: `GateBenefitCoupling` implements $F=-w\,\eta_{\text{gate}}\,\Delta_{\text{benefit}}$, so $\partial F/\partial\eta_{\text{gate}}=-w\,\Delta_{\text{benefit}}$. This shows analytically that a caller-supplied nonzero benefit estimate gives the gate a gradient even at $\eta_{\text{gate}}=0$.
- CGBC direction and empirical behavior: the sign of the supplied benefit estimate controls the force direction. Correct-sign estimates can push a gate upward under the monotone guard. Wrong-sign estimates can push the gate in the wrong direction and must be tested or guarded.
- PSON validity: orthogonal projection gives $g^\top\delta=0$, so the first-order energy change is zero in the stated quadratic setting.
- PSON cost and empirical behavior: the second-order term $\tfrac12\beta^2\delta^\top H\delta$ can be positive, so the implementation relies on small magnitude, precision scaling, and rejection/restoration. The measured ablation in Section 7 supports lower noise curvature cost for precision-orthogonal noise. Escape behavior and task-level effects remain empirical.
- Small-Gain validity: in quadratic/SPD regimes, if the Gershgorin estimate upper-bounds the largest eigenvalue and $\alpha<2/L$, then $I-\alpha H$ is contractive.
- Small-Gain benefit: the current allocator spends a global estimated margin and reports row margins. Fewer steps or lower final energy depends on the task and remains empirical.
- Stiffness/GaBP validity: for quadratic/SPD blocks, synchronous stiffness updates match Jacobi-style preconditioned gradient updates. Sequential Gauss-Seidel remains an algebraic reference and future scheduler target in this implementation.

### 1.2 Scope of contribution and empirical regime boundary

This repository uses standard optimization mathematics:
- gradient descent stability condition $\alpha < 2/L$,
- Gershgorin-style row-sum upper bounds for conservative Lipschitz control,
- diagonal preconditioning and Jacobi equivalence for quadratic/SPD blocks.

The repository-specific contribution is the composable curvature contract:
- modules can expose local curvature through `SupportsPrecision.curvature`,
- couplings can expose row-wise curvature bounds through `SupportsCouplingCurvature.coupling_curvature_bounds`,
- the coordinator composes these values into one diagonal precision cache and one global Lipschitz estimate used by preconditioning, step capping, and precision-aware noise scaling.

Current tests show two regimes:
- Tight-bound quadratic regime (`tests/test_precision_conditioning.py::test_curvature_awareness_converges_where_plain_gd_stalls_above_2_over_L`): with $\lambda_{\max}=33$, we get $2/L=0.0606$. A requested step of $0.1$ stalls for plain gradient descent, while curvature-aware modes converge.
- Conservative-bound mixed regime (`tests/test_precision_conditioning.py::test_gershgorin_cap_can_be_conservative_on_mixed_preconditioned_problem`): the initial cap is below a requested step of $0.1$, yet both guarded and unguarded preconditioned runs converge. Over a fixed step budget, the guarded run is more conservative and can end at higher final energy.

---

## 2. Theoretical Framework

### 2.1 Four views
- Physics (Energy Minimization): Relax toward lower energy under local and coupling terms.
- Control theory: Small‑gain constraints and Gershgorin step caps provide sufficient contraction conditions in the stated linear/SPD regimes.
- Statistics (Gaussian Graphical Models): Couplings act as messages; precision (inverse variance) is stiffness.
- Information Theory (Channel Capacity): Manage bandwidth vs error; adapt to SNR via precision‑aware updates.

### 2.2 Precision‑Scaled Orthogonal Noise (PSON)
Standard Langevin noise can break monotonicity. We inject noise in the tangent plane orthogonal to the gradient and scale it by inverse precision (local curvature):

$$
\xi_{\mathrm{injection}} \propto \Lambda^{-1}\,\mathrm{proj}_{\nabla \mathcal{F}^\perp}\big(\mathcal{N}(0, I)\big)
$$

(Eq. 1)

PSON explores flat directions (null‑space) without fighting descent to first order. In the synthetic ablation in Section 7, precision scaling reduced the measured noise curvature cost relative to isotropic and unscaled orthogonal noise.
When a problem metric $M$ is available, we use an $M$-orthogonal projection (replace “⊥” by “⊥_M”) and re‑project after precision weighting to preserve $M$‑orthogonality.

Proposition (Quadratic PSON first-order property). Let $F(x) = \tfrac12 (x-x^\star)^\top H (x-x^\star)$ with $H \succeq 0$ and gradient $g = \nabla F(x) = H(x-x^\star)$. Let $\delta$ be a noise vector projected orthogonal to $g$ (Euclidean or metric‑orthogonal) and scaled by $\Lambda^{-1}$. Then the first‑order change vanishes, $g^\top \delta = 0$, and
$\Delta F \;=\; F(x+\beta\delta) - F(x) \;=\; \tfrac12 \beta^2 \delta^\top H \delta \;\ge\; 0.$
Thus, a pure noise move is generally second-order uphill in positive-curvature directions. The implementation preserves accepted-step monotonicity through the down-only acceptance rule; precision scaling ($\Lambda^{-1}$) reduces the curvature cost by biasing $\delta$ toward low-curvature directions.

### 2.3 Counterfactual gate-benefit coupling (CGBC)
Counterfactual gate-benefit coupling (CGBC), nicknamed a wormhole coupling in the implementation, lets closed gates receive forces proportional to a caller-supplied estimate of downstream benefit. With gate-benefit energy

$$
F_{\text{gate}} = -w\, \eta_{\text{gate}}\, \Delta_{\text{benefit}},
$$

(Eq. 2)
the gradient w.r.t. the gate is independent of the current gate value:

$$
\frac{\partial F}{\partial \eta_{\text{gate}}} = -w\, \Delta_{\text{benefit}}.
$$

(Eq. 3)
This provides a non-local gate force akin to the “nudge” in Equilibrium Propagation. It supplies a gate gradient without backpropagating through an inactive path.

Explicit sign check. From (Eq. 3), $\mathrm{sign}\big(\partial F/\partial \eta_{\text{gate}}\big) = -\,\mathrm{sign}(\Delta_{\text{benefit}})$. Thus when downstream benefit is positive, the gradient pushes the gate upward (reducing energy), irrespective of the current $\eta_{\text{gate}}$; conversely for negative benefit.

### 2.4 Stability and the GaBP Link
The iteration is contractive in tested quadratic/SPD regimes when the Gershgorin Lipschitz estimate upper-bounds the largest eigenvalue and the step cap keeps $\alpha < 2/L$. For strictly quadratic sub‑problems with SPD precision $J$, minimizing $F(x)=\tfrac{1}{2}x^\top J x - h^\top x$ is equivalent to solving $Jx = h$. In this regime the GaBP mean update with a synchronous (resp. sequential) schedule is algebraically equivalent to Jacobi (resp. Gauss–Seidel). The implemented stiffness‑based step $x \leftarrow x - D^{-1}(Jx - h)$ realizes the synchronous Jacobi form without explicit message objects. Convergence holds under walk‑summability / $\rho(I - D^{-1}J) < 1$ for the Jacobi form; the coordinator's separate step cap enforces the standard gradient iteration condition $\rho(I-\alpha H)<1$ in the quadratic/SPD case when its Lipschitz bound is valid. See “GaBP ↔ Linear Solvers” in the repository documentation for the derivation and references.

---

## 3. Message Passing ↔ Gradient Descent: When Are They the Same?

Consider quadratic energy

$$
\displaystyle F(x) = \tfrac{1}{2} x^\top J x - h^\top x
$$

(Eq. 4)
with SPD precision matrix $J$.

We denote $D = \mathrm{diag}(J)$ and write $J = D + L + U$ with $L$ strictly lower‑ and $U$ strictly upper‑triangular parts. Solving $Jx = h$ via iterative methods yields the following equivalences:

- GaBP (means) with a synchronous schedule matches Jacobi; with a sequential schedule matches Gauss–Seidel (GS).
- Gradient descent with diagonal preconditioning ($\alpha = D^{-1}$) reproduces Jacobi; with triangular preconditioning ($(D+L)^{-1}$) reproduces GS.

Therefore, for Gaussian/quadratic sub‑problems under SPD and standard scheduling, “message passing” and “(preconditioned) gradient descent” are the same computation up to ordering. Convergence requires the spectral radius of the iteration matrix < 1; for GaBP this is “walk‑summability,” closely related to diagonal dominance. This dovetails with the Small‑Gain constraint (loop gains < 1).

Scope and realization. We scope GaBP claims strictly to SPD/quadratic blocks and the standard scheduling equivalence (Jacobi/GS). The current implementation realizes the synchronous Jacobi form via per‑coordinate stiffness‑based updates: it divides the gradient by the diagonal curvature $\Lambda_{ii}$ aggregated from module precision and coupling curvature (quadratic and active hinges). It does not introduce explicit message objects. Sequential GS scheduling is an algebraic reference and future scheduler target, not a separate implemented stiffness mode in this version.

---

## 4. Architecture & Mechanisms

### 4.1 Modules, Energies, and Precision
Modules expose order parameters and implement local energies. Couplings encode interactions (springs, hinges, and CGBC/wormhole terms). A `SupportsPrecision` interface elevates curvature (precision) to a first‑class signal. The coordinator aggregates a diagonal precision vector $\Lambda$ from module curvature and coupling curvature (quadratic and active hinges) and, when enabled (`use_stiffness_updates`), applies per‑coordinate updates $\Delta \eta_i = -(\partial F/\partial \eta_i)/(\Lambda_{ii}+\varepsilon)$. This same $\Lambda$ modulates PSON to emphasize exploration along flat directions. Vectorized graph caches avoid Python overhead.

### 4.2 Stability Projector (Small‑Gain Allocator)
The implementation uses a Gershgorin-style Lipschitz estimate as a stability projector for the update step. For quadratic/SPD blocks, the coordinator caps the step below the standard $2/L$ gradient-descent bound. The Small‑Gain adapter then treats the remaining estimated margin as a budget for bounded coupling-weight increases. In mixed regimes (gates/hinges) this remains a conservative guard and telemetry mechanism rather than a proof of global nonlinear contraction.

Algorithm (implemented step cap). Let $L$ denote a conservative Gershgorin-style upper bound on the gradient Lipschitz constant. The coordinator estimates $L$ from local curvature and coupling curvature using row sums:

$$
L \;\le\; \max_i \left(d_i + \sum_{j\neq i} |c_{ij}|\right).
$$

(Eq. 5)

Given requested step size $\alpha_{\mathrm{req}}$, the used step is capped by

$$
\alpha_{\mathrm{used}} \;=\; \min\!\Big(\alpha_{\mathrm{req}},\; \gamma\,\frac{2}{L}\Big), \qquad 0 < \gamma < 1.
$$

(Eq. 6)

The recorded contraction margin is $(2/L)-\alpha_{\mathrm{used}}$. The Small‑Gain adapter receives global and row margin estimates plus per-coupling Lipschitz costs, then applies bounded weight increases only while the predicted global spend remains inside the configured budget fraction.

Bound (quadratic/SPD case). For $F(x)=\tfrac12x^\top Hx-h^\top x$ with SPD $H$, if $L$ upper-bounds $\lambda_{\max}(H)$ and $\alpha_{\mathrm{used}} < 2/L$, then the gradient iteration matrix $I-\alpha_{\mathrm{used}}H$ has spectral radius $<1$. Thus the quadratic iteration is contractive. In mixed regimes, the same mechanism is a conservative step cap plus monotone acceptance guard, not a complete nonlinear stability proof.

### 4.3 Counterfactual gate-benefit couplings (CGBCs)
`GateBenefitCoupling` implements CGBC, with “wormhole coupling” retained as the implementation nickname. It injects non-local gradients even for closed connections when the caller supplies a downstream benefit estimate. Damped variants provide smoother activation curves. In the current repository this mechanism is tested on synthetic gating cases; broader planning and sequence uses remain hypotheses to test with task-level ablations.

### 4.4 Precision‑Scaled Orthogonal Noise (PSON)
Null‑space exploration with a monotone acceptance guard. Precision‑aware scaling gives larger perturbations to slack variables while stiff variables take smaller steps. The current evidence supports lower curvature cost for precision‑orthogonal noise in the synthetic ablation; escape behavior and task-level effects remain empirical claims to test on broader tasks.

---

## 5. Algorithms (Composing Matrix Math and Messages)

We retain Dynamic Gradient‑Based Energy Minimization as the default inner solver and augment it with optional kernels per factor family.

In mature supervised-learning recipes, catastrophic gradient-descent instability is uncommon because many guardrails are already present. In custom energy systems with non-smooth terms, stiff constraints, coupled objectives, or feedback-like dynamics, step-size sensitivity becomes more central. The optional proximal and ADMM paths exist for those regimes, not as a replacement for ordinary gradient methods.

**Penalty vs Primal–Dual (placement).** By default, the system operates as a penalty/augmented‑Lagrangian scheme (primal variables with adaptive penalty weights). When ADMM mode is enabled, the quadratic-coupling path uses primal–dual-style iterations with explicit auxiliary variables and dual updates.

Augmented kernels:
- Quadratic/Gaussian blocks: GaBP‑style synchronous Jacobi updates = precision‑weighted linear steps with per‑iteration cost $O(\mathrm{nnz}(J))$; no global factorization required.
- Non‑Gaussian/gated/hinge terms: gradient with line search, proximal updates, or experimental ADMM blocks with acceptance guards.
- Mixed graphs: hybrid passes, with GaBP on quadratic stars and prox or gradient on other blocks, under a common stability projector.

**Pseudocode Sketch (Conceptual):**

```python
# One relaxation pass
for block in factorization_order:
    if block.is_quadratic_and_spd():
        # GaBP-style synchronous Jacobi update (precision-weighted)
        x_block = solve_local_system(block)  # matvec-based, no global factorization
    elif block.has_closed_form_prox():
        x_block = prox_update(block)
    else:
        x_block = gradient_step(block, preconditioner=diag_precision)
        # Or, per-coordinate stiffness update (force / stiffness):
        # eta_i -= grad_i / (diag_precision_i + eps)
    apply_pson_tangent_noise(x_block)         # orthogonal, precision-scaled
    enforce_small_gain_stability_projection() # cap the step under the supported stability assumptions
accept_if_monotone_or_guarded()
```

---

## 6. Observability

The implementation includes relaxation trackers and stability telemetry: per‑step ΔF, acceptance provenance, contraction margins (global and row), precision‑diagonal stats (min/median/max), and “budget vs spend” for Small‑Gain. These signals make convergence and stability behavior inspectable during experiments.

---

## 7. Empirical Guidance and Repro

- Orthogonal vs isotropic noise: compare ΔF histograms and sharpness at matched loss.
- Precision‑aware vs uniform noise scaling: escape events, ΔF90, final energy.
- Small‑Gain vs line‑search‑only vs GradNorm: ΔF90, backtracks, final energy on dense graphs.
- CGBC/wormhole ablation: activation/opening rates vs energy drop versus hinge/quadratic baselines.

**Table 1: PSON ablation summary (synthetic quadratic and mixed-gate tasks)**

Command: `uv run python -m experiments.ablate_pson_noise --trials 30 --steps 80`

Scenario details: 30 seeds per condition, 80 relaxation steps, size 12. The quadratic task uses local quadratic wells plus quadratic couplings. The mixed task adds CGBC gate-benefit couplings to the same synthetic graph family. Values are mean ± sample standard deviation.

| Scenario | Noise mode | ΔF90 steps | Final energy | Acceptance rate | Rejected steps | Noise curvature cost |
|---|---:|---:|---:|---:|---:|---:|
| Quadratic | Isotropic | 6.53 ± 0.51 | 0.1053 ± 0.0352 | 0.947 ± 0.007 | 1.00 ± 0.00 | 1.238e-3 ± 3.846e-4 |
| Quadratic | Orthogonal | 6.43 ± 0.50 | 0.1041 ± 0.0351 | 0.959 ± 0.007 | 1.00 ± 0.00 | 1.150e-3 ± 3.737e-4 |
| Quadratic | Precision-orthogonal | 6.43 ± 0.50 | 0.1033 ± 0.0352 | 0.965 ± 0.005 | 1.00 ± 0.00 | 5.077e-4 ± 1.568e-4 |
| Mixed-gate | Isotropic | 6.53 ± 0.51 | 0.0442 ± 0.0360 | 0.947 ± 0.007 | 1.00 ± 0.00 | 1.238e-3 ± 3.846e-4 |
| Mixed-gate | Orthogonal | 6.40 ± 0.50 | 0.0431 ± 0.0359 | 0.960 ± 0.006 | 1.00 ± 0.00 | 1.150e-3 ± 3.738e-4 |
| Mixed-gate | Precision-orthogonal | 6.40 ± 0.50 | 0.0424 ± 0.0360 | 0.964 ± 0.005 | 1.00 ± 0.00 | 5.077e-4 ± 1.571e-4 |

Interpretation. Precision-orthogonal noise reduced the curvature cost of injected noise by roughly 56% relative to orthogonal noise and roughly 59% relative to isotropic noise in both synthetic tasks. Final energy and ΔF90 were similar across modes, with small final-energy differences for precision-orthogonal noise in this sweep. These results support the curvature-cost claim. They do not by themselves show lower escape time from local minima or broader task behavior.

**Reproducibility Commands (Windows PowerShell):**

```powershell
# CGBC/wormhole demo
uv run python -m experiments.demo_wormhole

# Unit tests (subset)
uv run -m pytest tests -k "gate_benefit or couplings" -v

# Adapter comparison sweep (example)
uv run python -m experiments.benchmark_delta_f90 --configs default gradnorm smallgain --steps 60

# SmallGain validation sweep (see docs/SMALLGAIN_VALIDATION_FINAL.md)
uv run python -m experiments.sweep_smallgain --quick

# PSON noise ablation used for Table 1
uv run python -m experiments.ablate_pson_noise --trials 30 --steps 80
```

Note: To enable stiffness‑based per‑coordinate updates in your own scripts, construct the coordinator with `use_stiffness_updates=True`; adapters (Small‑Gain, etc.) remain unchanged and continue to shape the effective stiffness through term weights.

---

## 8. Future Work

- Asynchronous/priority scheduling variants for sparse graphs: implement prioritized updates and compare wall‑clock efficiency against synchronous passes.
- Extend the isotropic vs orthogonal vs precision‑orthogonal noise ablation to harder nonconvex and task-level settings.

---

## 9. Limitations

- GaBP equivalence applies to Gaussian/quadratic sub‑problems with SPD precision; mixed regimes require hybrid updates and guards.
- Walk‑summability/diagonal‑dominance violations can stall/oscillate; Small‑Gain projection mitigates but cannot fix poor modeling.
- Precision tracking uses diagonal approximations by default; full metrics require SPD and careful conditioning.
- CGBC benefit estimation quality affects activation dynamics; use conservative estimates with monotone acceptance. Broader planning and sequence claims require task-level tests beyond the synthetic gating tests.

---

## 10. Related Work

- Equilibrium Propagation (Scellier & Bengio): nudge‑based non‑local gradient in energy‑based models.
- Gaussian BP and walk‑sums (Weiss & Freeman; Malioutov et al.): equivalence to classical linear solvers and convergence conditions.
- Small‑gain/passivity (Zames; Vidyasagar): loop‑gain constraints for stability in nonlinear feedback systems.
- Operator‑splitting/ADMM (Boyd et al.): proximal and primal‑dual updates for composite objectives.

Walk‑summability vs diagonal dominance. Following Malioutov et al., walk‑summability gives a sufficient convergence condition for GaBP and can be related to diagonal dominance and spectral‑radius conditions for Jacobi/GS. Our Small‑Gain projector can be viewed as enforcing a diagonal‑dominance‑like margin (via Gershgorin bounds), aligning these views.

---

## 11. Conclusion

The main result is a coordination mechanism: curvature becomes a component contract. When modules and couplings report truthful local stiffness, the coordinator can compose those reports into stability-controlled, precision-aware relaxation without choosing a separate step rule for each interaction. The tests show both sides of that claim. Tight quadratic bounds change the outcome, while conservative mixed-regime bounds can trade speed for margin. The framework depends on accurate curvature contracts, explicit guards, and measured regime boundaries.

---

### References

Acknowledgement: this work was influenced by Furlat's Abstractions repository, especially its distributed-computation techniques: https://github.com/furlat/Abstractions

Theory references:

1. Weiss, Y., & Freeman, W. T. (2001). Correctness of Belief Propagation in Gaussian Graphical Models of Arbitrary Topology. Neural Computation.
2. Malioutov, D., Johnson, J. K., & Willsky, A. S. (2006). Walk‑sums and belief propagation in Gaussian graphical models. Journal of Machine Learning Research.
3. Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM.
4. Scellier, B., & Bengio, Y. (2017). Equilibrium Propagation: Bridging the Gap Between Energy‑Based Models and Backpropagation. Frontiers in Neuroscience.
5. Zames, G. (1966). On the input‑output stability of time‑varying nonlinear feedback systems. IEEE TAC.
6. Vidyasagar, M. (1993). Nonlinear Systems Analysis. Prentice Hall.
7. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends in Machine Learning.

---

### Citation

If you use this repository in your research, please cite it. This is ongoing work; we would like to know your opinions and experiments. Thank you.

**Authors:** Oscar Goldman - Shogu Research Group @ Datamutant.ai, subsidiary of 温心重工業.

**Reference (author-year format):** Goldman, O. (2025). *Complexity from Constraints: The Neuro-Symbolic Homeostat*. Software repository. Shogu Research Group @ Datamutant.ai, subsidiary of 温心重工業.


---

### License

ccb4

© 2025 Oscar Goldman
Status: work in progress...
---

### Notes on Implementation

- Codebase: Python, protocol‑based architecture; hot‑swappable NumPy/Torch/JAX backends.
- Vectorization: Compile‑time graph vectorization cache to amortize sparse passes.
- Observability: Event‑driven telemetry (RelaxationTracker) for energy descent, stability margins, and adapter spend.
- Precision Layer: Implemented; diagonal curvature aggregates module and coupling curvature; supports per‑coordinate stiffness‑based steps (`use_stiffness_updates`) and precision‑scaled PSON.

