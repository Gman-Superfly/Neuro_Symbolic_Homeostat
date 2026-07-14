# Complexity from Constraints: The Neuro‑Symbolic Homeostat
## Composable Energy Relaxation with Precision-Scaled Orthogonal Noise and Stability Guards

**Authors:** Oscar Goldman
**Date:** November 2025
**Revision:** July 2026
**Status:** Research prototype with mechanism-level synthetic validation

### Abstract

We present a neuro-symbolic coordination framework that represents selected logical constraints as energy terms and computes constraint-conditioned order-parameter states by relaxation. The main mechanism is a composable curvature contract: modules report local stiffness, couplings report curvature bounds, and the coordinator composes those reports into precision-aware updates, conservative step caps, and noise scaling. The optimization mathematics is standard, using diagonal preconditioning, Gershgorin row-sum bounds, and the Jacobi form of diagonally preconditioned relaxation on quadratic/SPD systems. Gaussian belief propagation solves the same Gaussian linear-inference problem through explicit messages, but this repository does not implement GaBP or claim stepwise equivalence in general. The repository-specific contribution is the typed module and coupling contract. Tests show a regime boundary: tight quadratic bounds can make curvature-aware relaxation converge where plain gradient descent stalls above \(2/L\), while conservative mixed-regime bounds can trade speed for margin. Counterfactual gate-benefit couplings and precision-scaled orthogonal noise are included as typed energy mechanisms with synthetic validation; broader task-level benefits remain empirical.

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
- PSON cost and empirical behavior: the second-order term $\tfrac12\beta^2\delta^\top H\delta$ can be positive, so the implementation relies on small magnitude, precision scaling, and rejection/restoration. The measured ablation in Section 7 supports lower noise curvature cost for precision-orthogonal noise. A controlled anisotropic double-well experiment also measures escape behavior in one designed regime; task-level effects remain untested.
- Stability-guard validity: in quadratic/SPD regimes, if the Gershgorin estimate upper-bounds the largest eigenvalue and $\alpha<2/L$, then $I-\alpha H$ is contractive.
- Small-Gain benefit: the current allocator spends a global estimated margin and reports row margins. Fewer steps or lower final energy depends on the task and remains empirical.
- Stiffness/Jacobi validity: for quadratic/SPD systems, synchronous stiffness updates match Jacobi-style preconditioned gradient updates. Gaussian BP is a related message-passing solver for the same linear-inference problem; it is not implemented or tested here. Sequential Gauss-Seidel remains an algebraic reference and future scheduler target.

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
When a symmetric positive definite problem metric $M$ is available, we define the metric gradient as $g_M=M^{-1}g$, where $g=\nabla F(x)$ is the ordinary gradient covector. The metric projection is $M$-orthogonal to $g_M$, which is equivalent to $g^\top\delta=0$. A dense metric uses a linear solve with $M$; matrix-free callers can provide the action of $M^{-1}$ through `metric_solve`. After precision weighting, the implementation re-projects with the same geometry.

Proposition (Quadratic PSON first-order property). Let $F(x) = \tfrac12 (x-x^\star)^\top H (x-x^\star)$ with $H \succeq 0$ and ordinary gradient $g = \nabla F(x) = H(x-x^\star)$. In Euclidean mode, project the noise orthogonal to $g$. In metric mode with SPD metric $M$, project along $g_M=M^{-1}g$ using $\delta=z-\frac{g^\top z}{g^\top M^{-1}g}M^{-1}g$. Apply any precision scaling before a final projection with the selected geometry. Then the final vector satisfies $g^\top \delta = 0$, and
$\Delta F \;=\; F(x+\beta\delta) - F(x) \;=\; \tfrac12 \beta^2 \delta^\top H \delta \;\ge\; 0.$
Thus, a pure noise move is generally second-order uphill in positive-curvature directions. The implementation preserves accepted-step monotonicity through the down-only acceptance rule. Precision scaling ($\Lambda^{-1}$) reduces the measured curvature cost in the synthetic ablation by biasing $\delta$ toward low-curvature directions before the final orthogonality projection.

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

### 2.4 Stability and the Gaussian linear-system link
The iteration is contractive in tested quadratic/SPD regimes when the Gershgorin Lipschitz estimate upper-bounds the largest eigenvalue and the step cap keeps $\alpha < 2/L$. For a quadratic objective with SPD precision $J$, minimizing $F(x)=\tfrac{1}{2}x^\top Jx-h^\top x$ is equivalent to solving $Jx=h$. The implemented stiffness step $x \leftarrow x-D^{-1}(Jx-h)$ is exactly the Jacobi iteration for this system. Gaussian belief propagation is another distributed method for Gaussian inference and can recover the same mean when its message updates converge, subject to its own conditions. General GaBP also updates message precisions, so its transient iterations are not identified with the coordinator's Jacobi trajectory. Walk-summability is a sufficient condition used in the GaBP literature; the tested coordinator condition is $\rho(I-D^{-1}J)<1$ for its Jacobi form. The separate scalar step cap enforces $\rho(I-\alpha H)<1$ when the composed Lipschitz bound is valid.

---

## 3. Quadratic Relaxation and Related Message Passing

Consider quadratic energy

$$
\displaystyle F(x) = \tfrac{1}{2} x^\top J x - h^\top x
$$

(Eq. 4)
with SPD precision matrix $J$.

We denote $D = \mathrm{diag}(J)$ and write $J = D + L + U$ with $L$ strictly lower and $U$ strictly upper triangular. Classical linear iterations give:

- The update $x^{t+1}=x^t-D^{-1}(Jx^t-h)$ is Jacobi and is exactly the implemented stiffness trajectory in the tested quadratic case.
- A triangular solve with $(D+L)^{-1}$ gives the Gauss-Seidel trajectory; that scheduler is not implemented here.
- GaBP uses edge messages with evolving cavity precisions. When it converges on the corresponding Gaussian model, its mean solves $Jx=h$, but its intermediate updates need not match Jacobi or Gauss-Seidel.

The shared linear system provides the connection between the methods. It does not make the algorithms stepwise identical. Jacobi convergence requires $\rho(I-D^{-1}J)<1$. Walk-summability provides a separate sufficient convergence condition for Gaussian BP and is related to diagonal-dominance classes.

Scope and realization. The current implementation realizes the synchronous Jacobi form through per-coordinate stiffness updates. It divides the gradient by diagonal curvature $\Lambda_{ii}$ aggregated from module precision and coupling curvature. It has no Gaussian message objects, cavity-precision updates, or dedicated sequential Gauss-Seidel scheduler.

---

## 4. Architecture & Mechanisms

### 4.1 Modules, Energies, and Precision
Modules expose order parameters and implement local energies. Couplings encode interactions (springs, hinges, and CGBC/wormhole terms). A `SupportsPrecision` interface elevates curvature (precision) to a first‑class signal. The coordinator aggregates a diagonal precision vector $\Lambda$ from module curvature and coupling curvature (quadratic and active hinges) and, when enabled (`use_stiffness_updates`), applies per‑coordinate updates $\Delta \eta_i = -(\partial F/\partial \eta_i)/(\Lambda_{ii}+\varepsilon)$. This same $\Lambda$ modulates PSON to emphasize exploration along flat directions. Vectorized graph caches avoid Python overhead.

### 4.2 Stability Guard and Small-Gain Allocator
The implementation uses a Gershgorin-style Lipschitz estimate as a stability guard for the update step. For quadratic/SPD systems, the coordinator caps the step below the standard $2/L$ gradient-descent bound. The optional Small-Gain adapter treats the remaining estimated margin as a budget for bounded coupling-weight increases. The step cap supplies the scoped contraction result. The allocator supplies a global curvature-spend policy and telemetry; it is not a nonlinear small-gain certificate.

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

The maintained default is guarded gradient relaxation, with optional diagonal stiffness scaling. `SolverConfig` selects one of three mutually exclusive paths: gradient, proximal, or ADMM-like relaxation. Grouped `CoordinatorConfig` objects separate gradient, execution, guard, noise, weight, and solver settings for new callers; the flat constructor remains for compatibility.

In mature supervised-learning recipes, catastrophic gradient-descent instability is uncommon because many guardrails are already present. In custom energy systems with non-smooth terms, stiff constraints, coupled objectives, or feedback-like dynamics, step-size sensitivity becomes more central. The optional proximal and ADMM paths exist for those regimes, not as a replacement for ordinary gradient methods.

**Penalty and primal-dual placement.** The default path relaxes primal order parameters under the configured energy and optional adaptive term weights. The ADMM-like path introduces auxiliary and scaled dual variables for selected quadratic and hinge terms; gate-benefit terms use either their gradient or a damped proximal-linear update. This path remains experimental because the current evidence covers synthetic conformance cases rather than general nonconvex convergence.

Implemented kernels:
- Default path: analytic or finite-difference gradients, optional diagonal stiffness scaling, a Gershgorin-style step cap, optional tangent noise, and accepted-step guarding.
- Proximal path: local gradient updates followed by supported pairwise or star-block proximal updates.
- Experimental ADMM-like path: selected quadratic, hinge, and gate-benefit updates with auxiliary and dual variables plus primal and dual residual telemetry.

`tests/test_solver_conformance.py` applies one contract to all three paths. On a bounded two-variable convex quadratic with a closed-form optimum, each path returns finite in-range states, preserves non-increasing accepted energy, and reaches the reference within its stated numerical tolerance. Deliberately oversized proximal and ADMM-like steps test rejection and restoration. On the ADMM convex reference, both recorded residuals fall below $10^{-3}$ after 100 accepted steps. Gate-benefit tests cover the separate proximal-linear update. These checks support the implemented paths on the tested objectives; they do not establish convergence for arbitrary nonconvex compositions.

**Pseudocode Sketch (Implemented Default Path):**

```python
for attempt in range(steps):
    baseline = energy(eta, current_weights)
    gradient = compose_gradient(eta, current_weights)
    diagonal_curvature = compose_precision(eta, current_weights)
    lipschitz_bound = compose_row_sum_bound(eta, current_weights)
    step = min(requested_step, safety_fraction * 2 / lipschitz_bound)
    direction = scale_by_diagonal_curvature(gradient, diagonal_curvature)
    noise = build_tangent_noise(gradient, diagonal_curvature)
    proposal = project_to_box(eta - step * direction + noise)
    if energy(proposal, current_weights) <= baseline:
        eta = proposal
        emit_accepted_state(eta)
        current_weights = adapt_weights_after_acceptance(eta, current_weights)
    elif not continue_after_rejection:
        break
```

---

## 6. Observability

The implementation includes relaxation trackers and stability telemetry: per-step ΔF, acceptance provenance, contraction margins, row and global margin estimates, precision-diagonal stats (min/mean/max), and budget-versus-spend fields for Small-Gain. These signals make convergence and stability behavior inspectable during experiments.

---

## 7. Empirical Guidance and Repro

- Orthogonal vs isotropic noise: compare ΔF histograms and sharpness at matched loss.
- Precision‑aware vs uniform noise scaling: escape events, ΔF90, final energy.
- Small-Gain vs line-search-only vs GradNorm: fixed-reference energy, adaptive-objective energy, ΔF90, and backtracks.
- CGBC/wormhole ablation: activation/opening rates vs energy drop versus hinge/quadratic baselines.

**Table 1: Paired PSON curvature-cost ablation across generated problem families**

Command: `uv run python -m experiments.ablate_pson_noise --trials 30 --steps 80 --noise-cost-samples 32 --bootstrap-samples 10000`

Protocol: each family uses 30 generated seeds and 80 attempted relaxation steps. The three noise modes receive the same problem instance and raw Gaussian draws. Rejected proposals restore the previous state, and the remaining schedule continues. For each mode and seed, noise curvature cost is the mean of 32 perturbation draws at the initial state. The effect is computed as the paired percentage reduction in precision-orthogonal cost relative to each baseline. Intervals are percentile 95% confidence intervals from 10,000 paired hierarchical bootstrap resamples over problem seeds and draw indices.

| Scenario | Structure | Size | Reduction vs isotropic | Reduction vs orthogonal |
|---|---|---:|---:|---:|
| Quadratic chain | Chain, quadratic | 12 | 59.01% [57.79%, 60.08%] | 55.08% [53.69%, 56.40%] |
| Mixed gate chain | Chain, quadratic plus linear CGBC | 12 | 59.01% [57.79%, 60.15%] | 55.08% [53.69%, 56.39%] |
| Quadratic star | Hub-and-spoke, shuffled curvature | 12 | 77.82% [76.45%, 78.96%] | 74.32% [72.88%, 75.51%] |
| Quadratic dense | Dense random graph | 6 | 56.03% [53.53%, 58.28%] | 46.87% [43.13%, 50.19%] |
| Ill-conditioned ring | Ring, local curvature ratio 400 | 24 | 84.46% [83.88%, 84.91%] | 82.78% [82.13%, 83.27%] |
| Nonlinear quartic | Chain, state-dependent curvature | 12 | 46.06% [42.73%, 49.22%] | 42.34% [38.90%, 45.70%] |
| Active hinges | Chain with active directed/asymmetric hinges | 12 | 45.89% [44.22%, 47.34%] | 40.95% [39.32%, 42.45%] |

![Paired PSON curvature-cost reductions](docs/figures/pson_cost_reduction.png)

Interpretation. Precision-orthogonal noise had lower measured initial-state curvature cost in every generated family. Mean paired reductions ranged from 45.89% to 84.46% relative to isotropic noise and from 40.95% to 82.78% relative to unscaled orthogonal noise. The mixed-gate and quadratic-chain cost results are nearly identical because the added CGBC term is linear and therefore changes the gradient but not the curvature geometry. Mean energy drop remained similar across noise modes within each family, while precision-orthogonal noise had the highest mean acceptance rate in all seven families under the fixed attempt budget. Raw trial metrics are recorded in `logs/pson_noise_ablation.csv`; paired effects and bootstrap intervals are recorded in `logs/pson_noise_ablation_summary.csv`.

These intervals quantify sampled seed and perturbation-draw variation under the specified generators. They do not include uncertainty about the choice of graph families, energy constructions, or hyperparameters, and they do not establish task-level benefit. Real-model evaluation is deferred.

### 7.1 Closed-form noise-cost reference

For a three-dimensional diagonal quadratic with curvature $(2,4,16)$, a gradient aligned with the first coordinate, and fixed noise norm $0.02$, the expected isotropic cost is $\beta^2\operatorname{tr}(H)/3$. Orthogonal projection leaves a two-dimensional tangent plane with expected cost $\beta^2(4+16)/2$. In that plane, inverse-curvature weighting gives the weighted harmonic expression $\beta^2(4w_2+16w_3)/(w_2+w_3)$, where $w_i=(H_{ii}+\varepsilon)^{-1}$.

| Mode | Analytic cost | Monte Carlo cost, 100,000 draws | Relative error |
|---|---:|---:|---:|
| Isotropic | 2.933333e-3 | 2.936314e-3 | 0.102% |
| Orthogonal | 4.000000e-3 | 4.000612e-3 | 0.015% |
| Precision-orthogonal | 2.560000e-3 | 2.559886e-3 | 0.004% |

This reference checks the implemented normalization, projection, and precision weighting against a case with a closed-form expectation. It also shows that orthogonality alone can raise curvature cost when the removed gradient direction is relatively flat.

### 7.2 Sampled curvature-contract audit

Command: `uv run python -m experiments.audit_curvature_contract --samples 32 --strict`

The auditor compares reported module and coupling curvature with finite-difference Hessian entries. Across 32 states in each of the seven generated families, all 6,080 sampled component-state records were covered; none were unreported or underreported. A separate negative test deliberately underreports a custom edge and verifies that the auditor labels it. This is sampled diagnostic evidence, not a proof between sampled states.

### 7.3 Controlled nonconvex escape

The escape benchmark uses one asymmetric double-well coordinate and seven stiff quadratic distractor coordinates. All modes start at the higher stationary well. Noise modes receive the same fixed Euclidean norm of $0.55$, 40 attempts, and down-only acceptance. The construction is intentionally anisotropic: inverse-curvature weighting allocates more of the fixed norm to the escape coordinate.

| Mode | Escapes / 200 | Escape rate | Mean energy drop |
|---|---:|---:|---:|
| No noise | 0 | 0.0% | 0.000000 |
| Isotropic | 0 | 0.0% | 0.000000 |
| Orthogonal | 0 | 0.0% | 0.000000 |
| Precision-orthogonal | 66 | 33.0% | 0.006974 |

The paired escape-rate difference for precision-orthogonal noise versus no noise was 33.0 percentage points with a percentile 95% bootstrap interval of [26.5, 39.5]. The same interval applies versus isotropic noise because that baseline had no escapes. This result supports escape behavior only for this controlled anisotropic construction. At the initial stationary point the gradient is zero, so orthogonality is vacuous; the observed difference isolates precision allocation rather than tangent projection.

![Controlled anisotropic double-well escape](docs/figures/pson_escape_rate.png)

### 7.4 Local runtime scaling

The scaling benchmark varies problem size over 16, 64, and 256 variables and edge count from 16 to 4,096. Each graph regime runs in a fresh Python process with two warmups and seven timed repeats. Median relaxation time ranged from 0.846 ms per step at 16 variables and 16 edges, with an interquartile range of 0.805 to 0.896 ms, to 48.141 ms at 256 variables and 4,096 edges, with an interquartile range of 47.275 to 51.679 ms. Peak traced Python allocation ranged from 20.7 KiB to 1.16 MiB; this `tracemalloc` measurement does not include every native allocation or process-level resident byte.

The recorded run used Windows, CPython 3.12.1, NumPy 2.2.4, and the `scipy-openblas` backend. A descriptive log-linear fit over the nine measured regimes gave size and edge-count exponents of 0.413 and 0.536 with $R^2=0.994$. Correlation between graph dimensions and implementation details limits that fit, so it is an empirical summary of this matrix rather than an asymptotic complexity result. The benchmark records raw samples and environment metadata in `logs/scaling_benchmark.csv`; `logs/scaling_model.json` stores the fit. Measurements from a second machine or software stack remain future evidence.

![Recorded local runtime scaling](docs/figures/runtime_scaling.png)

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

# Verify tests, audits, references, figures, paths, and recorded artifacts
.\scripts\verify_publication.ps1

# Regenerate the full recorded experiment set before verification
.\scripts\verify_publication.ps1 -Regenerate
```

Note: To enable stiffness‑based per‑coordinate updates in your own scripts, construct the coordinator with `use_stiffness_updates=True`; adapters (Small‑Gain, etc.) remain unchanged and continue to shape the effective stiffness through term weights.

---

## 8. Future Work

- Asynchronous/priority scheduling variants for sparse graphs: implement prioritized updates and compare wall‑clock efficiency against synchronous passes.
- Extend the controlled escape result to nonconvex families that do not deliberately align low curvature with the escape coordinate.
- Evaluate constraint correction on real model outputs. Real-model evaluation is deferred in the current repository release.

---

## 9. Limitations

- The implemented Jacobi equivalence applies to quadratic/SPD systems. Gaussian BP is related literature context, not an implemented solver path.
- Curvature underreporting invalidates the step-cap contraction premise. Monotone acceptance can reject and restore an uphill proposal, but it cannot recover the missing bound or guarantee progress.
- Precision tracking uses diagonal approximations by default; full metrics require SPD and careful conditioning.
- The controlled escape construction is designed to test anisotropic precision allocation and is not representative of arbitrary nonconvex landscapes.
- The curvature audit samples states and can detect observed underreporting; it cannot certify unsampled states.
- CGBC benefit estimation quality affects activation dynamics; use conservative estimates with monotone acceptance. Broader planning and sequence claims require task-level tests beyond the synthetic gating tests.

---

## 10. Related Work

- Equilibrium Propagation (Scellier & Bengio): nudge‑based non‑local gradient in energy‑based models.
- Gaussian BP and walk-sums (Weiss & Freeman; Malioutov et al.): Gaussian mean inference and sufficient convergence conditions.
- Small‑gain/passivity (Zames; Vidyasagar): loop‑gain constraints for stability in nonlinear feedback systems.
- Operator‑splitting/ADMM (Boyd et al.): proximal and primal‑dual updates for composite objectives.

Walk-summability and diagonal dominance. Following Malioutov et al., walk-summability gives a sufficient convergence condition for Gaussian BP and is related to diagonal-dominance classes. The coordinator separately uses Gershgorin row sums to bound quadratic curvature for its gradient step. These mechanisms share matrix structure but establish different algorithm-specific conditions.

---

## 11. Conclusion

The main result is a composable energy-relaxation contract with curvature-aware safeguards. When modules and couplings report truthful local stiffness, the coordinator can compose those reports into stability-controlled, precision-aware relaxation without choosing a separate step rule for each interaction. The tests show both sides of that claim. Tight quadratic bounds change the outcome, while conservative mixed-regime bounds can trade speed for margin. The framework depends on accurate curvature contracts, explicit guards, and measured regime boundaries.

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

MIT License

© 2025 Oscar Goldman
Status: research prototype
---

### Notes on implementation

- Codebase: Python, protocol-based architecture using NumPy for the current implementation.
- Vectorization: runtime coupling caches amortize sparse gradient and energy passes for supported coupling families.
- Observability: callback-driven telemetry (`RelaxationTracker` and `EnergyBudgetTracker`) records energy descent, stability margins, precision summaries, and adapter spend.
- Precision layer: implemented; diagonal curvature aggregates module and coupling curvature, supports per-coordinate stiffness-based steps (`use_stiffness_updates`), and supports precision-scaled PSON with re-projection after weighting.
- Solver layer: one explicit mode dispatches to guarded gradient, proximal, or ADMM-like relaxation. The split solvers restore rejected states and report mode-specific acceptance and residual metrics.
- Configuration layer: immutable grouped configuration is the preferred entry point; flat coordinator keyword arguments remain available for compatibility and focused ablations.

