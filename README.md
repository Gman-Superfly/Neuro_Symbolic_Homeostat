# The Neuro-Symbolic Homeostat
### Constraints as Energy. Inference as Relaxation.

> A composable energy-relaxation contract for constraint-aware coordination across modular models.

---

## Context and origins

**Note:** This repository is a focused validation lab for the **Neuro-Symbolic Homeostat**, a specific component of the broader **Datamutant** library.

It serves as a validation lab for ideas from *Complexity from Constraints*. The core mechanisms in this repository include precision-scaled orthogonal noise, counterfactual gate-benefit couplings (CGBCs, nicknamed wormhole couplings), and stability projectors. The repository documents synthetic demos and implementation details for these mechanisms, and it feeds design and validation results back into the broader Datamutant ecosystem.

---

## Proof status

This repository separates mechanism validity from empirical benefit. A mechanism can be mathematically well-defined without showing that it reduces step count, lowers final energy, or improves behavior across problem classes.

What is analytically supported:

- **CGBC** gives a nonzero gate gradient at \(\eta_{\text{gate}} = 0\) when the caller supplies a nonzero benefit value and the effective coupling coefficient is nonzero. The coordinator snapshots the value as a finite float for the complete solver call; CGBC consumes this credit signal and does not derive it.
- **PSON** re-projects after inverse-precision weighting, so the returned perturbation has zero first-order energy change above the numerical gradient threshold. The final box-feasibility operation scales the complete vector uniformly and preserves this tangency.
- **Ordinary projected gradient descent** contracts in the Euclidean norm on an SPD quadratic when a valid raw Hessian bound gives \(0<\alpha<2/L_H\).
- **Diagonal-preconditioned projected relaxation** contracts in the \(P\)-norm on an SPD quadratic when the guard bounds \(P^{-1/2}HP^{-1/2}\) using the exact positive diagonal \(P\) executed by the update and keeps \(0<\alpha<2/L_P\).
- **Stiffness updates** are weighted Jacobi under quadratic/SPD assumptions when \(P=\operatorname{diag}(H)\); classical Jacobi is the case \(\alpha=1\). Gaussian BP remains related literature context.

What remains empirical to be fully explored by other repos:

- Whether these mechanisms reduce step count.
- Whether final energy decreases more than baseline methods across problem classes.
- Whether CGBC helps when benefit estimates are noisy or wrong.
- Whether the combined system beats simpler baselines.

The current demos test specific synthetic cases.

The optimization mathematics used here is standard: diagonal preconditioning, Gershgorin row-sum bounds, projected gradient descent, and weighted Jacobi. The repository-specific design is the composable curvature contract. Modules expose local stiffness, couplings expose curvature bounds, and the coordinator composes those reports into a raw Hessian bound and a geometry-matched update bound. Randomized SPD tests target the implemented matrix \(I-\alpha P^{-1}H\), including box projection, scale invariance, and a small-precision counterexample. Built-in hinges use worst-case curvature across active-set crossing. Nonlinear, state-dependent, and custom terms require segment-valid bounds or projected Armijo.

---

## The big idea

This repository studies a specific coordination problem:
*   Neural scorers can produce useful continuous outputs while violating explicit constraints.
*   Symbolic rule systems can enforce constraints but often require crisp inputs and discrete choices.

The **Neuro-Symbolic Homeostat** treats some logical constraints as energy terms.
Instead of running a discrete proof procedure, we define an energy function where violating a rule increases cost. The system then relaxes toward lower energy, seeking a state that satisfies more constraints while staying close to the data.

## Intended use

The strongest intended use is a **constraint-aware coordination layer around existing models**, especially when several continuous modules must satisfy shared rules without being rebuilt as one monolithic model.

### Most credible near-term uses

- **Structured output correction:** Adjust probabilities, scores, allocations, or confidence values under constraints such as mutual exclusion, ordering, budgets, and consistency.
- **Neuro-symbolic inference:** Express logical preferences as differentiable energy terms and relax neural outputs toward a more consistent state.
- **Modular model coordination:** Combine independently developed scorers, gates, retrieval modules, or specialist models through a shared energy contract.
- **Adaptive routing and gating:** Use CGBC to send a signal to a closed gate when the caller can estimate the downstream benefit of opening it.
- **Resource allocation:** Coordinate bounded quantities under coupling constraints, including compute budgets, mixture weights, scheduling priorities, and capacity assignments.
- **Iterative probabilistic inference:** Apply the curvature contract to Gaussian-like or locally quadratic systems where diagonal stiffness and coupling curvature can be estimated.
- **Constrained control prototypes:** Coordinate bounded control variables with monotone acceptance and stability guards. Safety-critical use requires domain-specific analysis, proofs, and testing beyond the evidence in this repository.

### Technical distinction

The repository-specific contribution is the **composable energy-relaxation contract**:

- Modules report local energy, gradients, and available curvature.
- Couplings report interaction energy and curvature bounds.
- The coordinator composes those reports into guarded updates.
- PSON biases exploration toward lower-curvature coordinates while remaining tangent to the current energy level set to first order after its final projection.
- CGBC applies caller-supplied frozen credit to inactive gates.
- Proximal and ADMM-like paths handle the supported composite terms.
- Rejection and restoration prevent an unsuccessful proposal from replacing the last accepted state.

This contract targets systems whose components are owned or trained separately but must behave coherently when assembled.

The local curvature paths remain separate. `SupportsPrecision.curvature` feeds the diagonal cache used for \(P\) and precision-aware noise. The local terms in raw and normalized Gershgorin bounds are independently finite-differenced from local gradients. Supported coupling curvature feeds both paths; built-in hinges use starting-state curvature in the cache and worst-case cross-region curvature in the step bound. The raw Hessian bound is not reconstructed from the precision cache.

### Longer-term research directions

The following applications are prospective and have not been validated by the current synthetic experiments:

- Multi-agent or multi-model consensus over shared continuous state.
- Constraint-aware retrieval and reranking.
- Planning systems with soft rules and bounded decision variables.
- Scientific inverse problems assembled from heterogeneous factors.
- Continual systems where modules or constraints can be added without retraining every component.
- Test-time adaptation where model weights remain fixed and only interpretable order parameters relax.

The current evidence supports mechanism behavior on synthetic problem families. Real-model accuracy, planning quality, control performance, and operational safety remain evaluation targets.

### The philosophy: descending is for boomers on escalators, Surf like a chad

Standard optimization asks: *"Given a fixed energy function, how do I find the minimum?"*
**Our approach: shape and move**

We do not rely on a single optimizer. The system combines objective shaping, stability projection, preconditioned relaxation, and tangent exploration. Some parts change the effective objective, while others change the update geometry or trajectory.

See **[Objective shaping](docs/README_MOVING_THE_LANDSCAPE.md)** for the unified strategy.

---

### Core concepts (simply explained)

#### 1. Relaxation
Modules expose variables called order parameters. Couplings connect those variables through energy terms. When new data arrives, the coordinator updates the order parameters to reduce total energy while respecting the configured guards.

#### 2. Precision-scaled orthogonal noise (PSON)
*The Problem:* Exploration can help avoid stalls, but random noise can add an uphill component against the gradient.
*The Solution:* **Tangent Noise**. PSON projects a draw against the gradient, applies inverse-precision weights, and projects again because weighting generally breaks the first orthogonality condition. It normalizes the result and uses one uniform scale to keep the full perturbation inside the box. Above the numerical gradient threshold, the realized noise remains tangent to first order. Near a stationary point, the projection falls back to unconstrained exploration because the gradient does not define a reliable normal. Tangency does not imply zero second-order curvature.

#### 3. The curvature-based stability guard
*The Problem:* If components interact too strongly, feedback loops can cause divergence or oscillation.
*The Solution:* A stability guard that bounds the geometry of the executed update. Ordinary updates use a raw Hessian row-sum bound. Preconditioned updates use a normalized bound on \(P^{-1/2}HP^{-1/2}\), with the same positive diagonal \(P\) used to divide the gradient. This gives a box-projected contraction theorem in the \(P\)-norm for SPD quadratics. State-dependent or custom curvature must cover the realized proposal segment; projected Armijo supplies a checked fallback when that contract is unavailable.

Analytic derivatives are used when components provide them. Every fallback derivative is evaluated with a box-aware second-order stencil, including isolated local objectives in partially coupled graphs. These stencils reproduce quadratic gradients at the box edges up to floating-point error; for general nonlinear energies they remain numerical approximations and do not enlarge the quadratic theorem.

The logged `step_cap_slack` is \((2/L_{\text{update}})-\alpha_{\text{used}}\). It is distance from the uncushioned step boundary, not a measured contraction factor. Older `contraction_margin` names remain deprecated compatibility aliases.

The optional `SmallGainWeightAdapter` is separate. It reallocates coupling-family weights under an estimated curvature-spend budget. The included benchmarks do not establish a task-quality advantage for that allocator.

#### 4. Counterfactual gate-benefit coupling (CGBC)
*The Problem:* In sparse logic gates, if a gate is closed, the local gradient can be zero. The gate may not receive the downstream signal that opening it would reduce energy.
*The Solution:* A **counterfactual gate-benefit coupling (CGBC)**, nicknamed a wormhole coupling. We add a coupling term that applies a gate gradient proportional to a caller-supplied benefit value. Before solver dispatch, the coordinator converts that value to a finite float inside a read-only top-level constraint snapshot. The value remains fixed for the complete solver call, and the caller's mapping is restored afterward. This lets closed gates receive a non-local update signal. The benefit estimator remains outside CGBC.

Plain CGBC and damped CGBC with power \(p=1\) are linear in the gate and contribute zero Hessian curvature. A damped term with \(p\ge2\) reports the exact box-wide absolute diagonal bound \(|a|p(p-1)\) for energy \(-a\eta^p\). This magnitude bound does not imply positive curvature; the term is concave when \(a>0\), and contraction still requires an SPD total Hessian. For nonzero \(a\) and \(1<p<2\), the second derivative is unbounded at zero, so a fixed-step guarded run fails closed unless projected Armijo is enabled. A zero frozen coefficient removes the term and gives a zero curvature report. Powers below one are rejected.

#### 5. Distributed linear algebra processor
*The Insight:* This system does not run symbolic proofs step by step. It uses matrix updates, including projected weighted-Jacobi iterations on quadratic blocks. The update becomes classical Jacobi when \(P=\operatorname{diag}(H)\), \(\alpha=1\), the epsilon floor is inactive, and box clipping does not change the proposal.

---

## Applications that fit the current system

The current implementation is best matched to sparse coordination problems whose state is a collection of continuous order parameters in \([0,1]\). The application must express its local preferences and cross-variable constraints as energy terms. The coordinator can then propose bounded updates, reject proposals that raise the configured acceptance objective, and expose the resulting stability and acceptance telemetry.

### Strongest current fits

- **Constraint correction after another model produces scores.** Examples include sum-to-one correction, mutual exclusion, monotonic ordering, consistency between related confidence values, and bounded priority or resource allocation. The repository includes a small constraint-correction demonstration; it does not yet provide task-level results on a deployed model.
- **Coordination between modular AI components.** Perception, memory, planning, verification, or policy modules can expose bounded confidence or activation variables. Sparse energy couplings can reconcile incompatible outputs without requiring the coordinator to retrain those modules.
- **Gating and routing.** Order parameters can control the activation of tools, experts, memories, or reasoning branches. Plain CGBC can transmit a caller-supplied downstream-benefit estimate to a closed gate. The benefit estimator remains external, and a wrong-sign estimate produces the wrong update direction.
- **Sparse quadratic control and estimation.** Consensus, smoothing, distributed calibration, coupled set-point adjustment, and soft constraint enforcement fit the strongest theoretical regime when the assembled objective is an SPD quadratic. In that regime, the ordinary and diagonally preconditioned projected updates have the contraction guarantees stated in the paper and stability documentation.
- **Anisotropic exploration.** PSON is relevant when the energy has stiff and soft directions and isotropic noise spends too much curvature budget in expensive directions. The recorded synthetic experiments show lower realized full-Hessian noise cost across seven generated families. This is mechanism evidence, not evidence of improved accuracy or planning quality on a real task.
- **Repeated reconciliation under changing inputs.** A caller can update measurements, constraints, or benefit estimates between solver calls and relax a small state vector again. Values that define the objective are frozen within each call so a proposal is evaluated against one objective version.

### Current boundaries

The present evidence does not establish the system as an end-to-end neural-network trainer, a generic black-box optimizer, an exact discrete SAT solver, or a symbolic theorem prover. It also does not establish physical safety for safety-critical control: accepted-energy monotonicity concerns the configured mathematical objective, not every property of an external environment.

Unbounded variables require a different projection geometry. General nonlinear or nonconvex objectives require segment-valid curvature reports or projected Armijo backtracking, and may still converge only to a local stationary state. CGBC does not derive counterfactual benefit by itself. Large-scale and real-model performance remain evaluation targets.

The most direct current deployment pattern is therefore a **constraint-aware correction and coordination layer** after existing modules: consume their bounded outputs, apply explicit local and pairwise relationships, and return a lower-energy accepted state together with rejection and stability telemetry.

---

## Documentation map

We believe in documenting *why*, for our sanity.

### Theory and mechanisms
*   **[Thesis](The_Neuro_Symbolic_Homeostat_Paper_V1.md)**: Draft paper describing the current system and scope limits.
*   **[Moving the Landscape](docs/README_MOVING_THE_LANDSCAPE.md)**: High-level overview of how the four mechanisms combine.
*   **[Tangent Noise / PSON](docs/README_TANGENT_NOISE_PSON.md)**: How tangent noise controls first-order energy change.
*   **[Stability Guarantees](docs/STABILITY_GUARANTEES.md)**: The control theory math and scoped assumptions behind the guards.
*   **[Counterfactual gate-benefit coupling](docs/README_WORMHOLE.md)**: Non-local gate updates from caller-supplied benefit estimates. “Wormhole” is the implementation nickname.
*   **[Stiffness Updates](docs/README_STIFFNESS_UPDATES.md)**: Diagonal preconditioning, normalized bounds, and the weighted-Jacobi relationship.

### Implementation details
*   **[Auto Scheduling](docs/README_AUTO_SCHEDULING.md)**: How the system tunes its own noise and step sizes.
*   **[Observability](docs/README_OBSERVABILITY.md)**: How we measure what's happening (metrics, logs).
*   **[Safe Defaults](docs/README_SAFE_DEFAULTS.md)**: The recommended configuration for reproducible results.
*   **[Proximal and ADMM-like solvers](docs/README_OPERATOR_SPLITTING.md)**: Explicit solver selection, residual telemetry, and the current conformance boundary.
*   **[Reproducibility](docs/REPRODUCIBILITY.md)**: Local verification, artifact regeneration, figures, and manifest fields.
*   **[Distributed linear algebra processor](docs/DISTRIBUTED_LINEAR_ALGEBRA_PROCESSOR.md)**: Implementation specification for extending the energy, gradient, curvature, and acceptance contracts across versioned worker processes.

---

## Experiments and demos

We assert early and fail hard, no mercy. 

### Ready-to-Run (Windows PowerShell)

**1. Counterfactual gate-benefit coupling (CGBC), wormhole nickname**
Demonstrates how a caller-supplied benefit estimate creates a non-local gradient on a closed gate.
```powershell
python -m experiments.demo_wormhole
```

**2. Metric-consistent tangent projection**
Shows how an SPD metric changes the projection direction while preserving zero first-order energy change (noise dot ordinary gradient ≈ 0).
```powershell
python -m experiments.demo_metric_orthogonal
```

**3. Proximal relaxation**
Runs the maintained proximal solver on a small graph with quadratic and asymmetric hinge couplings.
```powershell
python -m experiments.demo_operator_splitting
```

**4. Small-Gain sweep**
Sweeps interaction strengths to test conservative stability boundaries in the included scenarios.
```powershell
python -m experiments.sweep_smallgain --quick
```

**5. Constraint correction**
Shows a small System-1 output before and after stability-guarded relaxation against explicit constraints: sum-to-one, mutual exclusion, and monotonicity.
```powershell
python -m experiments.demo_constraint_correction
```

**6. PSON problem-family ablation**
Compares isotropic, orthogonal, and precision-orthogonal noise on seven paired generated problem families. The summary uses the realized initial-state box-feasible full-Hessian cost \(\delta^\top H\delta\). The raw CSV records requested and realized costs, uniform box-scale statistics, and the diagonal curvature proxies separately.
```powershell
python -m experiments.ablate_pson_noise --trials 30 --steps 80 --noise-cost-samples 32 --bootstrap-samples 10000
```

This remains synthetic mechanism validation. Real-model evaluation is deferred.

**7. Curvature-contract audit**
Checks sampled finite-difference curvature against module and coupling reports.
```powershell
python -m experiments.audit_curvature_contract --samples 32 --strict
```

**8. Controlled nonconvex escape**
Measures paired escape rates in a documented anisotropic double-well construction.
```powershell
python -m experiments.benchmark_pson_escape --trials 200 --steps 40 --bootstrap-samples 10000
```

---

## Testing

Install the core and development dependencies:

```powershell
uv sync
```

The plotting scripts use an optional dependency group:

```powershell
uv sync --extra plots
```

Use this command as the canonical local test run:

```powershell
uv run -m pytest tests -v
```

The pytest defaults are recorded in `pyproject.toml`.

The solver conformance suite compares gradient, proximal, and ADMM-like modes against a closed-form convex optimum and checks rejected-state restoration:

```powershell
uv run python -m pytest -q tests\test_solver_conformance.py
```

Run the complete local publication check without regenerating long experiments:

```powershell
.\scripts\verify_publication.ps1
```

Use `-Regenerate` to rerun the full recorded experiment set before verification.

---

## Citation

If you use this repository in research, we love you.

**Authors:** Oscar Goldman, Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業).

---

Built by the Shogu Research Group.
