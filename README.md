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

- **CGBC** gives a nonzero gate gradient at \(\eta_{\text{gate}} = 0\) when \(\Delta_{\text{benefit}} \neq 0\).
- **PSON** has zero first-order energy change for perturbations orthogonal to the gradient.
- **The curvature-based step cap** gives a contraction condition under linear/SPD assumptions when the Gershgorin bound is valid.
- **Stiffness updates** match the Jacobi trajectory under quadratic/SPD assumptions; Gaussian BP remains related literature context.

What remains empirical to be fully explored by other repos:

- Whether these mechanisms reduce step count.
- Whether final energy decreases more than baseline methods across problem classes.
- Whether CGBC helps when benefit estimates are noisy or wrong.
- Whether the combined system beats simpler baselines.

The current demos test specific synthetic cases.

The optimization mathematics used here is standard: diagonal preconditioning, Gershgorin row-sum bounds, and the gradient descent condition \(\alpha < 2/L\). The repository-specific design is the composable curvature contract: modules expose local stiffness, couplings expose curvature bounds, and the coordinator composes those reports into precision-aware updates and stability caps. Current tests show both sides of this contract: in a tight quadratic regime, curvature awareness lets the system converge where plain gradient descent stalls above \(2/L\); in a mixed constraint regime, the conservative cap can trade speed for margin while preconditioning still converges.

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
- PSON explores low-curvature directions tangent to the current energy level set to first order.
- CGBC provides counterfactual credit to inactive gates from caller-supplied benefit estimates.
- Proximal and ADMM-like paths handle the supported composite terms.
- Rejection and restoration prevent an unsuccessful proposal from replacing the last accepted state.

This contract targets systems whose components are owned or trained separately but must behave coherently when assembled.

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
*The Solution:* **Tangent Noise**. We inject noise in directions that do not increase energy to first order. We scale this exploration with precision so uncertain coordinates receive larger perturbations than stiff coordinates.

#### 3. The curvature-based stability guard
*The Problem:* If components interact too strongly, feedback loops can cause divergence or oscillation.
*The Solution:* A stability guard that caps interactions using Gershgorin-based bounds and step-size checks. In linear and SPD regimes, this gives a conservative contraction condition. In mixed regimes, it acts as a practical guard.

The optional `SmallGainWeightAdapter` is separate. It reallocates coupling-family weights under an estimated curvature-spend budget. The included benchmarks do not establish a task-quality advantage for that allocator.

#### 4. Counterfactual gate-benefit coupling (CGBC)
*The Problem:* In sparse logic gates, if a gate is closed, the local gradient can be zero. The gate may not receive the downstream signal that opening it would reduce energy.
*The Solution:* A **counterfactual gate-benefit coupling (CGBC)**, nicknamed a wormhole coupling. We add a coupling term that applies a gradient on a gate proportional to a caller-supplied estimate of downstream benefit. This lets closed gates receive a non-local update signal.

#### 5. Distributed linear algebra processor
*The Insight:* This system does not run symbolic proofs step by step. It uses matrix updates, including Jacobi-style iterations on quadratic blocks, to perform relaxation.

---

## Documentation map

We believe in documenting *why*, for our sanity.

### Theory and mechanisms
*   **[Thesis](The_Neuro_Symbolic_Homeostat_Paper_V1.md)**: Draft paper describing the current system and scope limits.
*   **[Moving the Landscape](docs/README_MOVING_THE_LANDSCAPE.md)**: High-level overview of how the four mechanisms combine.
*   **[Tangent Noise / PSON](docs/README_TANGENT_NOISE_PSON.md)**: How tangent noise controls first-order energy change.
*   **[Stability Guarantees](docs/STABILITY_GUARANTEES.md)**: The control theory math and scoped assumptions behind the guards.
*   **[Counterfactual gate-benefit coupling](docs/README_WORMHOLE.md)**: Non-local gate updates from caller-supplied benefit estimates. “Wormhole” is the implementation nickname.
*   **[Stiffness Updates](docs/README_STIFFNESS_UPDATES.md)**: How precision acts as stiffness (Newton-like scaling).

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
Compares isotropic, orthogonal, and precision-orthogonal noise on seven paired generated problem families and writes hierarchical bootstrap summaries.
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
