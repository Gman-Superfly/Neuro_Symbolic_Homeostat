# Counterfactual gate-benefit coupling (CGBC)

Status: implemented as `GateBenefitCoupling`

Nickname: wormhole coupling

Scope: applying a caller-supplied, frozen benefit signal as a gate force during one relaxation run

## Mechanism

CGBC adds a linear gate term to the configured energy:

\[
F_{\mathrm{CGBC}}(\eta_{\mathrm{gate}})
=-w\,\eta_{\mathrm{gate}}\,\Delta_{\mathrm{benefit}}.
\]

Here, \(w\ge 0\) is the coupling weight, \(\eta_{\mathrm{gate}}\in[0,1]\) is the gate state, and \(\Delta_{\mathrm{benefit}}\) is a scalar supplied through `constraints[delta_key]`. The term is a linear surrogate for the caller's benefit signal. The coupling does not calculate, identify, or validate that signal.

During a relaxation run, \(\Delta_{\mathrm{benefit}}\) must remain frozen. Under that condition, the gate derivative is

\[
\frac{\partial F_{\mathrm{CGBC}}}{\partial\eta_{\mathrm{gate}}}
=-w\,\Delta_{\mathrm{benefit}}.
\]

The derivative does not depend on the current gate value. A positive supplied benefit produces a negative energy gradient, so a descent update pushes the gate upward. A negative supplied benefit pushes it downward. A zero value contributes no force.

The frozen linear term has zero Hessian. It therefore changes the gradient and the location of the augmented objective's optimum without adding curvature to that objective.

## What the coupling consumes

`GateBenefitCoupling` implements the two-coordinate coupling interface, but its energy and derivative use only the first coordinate and the named constraint value. The second coordinate is present for interface compatibility and receives a zero derivative from this term.

In code, the mechanism is:

```python
delta = float(constraints.get(self.delta_key, 0.0))
energy = -self.weight * float(eta_gate) * delta
gate_gradient = -self.weight * delta
domain_gradient = 0.0
```

The name counterfactual describes the intended semantics of the supplied scalar. It does not make the coupling a counterfactual estimator. If the caller supplies a heuristic, then CGBC applies that heuristic as a linear force.

If `delta_key` is absent, `GateBenefitCoupling` reads a default value of zero and contributes no force. Callers should test the configured key explicitly because a misspelling otherwise behaves like a valid zero-benefit input.

## Frozen-benefit run contract

The energy guard compares proposals against one configured objective. A changing benefit value between evaluations would invalidate a same-objective interpretation of the energy trace, so `relax_etas` freezes external constraints at the run boundary.

The coordinator makes a top-level constraint copy, separately copies and protects any `Mapping` supplied as `term_weights`, and converts every plain and damped gate-benefit delta to a finite float. A nonfinite delta is rejected before optimization starts. The coordinator installs the snapshot as a read-only mapping for the full run and restores the caller's original mapping when relaxation returns or raises. Mutation of the external benefit after a run starts therefore cannot change later energy, gradient, or acceptance evaluations within that run.

If an application needs to refresh the benefit estimate, it must start a new relaxation run. The next call takes a new snapshot and can consume the updated scalar. This boundary separates benefit estimation from gate relaxation and makes each run's energy trace refer to one benefit value.

## Possible benefit sources

The repository does not implement a general benefit estimator. An application may supply a value from:

- a learned predictor with its own validation and calibration record,
- a downstream loss difference measured by a separate evaluation,
- a bounded planning or lookahead rollout,
- external supervision, or
- a domain heuristic whose failure modes are documented.

These sources are alternatives, not equivalent guarantees. Each source needs its own sign convention, scale, provenance, and error analysis. The CGBC coupling only consumes the resulting scalar.

The demonstration uses a hard-coded value of `0.3` to expose the derivative at a closed gate. It does not show the system discovering a benefit, performing a counterfactual rollout, or learning an estimator.

## Relation to Equilibrium Propagation

Equilibrium Propagation introduces a nudged phase and obtains a learning signal from the contrast between free and nudged equilibria under its stated assumptions (Scellier and Bengio, 2017). CGBC shares the limited structural idea that an externally specified term can bias energy dynamics.

The implemented CGBC path does not run free and nudged phases, take their contrast, or derive parameter gradients by the Equilibrium Propagation procedure. It inserts one caller-supplied scalar into a typed linear gate term and relaxes the resulting augmented energy. The analogy is useful for recognizing an external nudge. It does not transfer Equilibrium Propagation's derivation or guarantees to CGBC.

Reference: Scellier, B., and Bengio, Y. (2017). Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation. *Frontiers in Computational Neuroscience*, 11, 24.

## Acceptance and semantic risk

The coordinator evaluates CGBC as part of the configured energy. Under down-only acceptance, an accepted proposal does not increase that augmented energy. This guard checks optimization consistency with the supplied scalar.

The guard cannot establish that \(\Delta_{\mathrm{benefit}}\) is correct or that opening the gate improves an external task. A wrong-sign or badly scaled value can make the surrogate energy favor an externally harmful action. The augmented objective can decrease exactly as implemented while the unrepresented task objective gets worse.

Applications must validate benefit semantics outside the coupling. Useful checks include provenance, sign tests, magnitude bounds, held-out calibration, and direct measurement of the external outcome that the benefit is intended to represent.

## Damped variant

`DampedGateBenefitCoupling` uses

\[
F_{\mathrm{damped}}
=-w\,d\,\eta_{\mathrm{gate}}^p\,s(\Delta)\,\Delta.
\]

Here, $d$ is `damping`, $p$ is `eta_power`, and \(s(\Delta)\) selects the configured positive or negative scale. Define \(a=w d s(\Delta)\Delta\), so the gate term is \(-a\eta_{\mathrm{gate}}^p\). This variant supports asymmetric scaling and nonlinear gate response.

The gate-independent closed-gate force applies to the linear case \(p=1\), whose Hessian contribution is zero. For \(p>1\), the derivative depends on \(\eta_{\mathrm{gate}}\), and the implementation returns zero at \(\eta_{\mathrm{gate}}\le 0\). The nonlinear variant therefore does not inherit the same closed-gate force.

For \(p\ge 2\), the implementation reports the finite box-wide diagonal curvature bound

\[
L_{\mathrm{gate}}=|a|p(p-1).
\]

Here, \(L_{\mathrm{gate}}\) bounds the absolute second derivative on \(\eta_{\mathrm{gate}}\in[0,1]\). The bound supplies curvature accounting for the gradient guard; it does not establish global convergence of a mixed objective.

For \(1<p<2\) and nonzero \(a\), the second derivative diverges as the gate approaches zero. The coupling reports an infinite box-wide bound, and a fixed-step guarded run fails closed unless projected line search is enabled. When the frozen coefficient \(a\) is zero, the complete gate term is zero and the coupling reports zero curvature. Construction rejects \(p<1\) because that range is incompatible with the closed unit-box contract used here.

## Usage

```python
from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling

# Compute this outside CGBC, then keep it fixed for this relaxation run.
benefit = estimate_domain_improvement(inputs)

coordinator = EnergyCoordinator(
    modules=[gate_module, domain_module],
    couplings=[
        (0, 1, GateBenefitCoupling(weight=1.0, delta_key="benefit")),
    ],
    constraints={"benefit": benefit},
)

result = coordinator.relax_etas([0.0, 0.5], steps=50)
```

The call applies the frozen scalar as part of the relaxation objective. The quality of `estimate_domain_improvement` remains outside the coupling contract.

## Demonstration and tests

Run the demonstration:

```powershell
uv run python -m experiments.demo_wormhole
```

The two demonstration scenarios use a hard-coded positive benefit to show the gate-force mechanism. They are not a controlled task-performance benchmark, and their final gate values can also depend on the configured stochastic noise.

Run the focused tests:

```powershell
uv run python -m pytest -q tests/test_wormhole_gradient_independence.py tests/test_damped_gate_benefit_coupling.py tests/test_coordinator_admm_gate_benefit.py
```

The tests cover the linear derivative's independence from gate state, sign behavior, the damped variant, and solver integration. They do not validate an application-specific benefit estimator.

## Boundaries

- CGBC applies a supplied scalar. It does not derive a counterfactual estimate.
- The benefit must remain frozen within a relaxation run.
- The linear term provides a closed-gate force when the supplied value is nonzero.
- A decrease in the augmented energy does not validate the external meaning of the benefit.
- The wormhole nickname refers to a non-local software coupling. It is not a physical tunneling or causality claim.
- Current repository evidence covers formula, sign, and synthetic integration tests. Broader planning, sequence, sparse-compute, and task-benefit claims require separate experiments.

## Related documentation

- `core/couplings.py` contains `GateBenefitCoupling` and `DampedGateBenefitCoupling`.
- `experiments/demo_wormhole.py` contains the hard-coded demonstration.
- `docs/STABILITY_GUARANTEES.md` describes the gradient-step assumptions and acceptance guard.
- `docs/README_OPERATOR_SPLITTING.md` describes the experimental proximal and ADMM-like integrations.
