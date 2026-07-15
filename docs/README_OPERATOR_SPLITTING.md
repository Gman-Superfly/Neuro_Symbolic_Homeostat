# Proximal and ADMM-like solvers

Status: experimental solver paths with convex conformance tests

Scope: bounded energy relaxation for the coupling families implemented in this repository

## Solver selection

`SolverConfig` selects exactly one relaxation path:

- `gradient`
- `proximal`
- `admm`

The coordinator dispatches to the selected path before entering the primary gradient loop. The modes are alternatives rather than layers applied to the same iteration.

```python
from core.coordinator import EnergyCoordinator
from core.solver_config import SolverConfig

coordinator = EnergyCoordinator(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    solver=SolverConfig.proximal_solver(
        steps=50,
        tau=0.05,
        block_mode="pairwise",
    ),
)

result = coordinator.relax_etas(initial_state)
```

The ADMM-like path uses the same dispatch point:

```python
solver = SolverConfig.admm_solver(
    steps=100,
    rho=1.0,
    step_size=0.02,
    gate_prox=True,
    gate_damping=0.5,
)
```

The maintained implementations live in `core/solvers/proximal.py` and `core/solvers/admm.py`. The old direct methods remain as deprecation wrappers for existing callers.

For a split solver, the iteration budget comes from its mode-specific `SolverConfig`. The `steps` argument passed directly to `relax_etas` controls only the primary gradient path and is bypassed by split-solver dispatch.

## Proximal path

Each proximal iteration applies projected local gradient steps and then the supported coupling operators:

- `prox_quadratic_pair` handles `QuadraticCoupling`.
- `prox_asym_hinge_pair` handles directed and asymmetric squared hinges.
- `prox_linear_gate` handles the local gate-benefit gradient as a linear proximal term.
- `pairwise` and `star` block schedules are available.

Unsupported coupling types are left unchanged by the coupling-operator stage. Their local modules can still receive their own local projected gradient steps. Callers must verify that every coupling needed by an application has a supported update.

The parameter `tau` controls both the local gradient step and the coupling proximal operators. The implementation does not derive `tau` from the primary gradient path's Lipschitz estimator.

## ADMM-like path

The ADMM-like implementation creates auxiliary and scaled dual variables for quadratic and hinge edges. One iteration performs:

1. an auxiliary update for each supported edge,
2. a projected primal gradient update,
3. an optional damped gate-benefit proximal-linear update, and
4. a scaled dual update.

Other couplings contribute to the primal gradient through their analytic derivative when available or through the solver's finite-difference fallback. Gate-benefit terms use either their ordinary gradient or the optional damped proximal-linear update.

The local proximal fallback and the local and coupling ADMM fallbacks share the primary solver's box-aware finite-difference primitive. It uses centered second-order differences in the interior and three-point second-order one-sided differences at the boundaries, so a boundary point is never probed outside \([0,1]\) or treated as having zero gradient solely because an upper probe was clipped.

`last_solver_metrics()` reports attempted, accepted, and rejected steps plus primal and dual residual histories. These residuals describe the implemented splitting constraints. Finite or decreasing residuals in one test case do not prove convergence for every configured objective.

## Acceptance and restoration

Both split paths evaluate the full configured energy after each candidate iteration. If a proposal raises energy above `monotonic_energy_tol`, then the solver restores the previous accepted state, records one rejection, and stops.

This rule establishes accepted-state monotonicity for the energy evaluated by the solver. It does not ensure that every iteration is accepted, that the solver makes progress, or that a nonconvex problem converges to a global or local optimum.

## Relationship to the gradient stability theorem

The proximal and ADMM-like paths do not repair, extend, or imply the spectral theorem for the primary gradient update. The gradient theorem concerns a map of the form

\[
x^+ = x-\alpha P^{-1}\nabla F(x)
\]

under its stated quadratic/SPD and normalized-Hessian bound. Here, \(\alpha\) is the gradient step, $P$ is the positive diagonal preconditioner, and $F$ is the configured energy. The split solvers use different update maps and separate parameters.

Because dispatch occurs before the primary gradient loop, gradient-path options such as `stability_guard`, `auto_step_from_lipschitz`, and PSON do not certify or control a proximal or ADMM-like iteration. A configuration may contain those fields, but selecting a split solver bypasses their gradient-loop behavior.

The split paths currently rely on their mode-specific parameter checks, full-energy rejection, state restoration, and focused conformance tests. A spectral or operator-theoretic convergence result for either split map would require a separate derivation matched to that implementation.

## Verification scope

`tests/test_solver_conformance.py` applies one contract to gradient, proximal, and ADMM-like modes:

- finite order parameters within the closed interval \([0,1]\),
- non-increasing accepted energy,
- agreement with a closed-form two-variable convex optimum within mode-specific tolerances,
- restoration after a deliberately oversized rejected step,
- inward movement from a box boundary for finite-difference-only local terms, plus an ADMM custom-coupling boundary check,
- finite ADMM residual histories that fall below \(10^{-3}\) on the convex reference case, and
- early rejection of invalid mode-specific configuration.

The gate-benefit tests cover the ADMM-like gate update separately. These tests support the implemented behavior on the recorded synthetic cases. They do not establish broad nonconvex convergence, large sparse graph behavior, or task-level benefit.

Run the focused checks:

```powershell
uv run python -m pytest -q tests/test_solver_conformance.py tests/test_admm_damped_gate_benefit.py tests/test_coordinator_admm_gate_benefit.py
```

Run the proximal demonstration:

```powershell
uv run python -m experiments.demo_operator_splitting
```

The demonstration reports energy before and after one small synthetic proximal run. It is an execution example, not comparative solver evidence.
