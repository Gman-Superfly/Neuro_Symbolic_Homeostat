# Proximal and ADMM-like solvers

Status: experimental solver paths with convex conformance tests

Scope: bounded energy relaxation for the coupling families implemented in this repository

## Solver selection

`SolverConfig` selects exactly one relaxation path. The constructor no longer accepts independent proximal and ADMM booleans, so callers cannot enable contradictory modes.

```python
from core.coordinator import EnergyCoordinator
from core.solver_config import SolverConfig

coordinator = EnergyCoordinator(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    solver=SolverConfig.proximal_solver(steps=50, tau=0.05),
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

## Implemented updates

The proximal path applies local projected gradient steps followed by supported coupling operators:

- `prox_quadratic_pair` handles `QuadraticCoupling`.
- `prox_asym_hinge_pair` handles directed and asymmetric squared hinges.
- `prox_linear_gate` handles the local linearization used for gate-benefit couplings.
- Pairwise and star-block schedules are available.

The ADMM-like path introduces auxiliary and scaled dual variables for quadratic and hinge couplings. It performs an auxiliary update, a projected primal gradient update, and a dual update. Gate-benefit terms use either their gradient or a damped proximal-linear update. `last_solver_metrics()` reports attempted, accepted, and rejected steps plus primal and dual residual histories.

Both paths evaluate the full configured energy after each proposal. An uphill proposal is rejected, the previous accepted state is restored, and relaxation stops. This guard establishes accepted-state monotonicity for the tested deterministic runs; it does not prove convergence for every nonconvex module or coupling combination.

## Verification

`tests/test_solver_conformance.py` applies one contract to gradient, proximal, and ADMM-like modes:

- finite order parameters within the closed interval $[0,1]$,
- non-increasing accepted energy,
- agreement with a closed-form two-variable convex optimum,
- restoration after a deliberately oversized rejected step,
- finite ADMM residual histories that contract below $10^{-3}$ on the convex reference case, and
- early rejection of invalid mode-specific configuration.

The gate-benefit tests cover the ADMM-like gate update separately. These tests support the implemented behavior on the recorded synthetic cases. Broader nonconvex convergence, large sparse graph behavior, and task-level benefit remain untested.

Run the focused checks with:

```powershell
python -m pytest -q tests\test_solver_conformance.py tests\test_admm_damped_gate_benefit.py tests\test_coordinator_admm_gate_benefit.py
```

The proximal demonstration remains available through `python -m experiments.demo_operator_splitting`.
