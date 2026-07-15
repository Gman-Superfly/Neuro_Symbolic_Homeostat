# Conservative coordinator configuration

Status: recommended starting point for deterministic checks and controlled PSON experiments

Scope: grouped configuration for the maintained gradient solver

## Deterministic starting point

New callers can use the grouped configuration surface. The flat `EnergyCoordinator` constructor remains available for compatibility and focused experiments.

```python
from core.config import CoordinatorConfig
from core.coordinator import EnergyCoordinator

config = CoordinatorConfig.deterministic()
coordinator = EnergyCoordinator.from_config(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    config=config,
)
```

This profile selects the gradient solver, disables noise, keeps the curvature guard active, and preserves accepted-state monotonicity. It is the smallest configuration for baseline and regression checks.

## Curvature-aware PSON experiment

PSON experiments require explicit noise settings because the deterministic profile sets the noise mode to `none`.

```python
from core.config import CoordinatorConfig, GradientConfig, GuardConfig, NoiseConfig
from core.coordinator import EnergyCoordinator

config = CoordinatorConfig(
    gradient=GradientConfig(
        line_search=True,
        max_backtrack=5,
        backtrack_factor=0.5,
        use_stiffness_updates=True,
    ),
    guards=GuardConfig(
        stability_guard=True,
        auto_step_from_lipschitz=True,
        assert_monotonic_energy=True,
    ),
    noise=NoiseConfig(
        mode="precision_orthogonal",
        magnitude=1e-2,
        schedule_decay=0.99,
        auto_controller=True,
        precision_aware=True,
    ),
)

coordinator = EnergyCoordinator.from_config(
    modules=modules,
    couplings=couplings,
    constraints=constraints,
    config=config,
)
```

The stability cap uses the composed curvature bound under the assumptions documented in `docs/STABILITY_GUARANTEES.md`. A stiffness update constructs \(P_{ii}=\max(\varepsilon,\Lambda_{ii})\); the normalized bound and gradient division consume that same positive diagonal. Precision-orthogonal noise scales the candidate toward lower-curvature coordinates and then re-projects it. Above the ordinary-gradient threshold, one uniform box-feasible scale preserves a zero dot product with that gradient to numerical tolerance.

Metric modes require either an SPD `metric_matrix` or a `metric_solve` callable that applies $M^{-1}$. The implementation uses the metric gradient $M^{-1}g$ as the projection direction and checks the final first-order condition against the ordinary gradient covector $g$.

The configured down-only guard restores rejected proposals. It establishes monotonicity only for accepted states under the evaluated objective. It does not guarantee that noise is second-order free or that a conservative curvature bound improves wall-clock convergence.

## Solver alternatives

Proximal and ADMM-like modes are selected through `SolverConfig`, not independent booleans. Their current validation scope is recorded in `docs/README_OPERATOR_SPLITTING.md`.

```python
from core.solver_config import SolverConfig

config = CoordinatorConfig.deterministic(
    solver=SolverConfig.proximal_solver(steps=50, tau=0.05)
)
```
