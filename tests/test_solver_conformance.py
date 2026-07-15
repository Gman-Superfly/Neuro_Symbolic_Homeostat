from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pytest

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from core.interfaces import EnergyCoupling, EnergyModule, OrderParameter
from core.solver_config import ADMMSolverConfig, ProximalSolverConfig, SolverConfig
from experiments.ablate_pson_noise import QuadraticWell


@dataclass(frozen=True)
class BoxOnlyQuadraticWell(EnergyModule):
    target: float

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        del constraints
        value = float(eta)
        if value < 0.0 or value > 1.0:
            raise ValueError("eta outside [0, 1]")
        return 0.5 * (value - self.target) ** 2


@dataclass(frozen=True)
class BoxOnlySeparableCoupling(EnergyCoupling):
    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        del constraints
        i = float(eta_i)
        j = float(eta_j)
        if i < 0.0 or i > 1.0 or j < 0.0 or j > 1.0:
            raise ValueError("eta outside [0, 1]")
        return 0.5 * (i * i + j * j)


def _analytic_solution() -> np.ndarray:
    k1, k2, coupling_weight = 3.0, 1.5, 0.6
    hessian = np.array(
        [
            [k1 + 2.0 * coupling_weight, -2.0 * coupling_weight],
            [-2.0 * coupling_weight, k2 + 2.0 * coupling_weight],
        ],
        dtype=float,
    )
    linear = np.array([k1 * 0.2, k2 * 0.8], dtype=float)
    return np.linalg.solve(hessian, linear)


def _coordinator(solver: SolverConfig) -> EnergyCoordinator:
    return EnergyCoordinator(
        modules=[QuadraticWell(0.2, 3.0), QuadraticWell(0.8, 1.5)],
        couplings=[(0, 1, QuadraticCoupling(weight=0.6))],
        constraints={},
        solver=solver,
        step_size=0.03,
        use_analytic=True,
        stability_guard=True,
        noise_mode="none",
        assert_monotonic_energy=True,
    )


@pytest.mark.parametrize(
    ("solver", "tolerance"),
    [
        (SolverConfig.gradient_solver(), 2e-5),
        (SolverConfig.proximal_solver(steps=500, tau=0.02), 2e-3),
        (SolverConfig.admm_solver(steps=500, rho=1.0, step_size=0.02, gate_prox=False), 2e-5),
    ],
)
def test_solver_matches_bounded_convex_reference(solver: SolverConfig, tolerance: float) -> None:
    coordinator = _coordinator(solver)
    initial = [0.95, 0.05]
    initial_energy = coordinator.energy(initial)
    accepted_energies: list[float] = []
    coordinator.on_energy_updated.append(accepted_energies.append)

    result = coordinator.relax_etas(initial, steps=500)

    assert np.all(np.isfinite(result))
    assert np.all((0.0 <= np.asarray(result)) & (np.asarray(result) <= 1.0))
    assert coordinator.energy(result) <= initial_energy + 1e-12
    assert all(next_value <= value + 1e-12 for value, next_value in zip(accepted_energies, accepted_energies[1:]))
    assert np.allclose(result, _analytic_solution(), atol=tolerance, rtol=0.0)


@pytest.mark.parametrize(
    "solver",
    [
        SolverConfig.proximal_solver(steps=1, tau=20.0),
        SolverConfig.admm_solver(steps=1, rho=1.0, step_size=20.0, gate_prox=False),
    ],
)
def test_split_solver_restores_rejected_candidate(solver: SolverConfig) -> None:
    coordinator = EnergyCoordinator(
        modules=[QuadraticWell(0.5, 1.0)],
        couplings=[],
        constraints={},
        solver=solver,
        noise_mode="none",
    )
    initial = [0.6]

    result = coordinator.relax_etas(initial)

    assert result == initial
    assert coordinator.last_solver_metrics()["accepted_steps"] == 0
    assert coordinator.last_solver_metrics()["rejected_steps"] == 1


@pytest.mark.parametrize(
    "solver",
    [
        SolverConfig.proximal_solver(steps=1, tau=0.1),
        SolverConfig.admm_solver(steps=1, rho=1.0, step_size=0.1, gate_prox=False),
    ],
)
def test_split_solver_finite_difference_moves_inward_from_upper_boundary(
    solver: SolverConfig,
) -> None:
    coordinator = EnergyCoordinator(
        modules=[BoxOnlyQuadraticWell(target=0.0)],
        couplings=[],
        constraints={},
        solver=solver,
        noise_mode="none",
    )

    result = coordinator.relax_etas([1.0])

    assert result[0] < 1.0
    assert coordinator.energy(result) < coordinator.energy([1.0])


def test_admm_coupling_fallback_is_box_aware_at_upper_boundary() -> None:
    coordinator = EnergyCoordinator(
        modules=[BoxOnlyQuadraticWell(target=1.0), BoxOnlyQuadraticWell(target=1.0)],
        couplings=[(0, 1, BoxOnlySeparableCoupling())],
        constraints={},
        solver=SolverConfig.admm_solver(steps=1, rho=1.0, step_size=0.1, gate_prox=False),
        noise_mode="none",
    )

    result = coordinator.relax_etas([1.0, 1.0])

    assert result == pytest.approx([0.9, 0.9], abs=1e-10)
    assert coordinator.energy(result) < coordinator.energy([1.0, 1.0])


def test_admm_residuals_contract_on_convex_reference() -> None:
    coordinator = _coordinator(SolverConfig.admm_solver(steps=100, rho=1.0, step_size=0.02))

    coordinator.relax_etas([0.95, 0.05])
    metrics = coordinator.last_solver_metrics()

    assert len(metrics["primal_residuals"]) == metrics["accepted_steps"] + metrics["rejected_steps"]
    assert len(metrics["dual_residuals"]) == len(metrics["primal_residuals"])
    assert np.all(np.isfinite(metrics["primal_residuals"]))
    assert np.all(np.isfinite(metrics["dual_residuals"]))
    assert metrics["primal_residuals"][-1] < metrics["primal_residuals"][0]
    assert metrics["dual_residuals"][-1] < metrics["dual_residuals"][0]
    assert metrics["primal_residuals"][-1] < 1e-3
    assert metrics["dual_residuals"][-1] < 1e-3


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ProximalSolverConfig(steps=0),
        lambda: ProximalSolverConfig(tau=0.0),
        lambda: ProximalSolverConfig(block_mode="invalid"),  # type: ignore[arg-type]
        lambda: ADMMSolverConfig(steps=0),
        lambda: ADMMSolverConfig(rho=0.0),
        lambda: ADMMSolverConfig(step_size=0.0),
        lambda: ADMMSolverConfig(gate_damping=1.1),
    ],
)
def test_invalid_solver_configuration_is_rejected(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()
