from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.config import CoordinatorConfig, GradientConfig, NoiseConfig
from core.coordinator import EnergyCoordinator
from core.solver_config import SolverConfig, SolverMode
from experiments.ablate_pson_noise import QuadraticWell


def test_grouped_config_constructs_coordinator() -> None:
    config = CoordinatorConfig(
        solver=SolverConfig.proximal_solver(steps=7, tau=0.02),
        gradient=GradientConfig(step_size=0.03, line_search=True),
        noise=NoiseConfig(mode="none"),
    )

    coordinator = EnergyCoordinator.from_config(
        modules=[QuadraticWell(0.5, 1.0)],
        couplings=[],
        constraints={},
        config=config,
    )

    assert coordinator.solver.mode == SolverMode.PROXIMAL
    assert coordinator.solver.proximal.steps == 7
    assert coordinator.step_size == 0.03
    assert coordinator.line_search
    assert coordinator.noise_mode == "none"


def test_deterministic_profile_disables_noise() -> None:
    config = CoordinatorConfig.deterministic()

    assert config.noise.mode == "none"
    assert config.noise.magnitude == 0.0
    assert config.solver.mode == SolverMode.GRADIENT


def test_grouped_config_is_immutable() -> None:
    config = CoordinatorConfig()

    with pytest.raises(FrozenInstanceError):
        config.gradient = GradientConfig(step_size=0.1)  # type: ignore[misc]
