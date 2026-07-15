from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math
import numpy as np
import pytest

import core.coordinator as coordinator_module
from core.coordinator import EnergyCoordinator
from core.coordinator_noise import apply_box_feasible_noise
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision


@dataclass
class CurvedModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Quadratic module with fixed curvature for noise-mode tests."""

    curvature_value: float

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return 0.5 * self.curvature_value * float(eta) * float(eta)

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return self.curvature_value * float(eta)

    def curvature(self, eta: OrderParameter) -> float:
        return self.curvature_value


def _coord(noise_mode: str) -> EnergyCoordinator:
    coord = EnergyCoordinator(
        modules=[CurvedModule(100.0), CurvedModule(1.0), CurvedModule(0.0)],
        couplings=[],
        constraints={},
        noise_mode=noise_mode,
        noise_magnitude=0.5,
        stability_guard=False,
    )
    coord._update_precision_cache([0.1, 0.1, 0.1])  # type: ignore[attr-defined]
    return coord


def test_isotropic_noise_mode_keeps_gradient_component() -> None:
    coord = _coord("isotropic")
    raw_noise = np.array([1.0, 0.0, 0.0], dtype=float)
    grad = np.array([1.0, 0.0, 0.0], dtype=float)

    noise = coord._build_noise_vector(raw_noise, grad, current_noise_mag=0.5)  # type: ignore[attr-defined]

    assert math.isclose(float(np.linalg.norm(noise)), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert float(np.dot(noise, grad)) > 0.0


def test_orthogonal_noise_mode_removes_gradient_component() -> None:
    coord = _coord("orthogonal")
    raw_noise = np.array([1.0, 1.0, 0.0], dtype=float)
    grad = np.array([1.0, 0.0, 0.0], dtype=float)

    noise = coord._build_noise_vector(raw_noise, grad, current_noise_mag=0.5)  # type: ignore[attr-defined]

    assert math.isclose(float(np.linalg.norm(noise)), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert abs(float(np.dot(noise, grad))) <= 1e-12


def test_precision_orthogonal_noise_stays_orthogonal_after_weighting() -> None:
    coord = _coord("precision_orthogonal")
    raw_noise = np.array([0.2, -0.4, 1.0], dtype=float)
    grad = np.array([1.0, 1.0, 0.0], dtype=float)

    noise = coord._build_noise_vector(raw_noise, grad, current_noise_mag=0.5)  # type: ignore[attr-defined]

    assert math.isclose(float(np.linalg.norm(noise)), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert abs(float(np.dot(noise, grad))) <= 1e-12
    assert abs(float(noise[2])) > abs(float(noise[0]))


def test_metric_orthogonal_noise_has_zero_first_order_component() -> None:
    metric = np.diag([1.0, 4.0, 0.5])
    coord = EnergyCoordinator(
        modules=[CurvedModule(100.0), CurvedModule(1.0), CurvedModule(0.0)],
        couplings=[],
        constraints={},
        noise_mode="metric_orthogonal",
        noise_magnitude=0.5,
        metric_matrix=metric,
        stability_guard=False,
    )
    raw_noise = np.array([0.2, -0.4, 1.0], dtype=float)
    grad = np.array([1.0, 1.0, 0.0], dtype=float)

    noise = coord._build_noise_vector(raw_noise, grad, current_noise_mag=0.5)  # type: ignore[attr-defined]

    assert math.isclose(float(np.linalg.norm(noise)), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert abs(float(np.dot(noise, grad))) <= 1e-12


def test_metric_orthogonal_mode_requires_metric_geometry() -> None:
    with pytest.raises(AssertionError, match="metric_matrix or metric_solve"):
        EnergyCoordinator(
            modules=[CurvedModule(1.0)],
            couplings=[],
            constraints={},
            noise_mode="metric_orthogonal",
        )


def test_metric_precision_noise_reprojects_after_curvature_weighting() -> None:
    metric = np.array(
        [
            [2.0, 0.4, 0.1],
            [0.4, 1.5, 0.2],
            [0.1, 0.2, 0.8],
        ],
        dtype=float,
    )
    coord = EnergyCoordinator(
        modules=[CurvedModule(100.0), CurvedModule(1.0), CurvedModule(0.0)],
        couplings=[],
        constraints={},
        noise_magnitude=0.5,
        precision_aware_noise_controller=True,
        metric_aware_noise_controller=True,
        metric_matrix=metric,
        stability_guard=False,
    )
    coord._update_precision_cache([0.1, 0.1, 0.1])  # type: ignore[attr-defined]
    raw_noise = np.array([0.2, -0.4, 1.0], dtype=float)
    grad = np.array([1.0, 1.0, 0.0], dtype=float)

    assert coord._resolved_noise_mode() == "metric_precision_orthogonal"  # type: ignore[attr-defined]
    noise = coord._build_noise_vector(raw_noise, grad, current_noise_mag=0.5)  # type: ignore[attr-defined]

    assert math.isclose(float(np.linalg.norm(noise)), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert abs(float(np.dot(noise, grad))) <= 1e-12
    assert abs(float(noise[2])) > abs(float(noise[0]))


def test_box_feasible_noise_uses_uniform_scaling_and_preserves_tangency() -> None:
    state = np.array([0.9, 0.2], dtype=float)
    tangent_noise = np.array([0.4, -0.8], dtype=float)
    gradient = np.array([2.0, 1.0], dtype=float)

    proposal = apply_box_feasible_noise(state, tangent_noise)
    realized_noise = proposal - state

    assert np.allclose(realized_noise, 0.25 * tangent_noise, rtol=0.0, atol=1e-12)
    assert np.all((0.0 <= proposal) & (proposal <= 1.0))
    assert abs(float(np.dot(gradient, realized_noise))) <= 1e-12


def test_relaxation_preserves_preproposal_tangency_after_deterministic_step(
    monkeypatch: Any,
) -> None:
    initial = np.asarray([0.8, 0.2], dtype=float)
    initial_gradient = np.asarray([0.8, 0.8], dtype=float)
    captured: list[tuple[np.ndarray, np.ndarray]] = []

    monkeypatch.setattr(
        np.random,
        "normal",
        lambda *args, **kwargs: np.asarray([1.0, -0.5], dtype=float),
    )

    def capture_realized(state: np.ndarray, noise: np.ndarray) -> np.ndarray:
        proposal = apply_box_feasible_noise(state, noise)
        captured.append((np.asarray(state, dtype=float).copy(), proposal - state))
        return proposal

    monkeypatch.setattr(coordinator_module, "apply_box_feasible_noise", capture_realized)
    coord = EnergyCoordinator(
        modules=[CurvedModule(1.0), CurvedModule(4.0)],
        couplings=[],
        constraints={},
        use_precision_preconditioning=False,
        stability_guard=False,
        noise_mode="precision_orthogonal",
        noise_magnitude=0.01,
        step_size=0.05,
        assert_monotonic_energy=False,
    )

    coord.relax_etas(initial.tolist(), steps=1)

    assert len(captured) == 1
    deterministic_state, realized_noise = captured[0]
    assert not np.allclose(deterministic_state, initial)
    assert abs(float(np.dot(initial_gradient, realized_noise))) <= 1e-12
    application_gradient = np.asarray(
        [deterministic_state[0], 4.0 * deterministic_state[1]],
        dtype=float,
    )
    assert abs(float(np.dot(application_gradient, realized_noise))) > 1e-8


def test_relaxation_builds_noise_from_raw_objective_gradient_when_updates_are_normalized() -> None:
    coord = EnergyCoordinator(
        modules=[CurvedModule(1.0)],
        couplings=[],
        constraints={},
        normalize_grads=True,
        stability_guard=False,
        noise_mode="orthogonal",
        noise_magnitude=0.1,
        assert_monotonic_energy=False,
    )
    captured: list[np.ndarray] = []

    def capture(raw_noise: np.ndarray, grad: np.ndarray, magnitude: float) -> np.ndarray:
        del raw_noise, magnitude
        captured.append(np.asarray(grad, dtype=float).copy())
        return np.zeros_like(grad)

    coord._build_noise_vector = capture  # type: ignore[method-assign]
    coord.relax_etas([1e-10], steps=1)

    assert len(captured) == 1
    assert np.array_equal(captured[0], np.asarray([1e-10]))
