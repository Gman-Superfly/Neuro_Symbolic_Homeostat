from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math
import numpy as np

from core.coordinator import EnergyCoordinator
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
