from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling, QuadraticCoupling
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision
from core.weight_adapters import SmallGainWeightAdapter
from experiments.demo_constraint_correction import run_constraint_correction


@dataclass(frozen=True)
class QuadraticModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Small local quadratic used for end-to-end relaxation tests."""

    target: float
    curvature_value: float = 1.0

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        diff = float(eta) - self.target
        return 0.5 * self.curvature_value * diff * diff

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return self.curvature_value * (float(eta) - self.target)

    def curvature(self, eta: OrderParameter) -> float:
        return self.curvature_value


def test_safe_defaults_keep_accepted_energy_monotone() -> None:
    modules = [QuadraticModule(0.2), QuadraticModule(0.8)]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, QuadraticCoupling(weight=0.2))],
        constraints={},
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
        step_size=0.05,
        stability_guard=True,
    )
    etas0 = [0.9, 0.1]
    energies = [coord.energy(list(etas0))]
    coord.on_energy_updated.append(lambda energy: energies.append(float(energy)))

    coord.relax_etas(list(etas0), steps=25)

    accepted = energies[1:]
    assert len(accepted) > 2
    for before, after in zip(accepted, accepted[1:]):
        assert after <= before + 1e-12


def test_gradient_norm_decreases_on_quadratic_relaxation() -> None:
    modules = [QuadraticModule(0.1, 2.0), QuadraticModule(0.9, 3.0)]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, QuadraticCoupling(weight=0.1))],
        constraints={},
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
        use_stiffness_updates=True,
        step_size=0.2,
        stability_guard=False,
    )
    etas0 = [0.8, 0.2]
    grad0 = float(np.linalg.norm(np.asarray(coord._grads(etas0), dtype=float)))  # type: ignore[attr-defined]
    etas1 = coord.relax_etas(list(etas0), steps=20)
    grad1 = float(np.linalg.norm(np.asarray(coord._grads(etas1), dtype=float)))  # type: ignore[attr-defined]

    assert grad1 < grad0


def test_contraction_margin_positive_across_coupling_sweep() -> None:
    for weight in (0.05, 0.2, 0.5):
        coord = EnergyCoordinator(
            modules=[QuadraticModule(0.3), QuadraticModule(0.7)],
            couplings=[(0, 1, QuadraticCoupling(weight=weight))],
            constraints={},
            use_analytic=True,
            enable_orthogonal_noise=False,
            noise_mode="none",
            step_size=0.05,
            stability_guard=True,
            log_contraction_margin=True,
        )
        coord.relax_etas([0.9, 0.1], steps=3)
        margins = [value for value in getattr(coord, "_contraction_margin_history") if np.isfinite(value)]
        assert margins, "expected contraction margins"
        assert min(margins) > 0.0


def test_constraint_correction_demo_reduces_all_reported_violations() -> None:
    result = run_constraint_correction()
    for key, before in result["raw_violations"].items():
        after = result["relaxed_violations"][key]
        assert after < before, f"{key} did not improve: {before} -> {after}"
    assert result["relaxed_energy"] < result["raw_energy"]


def test_wrong_sign_gate_benefit_pushes_gate_closed() -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5), QuadraticModule(0.0)],
        couplings=[(0, 1, GateBenefitCoupling(weight=1.0, delta_key="delta_benefit"))],
        constraints={"delta_benefit": -0.2},
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
        step_size=0.05,
        stability_guard=False,
        assert_monotonic_energy=False,
    )
    etas = coord.relax_etas([0.5, 0.0], steps=1)

    assert etas[0] < 0.5


def test_too_large_isotropic_noise_rejects_and_restores_state() -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5)],
        couplings=[],
        constraints={},
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="isotropic",
        noise_magnitude=1.0,
        step_size=0.05,
        stability_guard=False,
        assert_monotonic_energy=False,
    )
    etas0 = [0.5]
    etas1 = coord.relax_etas(list(etas0), steps=1)

    assert getattr(coord, "_rejected_steps") == 1
    assert etas1 == etas0


def test_sparse_smallgain_preserves_monotone_energy_but_is_not_speed_claim() -> None:
    modules = [QuadraticModule(0.2), QuadraticModule(0.8)]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, QuadraticCoupling(weight=0.1))],
        constraints={},
        weight_adapter=SmallGainWeightAdapter(budget_fraction=0.5, max_step_change=0.05),
        expose_lipschitz_details=True,
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
        step_size=0.03,
        stability_guard=True,
        assert_monotonic_energy=False,
    )
    energies = [coord.energy([0.9, 0.1])]
    coord.on_energy_updated.append(lambda energy: energies.append(float(energy)))

    coord.relax_etas([0.9, 0.1], steps=10)

    accepted = energies[1:]
    assert len(accepted) > 1
    assert accepted[-1] <= accepted[0]
