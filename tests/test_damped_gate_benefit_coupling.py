from __future__ import annotations

import math
import numpy as np
import pytest

from core.coordinator import EnergyCoordinator
from core.couplings import DampedGateBenefitCoupling
from experiments.ablate_pson_noise import QuadraticWell


def test_damped_gate_coupling_respects_scales() -> None:
    coup = DampedGateBenefitCoupling(
        weight=1.2,
        damping=0.5,
        eta_power=2.0,
        positive_scale=0.8,
        negative_scale=0.25,
    )
    eta_gate = 0.85
    # positive delta
    e_pos = coup.coupling_energy(eta_gate, 0.0, {"delta_eta_domain": 0.4})
    expected_pos = -1.2 * 0.5 * (eta_gate ** 2.0) * (0.4 * 0.8)
    assert math.isclose(e_pos, expected_pos, rel_tol=1e-9)
    # negative delta
    e_neg = coup.coupling_energy(eta_gate, 0.0, {"delta_eta_domain": -0.3})
    expected_neg = -1.2 * 0.5 * (eta_gate ** 2.0) * (-0.3 * 0.25)
    assert math.isclose(e_neg, expected_neg, rel_tol=1e-9)


def test_damped_gate_coupling_grad_matches_numeric() -> None:
    coup = DampedGateBenefitCoupling(
        weight=0.9,
        damping=0.6,
        eta_power=1.5,
        positive_scale=1.0,
        negative_scale=0.5,
    )
    eta_gate = 0.7
    constraints = {"delta_eta_domain": 0.25}
    base = coup.coupling_energy(eta_gate, 0.0, constraints)
    eps = 1e-6
    num = (coup.coupling_energy(eta_gate + eps, 0.0, constraints) - base) / eps
    gi, gj = coup.d_coupling_energy_d_etas(eta_gate, 0.0, constraints)
    assert gj == 0.0
    assert abs(gi - num) < 1e-4


def test_damped_gate_coupling_stability_sweep() -> None:
    coup = DampedGateBenefitCoupling(
        weight=1.0,
        damping=0.3,
        eta_power=1.0,
        positive_scale=0.7,
        negative_scale=0.4,
    )
    eta_gate = 0.9
    deltas = np.linspace(-0.3, 0.6, num=10)
    energies = [
        coup.coupling_energy(eta_gate, 0.0, {"delta_eta_domain": float(d)}) for d in deltas
    ]
    pos = [e for d, e in zip(deltas, energies) if d > 0]
    neg = [e for d, e in zip(deltas, energies) if d < 0]
    # positive deltas should yield monotonically decreasing (more negative) energy
    for a, b in zip(pos, pos[1:]):
        assert b <= a + 1e-9
    # negative deltas: penalty should relax as harm lessens (monotonic decrease)
    for a, b in zip(neg, neg[1:]):
        assert b <= a + 1e-9


def test_quadratic_gate_power_reports_exact_box_curvature_bound() -> None:
    coupling = DampedGateBenefitCoupling(
        weight=1.2,
        damping=0.5,
        eta_power=2.0,
        positive_scale=0.8,
    )
    diagonal_i, diagonal_j, off_diagonal = coupling.coupling_curvature_bounds(
        0.3,
        0.0,
        {"delta_eta_domain": 0.4},
    )

    assert diagonal_i == pytest.approx(0.384)
    assert diagonal_j == 0.0
    assert off_diagonal == 0.0

    coordinator = EnergyCoordinator(
        modules=[QuadraticWell(0.5, 0.0), QuadraticWell(0.5, 0.0)],
        couplings=[(0, 1, coupling)],
        constraints={"delta_eta_domain": 0.4},
        stability_guard=True,
    )
    snapshot = coordinator.inspect_state([0.3, 0.5])
    assert snapshot.lipschitz_bound == pytest.approx(0.384)
    assert snapshot.preconditioner_diagonal == pytest.approx((0.384, 1e-8))
    assert snapshot.update_lipschitz_bound == pytest.approx(1.0)


def test_subquadratic_gate_power_requires_line_search_for_guarded_gradient_solver() -> None:
    coupling = DampedGateBenefitCoupling(eta_power=1.5)

    fixed_step = EnergyCoordinator(
        modules=[QuadraticWell(0.5, 1.0), QuadraticWell(0.5, 1.0)],
        couplings=[(0, 1, coupling)],
        constraints={"delta_eta_domain": 0.2},
        stability_guard=True,
        line_search=False,
    )
    with pytest.raises(ValueError, match="curvature bound is infinite"):
        fixed_step.relax_etas([0.4, 0.5], steps=1)

    coordinator = EnergyCoordinator(
        modules=[QuadraticWell(0.5, 1.0), QuadraticWell(0.5, 1.0)],
        couplings=[(0, 1, coupling)],
        constraints={"delta_eta_domain": 0.2},
        stability_guard=True,
        line_search=True,
    )
    assert math.isinf(coordinator.inspect_state([0.4, 0.5]).lipschitz_bound)


def test_subquadratic_gate_power_has_zero_bound_when_frozen_coefficient_is_zero() -> None:
    coupling = DampedGateBenefitCoupling(eta_power=1.5)
    coordinator = EnergyCoordinator(
        modules=[QuadraticWell(0.5, 1.0), QuadraticWell(0.5, 1.0)],
        couplings=[(0, 1, coupling)],
        constraints={"delta_eta_domain": 0.0},
        stability_guard=True,
        line_search=False,
    )

    snapshot = coordinator.inspect_state([0.4, 0.5])
    result = coordinator.relax_etas([0.4, 0.5], steps=1)

    assert math.isfinite(snapshot.update_lipschitz_bound)
    assert all(math.isfinite(value) for value in result)


def test_gate_power_below_one_is_rejected_on_closed_box() -> None:
    with pytest.raises(ValueError, match="eta_power must be at least 1.0"):
        DampedGateBenefitCoupling(eta_power=0.5)

