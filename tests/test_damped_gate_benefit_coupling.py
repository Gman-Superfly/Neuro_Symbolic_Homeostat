from __future__ import annotations

import math
import numpy as np

from core.couplings import DampedGateBenefitCoupling


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

