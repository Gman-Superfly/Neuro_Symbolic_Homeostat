from __future__ import annotations

from typing import Any, List, Tuple, Dict

from core.coordinator import EnergyCoordinator
from modules.gating.energy_gating import EnergyGatingModule
from core.couplings import DampedGateBenefitCoupling, QuadraticCoupling


def _make_damped_gate_setup() -> Tuple[List[Any], List[Tuple[int, int, Any]], Dict[str, Any], List[Any]]:
    """Setup with DampedGateBenefitCoupling to test ADMM prox logic."""
    seq_mod = EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3)
    gate_mod = EnergyGatingModule(gain_fn=lambda _: 0.1, cost=0.05, a=0.25, b=0.35)
    modules = [seq_mod, gate_mod]
    couplings = [
        (0, 1, QuadraticCoupling(weight=0.3)),
        (0, 1, DampedGateBenefitCoupling(
            weight=0.8,
            delta_key="delta_eta_domain",
            damping=0.6,
            eta_power=1.5,
            positive_scale=0.9,
            negative_scale=0.4,
        )),
    ]
    constraints = {"delta_eta_domain": 0.08}
    inputs: List[Any] = [None, None]
    return modules, couplings, constraints, inputs


def test_admm_damped_gate_benefit_non_increasing_energy() -> None:
    """ADMM with DampedGateBenefitCoupling should produce non-increasing energy."""
    modules, couplings, constraints, inputs = _make_damped_gate_setup()
    coord = EnergyCoordinator(
        modules,
        couplings,
        constraints,
        use_analytic=True,
        use_admm=True,
        admm_steps=40,
        admm_rho=1.0,
        admm_step_size=0.05,
        admm_gate_prox=True,
        admm_gate_damping=0.5,
    )
    etas = coord.compute_etas(inputs)
    energies: List[float] = []
    coord.on_energy_updated.append(lambda F: energies.append(F))
    coord.relax_etas_admm(etas, steps=coord.admm_steps, rho=coord.admm_rho, step_size=coord.admm_step_size)
    # Energy should be non-increasing across accepted steps
    assert len(energies) >= 1
    for a, b in zip(energies, energies[1:]):
        assert b <= a + 1e-12, f"Energy increased: {a} -> {b}"


def test_admm_damped_gate_benefit_parity_with_gradient() -> None:
    """ADMM with DampedGateBenefitCoupling should reach similar energy as gradient descent."""
    modules, couplings, constraints, inputs = _make_damped_gate_setup()
    
    # Gradient baseline
    coord_grad = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints=constraints,
        use_analytic=True,
        line_search=False,
        step_size=0.05,
    )
    etas0_grad = coord_grad.compute_etas(inputs)
    e0_grad = coord_grad.energy(list(etas0_grad))
    etas_grad = coord_grad.relax_etas(list(etas0_grad), steps=50)
    e_grad = coord_grad.energy(etas_grad)

    # ADMM path with damped gate prox
    modules2, couplings2, constraints2, inputs2 = _make_damped_gate_setup()
    coord_admm = EnergyCoordinator(
        modules=modules2,
        couplings=couplings2,
        constraints=constraints2,
        use_admm=True,
        admm_steps=50,
        admm_rho=1.0,
        admm_step_size=0.05,
        admm_gate_prox=True,
        admm_gate_damping=0.5,
    )
    etas0_admm = coord_admm.compute_etas(inputs2)
    e0_admm = coord_admm.energy(list(etas0_admm))
    etas_admm = coord_admm.relax_etas_admm(etas0_admm, steps=coord_admm.admm_steps, rho=coord_admm.admm_rho, step_size=coord_admm.admm_step_size)
    e_admm = coord_admm.energy(etas_admm)

    # Both should reduce energy from start
    assert e_grad <= e0_grad + 1e-9
    assert e_admm <= e0_admm + 1e-9
    # Parity within a reasonable tolerance
    assert abs(e_admm - e_grad) <= 1e-2, f"ADMM {e_admm} vs Gradient {e_grad}"

