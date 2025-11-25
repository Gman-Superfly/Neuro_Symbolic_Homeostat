from __future__ import annotations

from typing import Any, List, Tuple, Dict

from core.coordinator import EnergyCoordinator
from modules.gating.energy_gating import EnergyGatingModule
from core.couplings import GateBenefitCoupling


def _make_simple_gate_setup() -> Tuple[List[Any], List[Tuple[int, int, Any]], Dict[str, Any], List[Any]]:
    # Two modules: one sequence-like and one gate target; simple positive delta to encourage opening
    seq_mod = EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3)
    gate_mod = EnergyGatingModule(gain_fn=lambda _: 0.1, cost=0.05, a=0.25, b=0.35)
    modules = [seq_mod, gate_mod]
    couplings = [(0, 1, GateBenefitCoupling(weight=0.8, delta_key="delta_eta_domain"))]
    constraints = {"delta_eta_domain": 0.08}
    # Inputs for modules (EnergyGatingModule ignores input features in this simple form)
    inputs: List[Any] = [None, None]
    return modules, couplings, constraints, inputs


def test_admm_gate_benefit_non_increasing_energy() -> None:
    modules, couplings, constraints, inputs = _make_simple_gate_setup()
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
        assert b <= a + 1e-12

