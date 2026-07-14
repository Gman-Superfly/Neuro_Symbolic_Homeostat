from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from core.coordinator import EnergyCoordinator
from core.curvature_audit import audit_curvature_contract
from core.interfaces import OrderParameter
from experiments.ablate_pson_noise import build_case


@dataclass(frozen=True)
class UnderreportedEdge:
    weight: float = 4.0

    def coupling_energy(self, eta_i: OrderParameter, eta_j: OrderParameter, constraints: Mapping[str, Any]) -> float:
        del constraints
        return self.weight * (float(eta_i) - float(eta_j)) ** 2

    def d_coupling_energy_d_etas(
        self, eta_i: OrderParameter, eta_j: OrderParameter, constraints: Mapping[str, Any]
    ) -> Tuple[float, float]:
        del constraints
        grad = 2.0 * self.weight * (float(eta_i) - float(eta_j))
        return grad, -grad

    def coupling_curvature_bounds(
        self, eta_i: OrderParameter, eta_j: OrderParameter, constraints: Mapping[str, Any]
    ) -> Tuple[float, float, float]:
        del eta_i, eta_j, constraints
        return 0.5, 0.5, 0.5


def test_auditor_covers_builtin_synthetic_contracts() -> None:
    case = build_case("active_hinges", seed=2, base_size=12)
    coord = EnergyCoordinator(case.modules, case.couplings, case.constraints, noise_mode="none")
    records = audit_curvature_contract(coord, [case.inputs])

    assert records
    assert all(record.status == "covered" for record in records)


def test_auditor_identifies_underreported_custom_edge() -> None:
    case = build_case("quadratic_chain", seed=1, base_size=12)
    coord = EnergyCoordinator(case.modules[:2], [(0, 1, UnderreportedEdge())], {}, noise_mode="none")
    records = audit_curvature_contract(coord, [[0.2, 0.8]])

    edge = next(record for record in records if record.component_type == "coupling")
    assert edge.status == "underreported"
    assert edge.maximum_underestimate_ratio > 10.0
