import math
from dataclasses import dataclass
from typing import Any, Mapping

from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision


@dataclass
class GatePenaltyModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Quadratic gate penalty with curvature exposed to stiffness updates."""

    curvature_value: float

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return 0.5 * self.curvature_value * float(eta) * float(eta)

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return self.curvature_value * float(eta)

    def curvature(self, eta: OrderParameter) -> float:
        return self.curvature_value


def test_wormhole_gate_gradient_independent_of_eta():
    coupling = GateBenefitCoupling(weight=2.5, delta_key="delta_eta_domain")
    constraints = {"delta_eta_domain": 1.234}

    # Two different gate values
    eta_gate_a = 0.0
    eta_gate_b = 0.9
    eta_other = 0.0  # unused for this coupling

    # Analytic gradients
    gi_a, _ = coupling.d_coupling_energy_d_etas(eta_gate_a, eta_other, constraints)
    gi_b, _ = coupling.d_coupling_energy_d_etas(eta_gate_b, eta_other, constraints)

    assert gi_a == gi_b, "Gradient w.r.t gate should not depend on gate value"

    # Finite-difference check (should match analytic and be the same for both points)
    eps = 1e-6
    def energy_at(eta_gate: float) -> float:
        return coupling.coupling_energy(eta_gate, eta_other, constraints)

    num_a = (energy_at(eta_gate_a + eps) - energy_at(eta_gate_a - eps)) / (2.0 * eps)
    num_b = (energy_at(eta_gate_b + eps) - energy_at(eta_gate_b - eps)) / (2.0 * eps)

    assert math.isfinite(num_a) and math.isfinite(num_b)
    assert abs(num_a - gi_a) <= 1e-6
    assert abs(num_b - gi_b) <= 1e-6
    assert abs(num_a - num_b) <= 1e-9


def test_wormhole_opens_closed_gate_and_lowers_energy_vs_baseline():
    constraints = {"delta_eta_domain": 0.2}
    modules = [GatePenaltyModule(curvature_value=0.2), GatePenaltyModule(curvature_value=1.0)]
    baseline = EnergyCoordinator(
        modules=modules,
        couplings=[],
        constraints=constraints,
        use_analytic=True,
        stability_guard=False,
        enable_orthogonal_noise=False,
        noise_mode="none",
        use_stiffness_updates=True,
        step_size=1.0,
    )
    wormhole = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, GateBenefitCoupling(weight=1.0, delta_key="delta_eta_domain"))],
        constraints=constraints,
        use_analytic=True,
        stability_guard=False,
        enable_orthogonal_noise=False,
        noise_mode="none",
        use_stiffness_updates=True,
        step_size=1.0,
    )
    etas0 = [0.0, 0.0]

    baseline_out = baseline.relax_etas(list(etas0), steps=3)
    wormhole_out = wormhole.relax_etas(list(etas0), steps=3)

    assert baseline_out[0] == 0.0
    assert wormhole_out[0] > 0.0
    assert wormhole.energy(wormhole_out) < baseline.energy(baseline_out)


