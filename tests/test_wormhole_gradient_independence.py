import math

from core.couplings import GateBenefitCoupling


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


