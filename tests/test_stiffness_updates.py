from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple, List

import math

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision
from core.couplings import QuadraticCoupling, GateBenefitCoupling


@dataclass
class LocalQuadraticModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Simple quadratic local energy: 0.5 * c * (eta - target)^2."""

    target: float
    curvature_c: float = 2.0

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x) if x is not None else 0.0

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        diff = float(eta) - float(self.target)
        return 0.5 * float(self.curvature_c) * (diff * diff)

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return float(self.curvature_c) * (float(eta) - float(self.target))

    def curvature(self, eta: OrderParameter) -> float:  # type: ignore[override]
        return float(self.curvature_c)


@dataclass
class FlatModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Zero-curvature, zero-energy module for coupling-only curvature tests."""

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x) if x is not None else 0.0

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return 0.0

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return 0.0

    def curvature(self, eta: OrderParameter) -> float:  # type: ignore[override]
        return 0.0


def _energy(coord: EnergyCoordinator, etas: List[float]) -> float:
    return float(coord.energy(list(map(float, etas))))


def test_stiffness_update_decreases_energy_quadratic() -> None:
    # Two independent quadratics (no couplings), exact Newton/Jacobi should jump to target
    mods = [LocalQuadraticModule(target=0.8, curvature_c=2.0), LocalQuadraticModule(target=0.3, curvature_c=2.0)]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=[],
        constraints={},
        use_analytic=True,
        stability_guard=False,          # decouple from global caps
        enable_orthogonal_noise=False,  # deterministic
        use_stiffness_updates=True,
        stiffness_epsilon=1e-9,
        step_size=1.0,                  # one Newton step to target
    )
    etas0 = [0.0, 1.0]
    E0 = _energy(coord, etas0)
    etas1 = coord.relax_etas(etas0, steps=1)
    E1 = _energy(coord, etas1)
    # Should move directly to targets and strictly reduce energy
    assert E1 < E0 - 1e-12, f"Energy did not decrease: {E0} -> {E1}"
    assert math.isclose(etas1[0], 0.8, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(etas1[1], 0.3, rel_tol=0, abs_tol=1e-9)


def test_coupling_curvature_added_to_precision() -> None:
    # No local curvature, only a quadratic coupling => diagonal curvature should be 2*w on both nodes
    mods = [FlatModule(), FlatModule()]
    coups = [(0, 1, QuadraticCoupling(weight=1.0))]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        use_analytic=True,
        stability_guard=False,
        enable_orthogonal_noise=False,
        use_stiffness_updates=True,
        stiffness_epsilon=1e-9,
        step_size=1.0,
    )
    etas = [0.2, 0.4]
    # Update precision cache and check diagonal
    coord._update_precision_cache(etas)  # type: ignore[attr-defined]
    diag = coord.get_precision_diagonal()
    assert len(diag) == 2
    assert math.isclose(diag[0], 2.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(diag[1], 2.0, rel_tol=0, abs_tol=1e-12)


def test_stiffness_vs_preconditioning_equivalence_module_only() -> None:
    # With module-only curvature and no couplings, stiffness-based update equals preconditioned GD for one step
    mods_a = [LocalQuadraticModule(target=0.6, curvature_c=4.0)]
    mods_b = [LocalQuadraticModule(target=0.6, curvature_c=4.0)]
    etas0 = [0.0]

    # Path A: stiffness updates
    coord_a = EnergyCoordinator(
        modules=mods_a,
        couplings=[],
        constraints={},
        use_analytic=True,
        stability_guard=False,
        enable_orthogonal_noise=False,
        use_stiffness_updates=True,
        stiffness_epsilon=1e-12,
        step_size=1.0,
    )
    out_a = coord_a.relax_etas(list(etas0), steps=1)

    # Path B: precision preconditioning only
    coord_b = EnergyCoordinator(
        modules=mods_b,
        couplings=[],
        constraints={},
        use_analytic=True,
        stability_guard=False,
        enable_orthogonal_noise=False,
        use_stiffness_updates=False,
        use_precision_preconditioning=True,
        precision_epsilon=1e-12,
        step_size=1.0,
    )
    out_b = coord_b.relax_etas(list(etas0), steps=1)

    assert math.isclose(out_a[0], out_b[0], rel_tol=0, abs_tol=1e-12)


def test_wormhole_gradient_unchanged() -> None:
    # Ensure GateBenefit contributes as a linear force (no curvature), unaffected by stiffness mode
    mods = [FlatModule(), FlatModule()]
    coups = [(0, 1, GateBenefitCoupling(weight=2.0, delta_key="delta_benefit"))]
    constraints = {"delta_benefit": 0.3}
    etas = [0.0, 0.5]  # gate closed, domain mid

    def gate_grad(coord: EnergyCoordinator) -> float:
        # Use analytic grads; gradient on gate = -w * delta
        grads = coord._analytic_grads(etas)  # type: ignore[attr-defined]
        return float(grads[0])

    coord_s = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints=constraints,
        use_analytic=True,
        stability_guard=False,
        enable_orthogonal_noise=False,
        use_stiffness_updates=True,
    )
    coord_g = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints=constraints,
        use_analytic=True,
        stability_guard=False,
        enable_orthogonal_noise=False,
        use_stiffness_updates=False,
    )
    gs = gate_grad(coord_s)
    gg = gate_grad(coord_g)
    # Expected -w*delta = -0.6
    assert math.isclose(gs, -0.6, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(gg, -0.6, rel_tol=0, abs_tol=1e-12)

