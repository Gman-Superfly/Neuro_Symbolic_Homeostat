"""Demonstrate proximal operator-splitting (prox-only path) on a tiny graph.

Exercises:
- prox_quadratic_pair for springs
- prox_asym_hinge_pair for asymmetric hinges

Compares energy before/after a short prox-only relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, List, Tuple

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling, AsymmetricHingeCoupling
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad
from core.solver_config import SolverConfig


@dataclass
class LandauModule(EnergyModule, SupportsLocalEnergyGrad):
    """Local double-well-like around target 1.0: a*(1-η)^2 + b*(1-η)^4"""
    a: float = 0.5
    b: float = 0.2

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x) if x is not None else 0.0

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        a = float(constraints.get("a_local", self.a))
        b = float(constraints.get("b_local", self.b))
        d = 1.0 - float(eta)
        return float(a * (d * d) + b * (d ** 4))

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        a = float(constraints.get("a_local", self.a))
        b = float(constraints.get("b_local", self.b))
        d = 1.0 - float(eta)
        return float(-2.0 * a * d - 4.0 * b * (d ** 3))


def build_problem() -> Tuple[List[Any], List[Tuple[int, int, Any]], Mapping[str, Any]]:
    mods = [LandauModule(a=0.6, b=0.1), LandauModule(a=0.4, b=0.15)]
    coups: List[Tuple[int, int, Any]] = [
        (0, 1, QuadraticCoupling(weight=0.8)),                   # spring
        (0, 1, AsymmetricHingeCoupling(weight=0.6, alpha_i=1.0, beta_j=1.5)),  # hinge with bias
    ]
    constraints: Mapping[str, Any] = {}
    return mods, coups, constraints


def main() -> None:
    mods, coups, constraints = build_problem()
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints=constraints,
        use_analytic=True,
        solver=SolverConfig.proximal_solver(steps=50, tau=0.05),
        stability_guard=True,
        auto_step_from_lipschitz=True,
        enable_orthogonal_noise=True,
        auto_noise_controller=True,
        precision_aware_noise_controller=True,
        noise_magnitude=1e-2,
        noise_schedule_decay=0.99,
    )

    etas0: List[float] = [0.2, 0.1]
    E0 = coord.energy(etas0)
    etas1 = coord.relax_etas(list(etas0))
    E1 = coord.energy(etas1)

    print("=== Proximal Operator-Splitting Demo ===")
    print(f"E0 = {E0:.6f} -> E1 = {E1:.6f} (dE = {E1 - E0:.6f})")
    print(f"Initial etas = {[f'{e:.3f}' for e in etas0]}")
    print(f"Final   etas = {[f'{e:.3f}' for e in etas1]}")


if __name__ == "__main__":
    main()


