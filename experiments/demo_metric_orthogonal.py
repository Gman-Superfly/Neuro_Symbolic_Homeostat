"""Demonstrate metric-orthogonal (M-orthogonal) noise projection and re-projection.

Shows:
- Euclidean vs M-orthogonal projection property (dot products ~ 0)
- A tiny relaxation run with metric-aware projection enabled
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, List

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision
from core.energy import project_noise_orthogonal, project_noise_metric_orthogonal


@dataclass
class LocalQuadratic(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    target: float = 0.0
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


def demo_metric_projection_property() -> None:
    rng = np.random.default_rng(0)
    g = rng.normal(0.0, 1.0, size=(4,))
    z = rng.normal(0.0, 1.0, size=(4,))
    # SPD metric with anisotropy
    M = np.diag([1.0, 5.0, 0.5, 2.0])

    z_e = project_noise_orthogonal(z, g)
    z_m = project_noise_metric_orthogonal(z, g, M=M)

    dot_e = float(np.dot(z_e, g))
    Mg = M @ g
    dot_m = float(np.dot(z_m, Mg))

    print("=== Metric-Orthogonal Projection Property ===")
    print(f"||g||^2 = {float(np.dot(g, g)):.4f}")
    print(f"Euclidean: z_e dot g = {dot_e:.4e} (~ 0)")
    print(f"Metric:    z_m^T M g = {dot_m:.4e} (~ 0)")
    print()


def demo_metric_projection_in_relaxation() -> None:
    # Two variables with a spring; enable M-orthogonal projection and precision-aware redistribution
    mods = [LocalQuadratic(target=0.8, curvature_c=2.0), LocalQuadratic(target=0.2, curvature_c=3.0)]
    coups = [(0, 1, QuadraticCoupling(weight=0.5))]

    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        use_analytic=True,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        enable_orthogonal_noise=True,
        noise_magnitude=1e-2,                 # small noise
        auto_noise_controller=True,           # schedule magnitude automatically
        precision_aware_noise_controller=True,  # exercise re-projection after weighting
        metric_aware_noise_controller=True,     # use M-orthogonal projection
    )
    # Provide an SPD metric
    coord.metric_matrix = np.diag([1.0, 10.0])

    etas0: List[float] = [0.0, 1.0]

    E0 = coord.energy(etas0)
    etas1 = coord.relax_etas(list(etas0), steps=25)
    E1 = coord.energy(etas1)

    print("=== Metric-Orthogonal Noise in Relaxation ===")
    print(f"E0 = {E0:.6f} -> E1 = {E1:.6f} (dE = {E1 - E0:.6f})")
    print(f"Final etas = {[f'{e:.3f}' for e in etas1]}")
    print()


def main() -> None:
    demo_metric_projection_property()
    demo_metric_projection_in_relaxation()


if __name__ == "__main__":
    main()


