"""Demonstrate constraint correction on a small System-1 output.

The demo uses hand-crafted noisy System-1 scores and an energy relaxation layer
to reduce explicit symbolic constraint violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from core.coordinator import EnergyCoordinator
from core.couplings import DirectedHingeCoupling
from core.interfaces import (
    EnergyModule,
    OrderParameter,
    SupportsCouplingCurvature,
    SupportsCouplingGrads,
    SupportsLocalEnergyGrad,
    SupportsPrecision,
)

__all__ = ["run_constraint_correction", "main"]


@dataclass(frozen=True)
class TargetScoreModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Keep a relaxed order parameter close to the raw System-1 score."""

    target: float
    stiffness: float = 1.0

    def compute_eta(self, x: Any) -> OrderParameter:
        """Return the supplied initial score.

        Args:
            x: Initial score in [0, 1].

        Returns:
            The initial order parameter.
        """
        eta = float(x)
        assert 0.0 <= eta <= 1.0, f"eta out of bounds: {eta}"
        return eta

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        """Compute quadratic deviation from the raw System-1 score."""
        diff = float(eta) - self.target
        return 0.5 * self.stiffness * diff * diff

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        """Compute derivative of the local quadratic energy."""
        return self.stiffness * (float(eta) - self.target)

    def curvature(self, eta: OrderParameter) -> float:
        """Return local quadratic curvature."""
        return self.stiffness


@dataclass(frozen=True)
class SumToOneCoupling(SupportsCouplingGrads, SupportsCouplingCurvature):
    """Penalize a two-gate sum that differs from one."""

    weight: float = 10.0

    def coupling_energy(self, eta_i: OrderParameter, eta_j: OrderParameter, constraints: Mapping[str, Any]) -> float:
        """Compute sum-to-one penalty."""
        residual = float(eta_i) + float(eta_j) - 1.0
        return self.weight * residual * residual

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        """Compute gradients of the sum-to-one penalty."""
        residual = float(eta_i) + float(eta_j) - 1.0
        grad = 2.0 * self.weight * residual
        return grad, grad

    def coupling_curvature_bounds(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float, float]:
        """Return exact Hessian row contributions for the sum penalty."""
        curvature = 2.0 * self.weight
        return curvature, curvature, curvature


@dataclass(frozen=True)
class ProductExclusionCoupling(SupportsCouplingGrads, SupportsCouplingCurvature):
    """Penalize two mutually exclusive gates being high together."""

    weight: float = 2.0

    def coupling_energy(self, eta_i: OrderParameter, eta_j: OrderParameter, constraints: Mapping[str, Any]) -> float:
        """Compute mutual-exclusion product penalty."""
        product = float(eta_i) * float(eta_j)
        return self.weight * product * product

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        """Compute gradients of the product penalty."""
        left = float(eta_i)
        right = float(eta_j)
        return 2.0 * self.weight * left * right * right, 2.0 * self.weight * right * left * left

    def coupling_curvature_bounds(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float, float]:
        """Return conservative Hessian row contributions on [0, 1]."""
        left = max(0.0, min(1.0, float(eta_i)))
        right = max(0.0, min(1.0, float(eta_j)))
        diag_i = 2.0 * self.weight * right * right
        diag_j = 2.0 * self.weight * left * left
        off = abs(4.0 * self.weight * left * right)
        return diag_i, diag_j, off


def _violations(etas: List[float]) -> Dict[str, float]:
    """Compute human-readable constraint violations."""
    assert len(etas) == 4, "expected four order parameters"
    action_a, action_b, low_risk, high_risk = [float(value) for value in etas]
    return {
        "sum_to_one": abs((action_a + action_b) - 1.0),
        "mutual_exclusion": action_a * action_b,
        "monotonicity": max(0.0, high_risk - low_risk),
    }


def run_constraint_correction(steps: int = 500) -> Dict[str, Any]:
    """Run the deterministic constraint-correction demo.

    Args:
        steps: Number of relaxation steps.

    Returns:
        Dictionary containing raw and relaxed outputs plus diagnostics.
    """
    assert steps > 0, "steps must be positive"
    raw_scores = [0.82, 0.71, 0.25, 0.68]
    modules = [TargetScoreModule(target=value, stiffness=0.25) for value in raw_scores]
    couplings = [
        (0, 1, SumToOneCoupling(weight=12.0)),
        (0, 1, ProductExclusionCoupling(weight=2.0)),
        (2, 3, DirectedHingeCoupling(weight=8.0)),
    ]
    coordinator = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={},
        use_analytic=True,
        use_stiffness_updates=False,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
        step_size=0.05,
        assert_monotonic_energy=False,
    )
    raw_violations = _violations(raw_scores)
    raw_energy = coordinator.energy(raw_scores)
    relaxed = coordinator.relax_etas(list(raw_scores), steps=steps)
    relaxed_values = [float(value) for value in relaxed]
    relaxed_violations = _violations(relaxed_values)
    relaxed_energy = coordinator.energy(relaxed_values)
    return {
        "raw": raw_scores,
        "relaxed": relaxed_values,
        "raw_violations": raw_violations,
        "relaxed_violations": relaxed_violations,
        "raw_energy": raw_energy,
        "relaxed_energy": relaxed_energy,
    }


def _format_vector(values: List[float]) -> str:
    """Format a vector for deterministic console output."""
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def main() -> None:
    """Print the before/after constraint-correction report."""
    result = run_constraint_correction()
    print("=" * 72)
    print("NEURO-SYMBOLIC CONSTRAINT CORRECTION DEMO")
    print("=" * 72)
    print("System-1 raw output:")
    print(f"  [action_a, action_b, low_risk, high_risk] = {_format_vector(result['raw'])}")
    print("Relaxed output:")
    print(f"  [action_a, action_b, low_risk, high_risk] = {_format_vector(result['relaxed'])}")
    print()
    print("Constraint violations:")
    for key, before in result["raw_violations"].items():
        after = result["relaxed_violations"][key]
        print(f"  {key}: {before:.6f} -> {after:.6f}")
    print()
    print(f"Energy: {result['raw_energy']:.6f} -> {result['relaxed_energy']:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
