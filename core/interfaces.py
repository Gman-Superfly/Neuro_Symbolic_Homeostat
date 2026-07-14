"""Interfaces for energy-coordinated micro-modules.

Exposes strict typed Protocols for modules and couplings.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable, Tuple

__all__ = [
    "OrderParameter",
    "EnergyModule",
    "EnergyCoupling",
    "SupportsLocalEnergyGrad",
    "SupportsCouplingGrads",
    "SupportsCouplingCurvature",
    "SupportsPrecision",
    "WeightAdapter",
]

OrderParameter = float


@runtime_checkable
class EnergyModule(Protocol):
    """A small module exposing an order parameter and a local energy."""

    def compute_eta(self, x: Any) -> OrderParameter:
        """Compute the order parameter η for the given input."""
        ...

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        """Compute the module's local free energy F(η; c)."""
        ...


@runtime_checkable
class EnergyCoupling(Protocol):
    """Sparse coupling between two modules' order parameters."""

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        """Energy contribution from coupling two order parameters."""
        ...


@runtime_checkable
class SupportsLocalEnergyGrad(Protocol):
    """Optional analytic derivative of local energy with respect to η."""

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        ...


@runtime_checkable
class SupportsCouplingGrads(Protocol):
    """Optional analytic partial derivatives of coupling energy w.r.t. (η_i, η_j)."""

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        ...


@runtime_checkable
class SupportsCouplingCurvature(Protocol):
    """Optional coupling interface exposing row-wise curvature bounds.

    Returns:
        A tuple `(diag_i, diag_j, off_ij)` where `diag_i` and `diag_j`
        bound each endpoint's diagonal Hessian contribution, and `off_ij`
        bounds the absolute off-diagonal Hessian contribution.
    """

    def coupling_curvature_bounds(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float, float]:
        ...


@runtime_checkable
class SupportsPrecision(Protocol):
    """Optional interface for modules exposing local stiffness/curvature.

    The coordinator uses this value for diagonal preconditioning, curvature
    bounds, and precision-scaled exploration.
    """

    def curvature(self, eta: OrderParameter) -> float:
        """Return the local stiffness (second derivative) at ``eta``.

        Larger values produce smaller preconditioned updates and noise scales.
        """
        ...


@runtime_checkable
class WeightAdapter(Protocol):
    """Optional adapter to update term weights based on gradient norms and energy.
    
    Implementations may apply strategies like GradNorm/PCGrad on energy terms.
    """

    def step(
        self,
        term_grad_norms: Mapping[str, float],
        energy: float,
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Return updated weight mapping: keys like 'local:Class', 'coup:Class' -> float weight."""
        ...


