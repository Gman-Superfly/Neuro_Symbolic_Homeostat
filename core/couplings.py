"""Standard coupling energies between order parameters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Tuple

from .interfaces import EnergyCoupling, OrderParameter

__all__ = [
    "QuadraticCoupling",
    "DirectedHingeCoupling",
    "AsymmetricHingeCoupling",
    "GateBenefitCoupling",
    "DampedGateBenefitCoupling",
]


@dataclass(frozen=True)
class QuadraticCoupling(EnergyCoupling):
    """Symmetric quadratic coupling: w * (η_i - η_j)^2."""
    weight: float = 1.0

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        assert self.weight >= 0.0, "weight must be non-negative"
        diff = float(eta_i) - float(eta_j)
        return float(self.weight * (diff * diff))

    # Optional analytic gradients
    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        diff = float(eta_i) - float(eta_j)
        gi = float(2.0 * self.weight * diff)
        gj = float(-2.0 * self.weight * diff)
        return gi, gj


@dataclass(frozen=True)
class DirectedHingeCoupling(EnergyCoupling):
    """Directed hinge-like coupling: w * max(0, η_j - η_i)^2.
    
    Encourages η_i to not fall below η_j.
    """
    weight: float = 1.0

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        assert self.weight >= 0.0, "weight must be non-negative"
        gap = max(0.0, float(eta_j) - float(eta_i))
        return float(self.weight * (gap * gap))

    # Optional analytic gradients
    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        gap = float(eta_j) - float(eta_i)
        if gap > 0.0:
            gi = float(-2.0 * self.weight * gap)
            gj = float(2.0 * self.weight * gap)
        else:
            gi = 0.0
            gj = 0.0
        return gi, gj


@dataclass(frozen=True)
class AsymmetricHingeCoupling(EnergyCoupling):
    """Asymmetric hinge coupling: w * max(0, β η_j - α η_i)^2.
    
    Scales contributions of i and j inside the hinge to bias directionality.
    """
    weight: float = 1.0
    alpha_i: float = 1.0  # scale on eta_i
    beta_j: float = 1.0   # scale on eta_j

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        assert self.weight >= 0.0, "weight must be non-negative"
        assert self.alpha_i >= 0.0 and self.beta_j >= 0.0, "scales must be non-negative"
        gap = self.beta_j * float(eta_j) - self.alpha_i * float(eta_i)
        if gap <= 0.0:
            return 0.0
        return float(self.weight * (gap * gap))

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        gap = self.beta_j * float(eta_j) - self.alpha_i * float(eta_i)
        if gap <= 0.0:
            return 0.0, 0.0
        gi = float(-2.0 * self.weight * gap * self.alpha_i)
        gj = float(2.0 * self.weight * gap * self.beta_j)
        return gi, gj

@dataclass(frozen=True)
class GateBenefitCoupling(EnergyCoupling):
    """Coupling that rewards opening a gate when domain improvement exists.
    
    Energy: F = - w * η_gate * Δη_domain
    Where:
      - η_gate in [0,1]
      - Δη_domain provided via constraints under key `delta_eta_domain`
    
    Note: Caller is responsible for computing Δη_domain (e.g., η_after - η_before)
    based on current inputs. This keeps the coupling generic and non-invasive.
    """
    weight: float = 1.0
    delta_key: str = "delta_eta_domain"

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        assert self.weight >= 0.0, "weight must be non-negative"
        # interpret eta_i as gate, eta_j as domain eta (unused directly)
        delta = float(constraints.get(self.delta_key, 0.0))
        eta_gate = float(eta_i)
        return float(-self.weight * eta_gate * delta)

    # Optional analytic gradients
    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        delta = float(constraints.get(self.delta_key, 0.0))
        gi = float(-self.weight * delta)
        gj = 0.0
        return gi, gj

    def coupling_curvature_bounds(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float, float]:
        """Return the exact zero Hessian of the frozen linear gate term."""
        del eta_i, eta_j, constraints
        return 0.0, 0.0, 0.0


@dataclass(frozen=True)
class DampedGateBenefitCoupling(EnergyCoupling):
    """Gate-benefit coupling with optional damping and asymmetric scaling.

    Energy: F = - w * (η_gate ** eta_power) * damping * scale(delta) * delta
    where scale(delta) applies positive/negative multipliers before damping.
    """

    weight: float = 1.0
    delta_key: str = "delta_eta_domain"
    damping: float = 1.0
    eta_power: float = 1.0
    positive_scale: float = 1.0
    negative_scale: float = 1.0

    def __post_init__(self) -> None:
        parameters = {
            "weight": self.weight,
            "damping": self.damping,
            "eta_power": self.eta_power,
            "positive_scale": self.positive_scale,
            "negative_scale": self.negative_scale,
        }
        if not all(math.isfinite(float(value)) for value in parameters.values()):
            raise ValueError("damped gate-benefit parameters must be finite")
        if self.weight < 0.0 or self.damping < 0.0:
            raise ValueError("weight and damping must be non-negative")
        if self.positive_scale < 0.0 or self.negative_scale < 0.0:
            raise ValueError("positive_scale and negative_scale must be non-negative")
        if self.eta_power < 1.0:
            raise ValueError("eta_power must be at least 1.0 on the closed unit box")

    def _scaled_delta(self, delta: float) -> float:
        if delta >= 0.0:
            return self.positive_scale * delta
        return self.negative_scale * delta

    def _effective_eta(self, eta_gate: float) -> float:
        if self.eta_power == 1.0:
            return eta_gate
        assert eta_gate >= 0.0, "eta must be non-negative"
        return float(max(0.0, eta_gate) ** self.eta_power)

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        assert self.weight >= 0.0, "weight must be non-negative"
        assert self.damping >= 0.0, "damping must be non-negative"
        delta = float(constraints.get(self.delta_key, 0.0))
        scaled_delta = self._scaled_delta(delta)
        eta_gate = float(eta_i)
        eta_eff = self._effective_eta(eta_gate)
        return float(-self.weight * self.damping * eta_eff * scaled_delta)

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        delta = float(constraints.get(self.delta_key, 0.0))
        scaled_delta = self._scaled_delta(delta)
        if scaled_delta == 0.0 or self.weight == 0.0 or self.damping == 0.0:
            return 0.0, 0.0
        eta_gate = float(eta_i)
        if self.eta_power == 1.0:
            grad = -self.weight * self.damping * scaled_delta
        else:
            if eta_gate <= 0.0:
                grad = 0.0
            else:
                grad = -self.weight * self.damping * scaled_delta * self.eta_power * (
                    eta_gate ** (self.eta_power - 1.0)
                )
        return float(grad), 0.0

    def coupling_curvature_bounds(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float, float]:
        """Return a box-wide absolute Hessian bound for the gate coordinate.

        For ``1 < eta_power < 2`` the second derivative diverges as the gate
        approaches zero, so no finite gradient-Lipschitz bound exists on the
        closed unit interval. The infinite result makes that limitation
        explicit to the stability guard; projected Armijo search remains the
        supported gradient-solver path for those powers.
        """
        del eta_i, eta_j
        scaled_delta = self._scaled_delta(float(constraints.get(self.delta_key, 0.0)))
        coefficient = self.weight * self.damping * scaled_delta
        if coefficient == 0.0 or self.eta_power == 1.0:
            return 0.0, 0.0, 0.0
        if self.eta_power < 2.0:
            return math.inf, 0.0, 0.0
        bound = abs(coefficient) * self.eta_power * (self.eta_power - 1.0)
        return float(bound), 0.0, 0.0


