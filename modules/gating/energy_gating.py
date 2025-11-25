"""Energy-gated expansion aligned with non-local, free-energy coordination.

η_gate is a one-step open probability derived from a non-negative hazard λ(net):
    net = gain - cost
    λ = softplus(k * net)
    η_gate = 1 - exp(-λ)   # memoryless open probability over Δt=1

Local energy discourages casual opening:
    F_gate(η) = a η^2 + b η^4  with a, b ≥ 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Callable
import math

from core.interfaces import EnergyModule, OrderParameter

GainFn = Callable[[Any], float]

__all__ = ["EnergyGatingModule"]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _softplus(x: float) -> float:
    # numerically stable softplus
    if x > 20.0:
        return x
    if x < -20.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


@dataclass
class EnergyGatingModule(EnergyModule):
    """Hazard-based gate; η_gate is P(open in one step).

    - gain_fn should return a positive value when expansion improves order (e.g., η_after - η_before).
    - cost models external constraint/penalty for expansion.
    """
    gain_fn: GainFn
    cost: float = 0.1
    k: float = 10.0  # slope; larger → crisper gating
    a: float = 0.1   # local energy weights
    b: float = 0.1
    use_hazard: bool = True  # if False, fall back to logistic σ(k*net)
    # Optional straight-through estimator (hard forward, soft formula retained internally)
    straight_through: bool = False
    st_threshold: float = 0.5

    def compute_eta(self, x: Any) -> OrderParameter:
        gain = float(self.gain_fn(x))
        net = gain - float(self.cost)
        if self.use_hazard:
            lam = _softplus(self.k * net)
            # one-step open probability from hazard
            eta_soft = 1.0 - math.exp(-lam)
        else:
            # logistic fallback
            eta_soft = _sigmoid(self.k * net)
        # Optional straight-through: hard forward decision for measurement/attribution paths
        if self.straight_through:
            eta_hard = 1.0 if eta_soft >= float(self.st_threshold) else 0.0
            eta = eta_hard
        else:
            eta = eta_soft
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        return float(eta)

    def hazard_rate(self, x: Any) -> float:
        """λ(net) ≥ 0 instantaneous expansion rate (per step)."""
        gain = float(self.gain_fn(x))
        net = gain - float(self.cost)
        lam = _softplus(self.k * net)
        assert lam >= 0.0 and math.isfinite(lam), "Invalid hazard rate"
        return float(lam)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        a = float(constraints.get("gate_alpha", self.a))
        b = float(constraints.get("gate_beta", self.b))
        assert a >= 0.0 and b >= 0.0, "alpha/beta must be non-negative"
        # Landau-like around zero: small η preferred unless justified by coupling/benefit
        return float(a * (eta ** 2) + b * (eta ** 4))

    # Optional analytic derivative for coordinator (duck-typed)
    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        """Analytic derivative d/dη of local energy a η^2 + b η^4 = 2aη + 4bη^3."""
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        a = float(constraints.get("gate_alpha", self.a))
        b = float(constraints.get("gate_beta", self.b))
        assert a >= 0.0 and b >= 0.0, "alpha/beta must be non-negative"
        return float(2.0 * a * eta + 4.0 * b * (eta ** 3))


