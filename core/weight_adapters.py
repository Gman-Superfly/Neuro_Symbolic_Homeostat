"""Weight adapter implementations for coordinating term balances.

Contains reactive adapters that satisfy the WeightAdapter protocol and can be
plugged into EnergyCoordinator to keep term contributions well-behaved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, TYPE_CHECKING

from .interfaces import WeightAdapter
from .agm_metrics import compute_agm_phase_metrics


@dataclass
class GradNormWeightAdapter:
    """GradNorm-style balancing for energy terms.

    Args:
        target_norm: Desired L2 norm per term (defaults to 1.0).
        alpha: Restoring-force strength; larger => faster corrections.
        update_rate: Fraction of the adjustment applied each step (0-1].
        floor: Minimum allowed weight returned by the adapter.
        ceiling: Maximum allowed weight; set None to disable.
        eps: Numerical guard to avoid division by zero.
    """

    target_norm: float = 1.0
    alpha: float = 1.5
    update_rate: float = 0.1
    floor: float = 0.1
    ceiling: float | None = 2.0
    eps: float = 1e-9

    def __post_init__(self) -> None:
        assert self.target_norm > 0.0, "target_norm must be positive"
        assert 0.0 < self.update_rate <= 1.0, "update_rate must be in (0, 1]"
        assert self.floor >= 0.0, "floor must be non-negative"
        if self.ceiling is not None:
            assert self.ceiling >= self.floor, "ceiling must be >= floor"

    def step(
        self,
        term_grad_norms: Mapping[str, float],
        energy: float,  # noqa: ARG002 - required by protocol
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        assert term_grad_norms, "GradNormWeightAdapter requires gradient norms"
        updated: Dict[str, float] = {str(k): float(v) for k, v in current.items()}
        target = max(self.target_norm, self.eps)
        finite_norms = [float(n) for n in term_grad_norms.values() if self._is_valid(n)]
        average_norm = float(sum(finite_norms) / len(finite_norms)) if finite_norms else target
        for key, raw_norm in term_grad_norms.items():
            norm = float(raw_norm) if self._is_valid(raw_norm) else average_norm
            ratio = norm / target
            adjustment = self.alpha * (1.0 - ratio)
            weight = float(updated.get(key, 1.0))
            weight *= 1.0 + self.update_rate * adjustment
            weight = self._clamp(weight)
            updated[key] = weight
        return updated

    def _is_valid(self, value: float) -> bool:
        return value >= 0.0 and value == value  # excludes NaNs without importing math

    def _clamp(self, value: float) -> float:
        v = max(value, self.floor)
        if self.ceiling is not None:
            v = min(v, self.ceiling)
        return v


__all__ = ["GradNormWeightAdapter"]


@dataclass
class AGMPhaseWeightAdapter:
    """Phase-adaptive weighting using AGM-style metrics on energy history.

    Policy (simple, conservative):
      - If rate is high and trend positive: slightly increase coupling weights,
        slightly reduce gate local energy weight (exploitation).
      - If rate is low or oscillation high: slightly reduce coupling weights,
        slightly increase gate local energy weight (exploration/regularization).

    Adjustments are multiplicative and gentle to avoid violent swings.
    """

    increase_factor: float = 1.05
    decrease_factor: float = 0.97
    gate_local_key: str = "local:EnergyGatingModule"
    energy_history: List[float] = field(default_factory=list)

    def step(
        self,
        term_grad_norms: Mapping[str, float],  # noqa: ARG002 - protocol compatibility
        energy: float,
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        # Append energy for phase assessment
        self.energy_history.append(float(energy))
        metrics = compute_agm_phase_metrics(self.energy_history)
        updated: Dict[str, float] = {str(k): float(v) for k, v in current.items()}

        # Identify coupling keys (simple heuristic: prefix "coup:")
        coupling_keys = [k for k in updated.keys() if k.startswith("coup:")]

        # Decide a regime
        rate = metrics["rate"]
        trend = metrics["trend"]
        oscillation = metrics["oscillation"]

        if rate > 0.7 and trend > 0.0 and oscillation < 0.05:
            # Exploitation: favor couplings, soften gate local slightly
            for k in coupling_keys:
                updated[k] = updated.get(k, 1.0) * self.increase_factor
            updated[self.gate_local_key] = updated.get(self.gate_local_key, 1.0) * self.decrease_factor
        elif rate < 0.3 or oscillation > 0.1:
            # Exploration or unstable: favor regularization via gate local, tame couplings
            for k in coupling_keys:
                updated[k] = updated.get(k, 1.0) * self.decrease_factor
            updated[self.gate_local_key] = updated.get(self.gate_local_key, 1.0) * self.increase_factor
        # Else keep weights unchanged

        return updated


__all__.extend(["AGMPhaseWeightAdapter"])





@dataclass
class SmallGainWeightAdapter:
    """Per-edge stability-margin allocator with row-aware greedy budgeting.

    Production intent: conservative, monotone-compatible allocator that spends a fraction
    of the available contractivity budget (global and per-row) to boost coupling families
    with the highest expected ΔF per ΔL ratio.

    Coordinator integration:
      - Before calling step(...), the coordinator should populate:
          self.edge_costs: Dict[str, float]    # ΔL per coupling key (e.g., 'coup:ClassName')
          self.row_margins: Dict[int, float]   # per-row margins m_r
          self.global_margin: float            # global margin
      - These are treated as snapshots for the current step only.

    Bounded updates and smoothing:
      - Per-step weight change is limited to ±max_step_change.
      - Scores (value/cost) use EMA to reduce noise.

    Observability:
      - last_allocations: Dict[str, float] of Δweight applied per key (for logging)
      - last_spent_global: float
      - last_spent_row: Dict[int, float]
    """

    # Budgeting
    budget_fraction: float = 0.7
    max_step_change: float = 0.10

    # Bounds and smoothing
    floor: float = 0.1
    ceiling: float = 3.0
    ema_alpha: float = 0.3
    eps: float = 1e-9

    # Snapshots injected by coordinator per step
    edge_costs: Dict[str, float] = field(default_factory=dict)      # ΔL per coupling key
    row_margins: Dict[int, float] = field(default_factory=dict)     # per-row margins
    global_margin: float = 0.0

    # Adapter state
    scores: Dict[str, float] = field(default_factory=dict)          # EMA(value/cost)

    # Observability
    last_allocations: Dict[str, float] = field(default_factory=dict)
    last_spent_global: float = 0.0
    last_spent_row: Dict[int, float] = field(default_factory=dict)

    def step(
        self,
        term_grad_norms: Mapping[str, float],
        energy: float,  # noqa: ARG002 - reserved for future use / compatibility
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        # Reset observability
        self.last_allocations = {}
        self.last_spent_global = 0.0
        self.last_spent_row = {r: 0.0 for r in self.row_margins.keys()}

        # Build values for coupling families that exist in term_grad_norms
        values: Dict[str, float] = {}
        for k, v in term_grad_norms.items():
            if isinstance(k, str) and k.startswith("coup:"):
                try:
                    values[k] = float(v) * float(v)  # grad_norm^2
                except Exception:
                    continue

        if not values:
            # Nothing to do
            return dict(current)

        # Costs per family (ΔL per unit relative scaling)
        costs: Dict[str, float] = {}
        for k in values.keys():
            c = self.edge_costs.get(k, 0.0)
            costs[k] = float(c) if c == c else 0.0  # guard NaN

        # Update EMA scores (value/cost)
        for k in values.keys():
            denom = costs[k] + self.eps
            raw = values[k] / denom if denom > 0.0 else values[k]
            old = self.scores.get(k, raw)
            self.scores[k] = self.ema_alpha * raw + (1.0 - self.ema_alpha) * old

        # Rank by score
        ranked = sorted(values.keys(), key=lambda kk: self.scores.get(kk, 0.0), reverse=True)

        # Budgets
        row_budget = {r: max(0.0, float(m)) * self.budget_fraction for r, m in self.row_margins.items()}
        global_budget = max(0.0, float(self.global_margin)) * self.budget_fraction

        # Prepare output mapping
        updated: Dict[str, float] = {str(k): float(v) for k, v in current.items()}

        # Greedy allocation
        for k in ranked:
            # Stop if global budget exhausted
            if global_budget - self.last_spent_global <= 0.0:
                break

            fam_key = k
            w_old = float(updated.get(fam_key, 1.0))
            w_new = w_old

            # Propose an increase within bounds
            proposed = min(self.ceiling, max(self.floor, w_old * (1.0 + self.max_step_change)))
            delta_w = proposed - w_old
            if delta_w <= 0.0:
                continue

            # Linearized ΔL spend for this increment
            cost = costs.get(k, 0.0)
            # Approximate proportionality of ΔL with relative scaling
            denom_scale = max(w_old, self.eps)
            delta_L = cost * (delta_w / denom_scale)

            # Check global budget
            if self.last_spent_global + delta_L > global_budget:
                continue

            # For a first implementation, we do not enforce per-row incidence booking here,
            # as row attribution requires edge->row mapping. Keep global cap conservative.
            # Future: accept an injected edge->rows map and update self.last_spent_row accordingly.

            # Commit
            w_new = proposed
            updated[fam_key] = w_new
            self.last_allocations[fam_key] = delta_w
            self.last_spent_global += delta_L

        return updated


__all__.extend(["SmallGainWeightAdapter"])
