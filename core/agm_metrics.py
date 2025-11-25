"""AGM phase metrics and uncertainty utilities.

Implements light-weight, numerically stable metrics to characterize convergence
phases and produce uncertainty summaries for adaptive control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _safe_mean(xs: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray(list(xs), dtype=float)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def _safe_var(xs: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray(list(xs), dtype=float)
    if arr.size <= 1:
        return float(default)
    return float(np.var(arr))


def compute_agm_phase_metrics(values: Sequence[float]) -> Dict[str, float]:
    """Compute AGM-inspired phase metrics from a scalar history (e.g., energy).

    Returns:
        dict with keys:
            - rate:    proxy for convergence speed in [0, 1]
            - variance: variance of recent rates
            - trend:   slope of recent rates (positive => improving)
            - oscillation: variance of rate changes (instability proxy)
    """
    # Require at least 2 points to form a pair
    if len(values) < 2:
        return {"rate": 0.0, "variance": 0.0, "trend": 0.0, "oscillation": 0.0}

    # Build pairwise "rates" = 1 - |A-H| / (A+H), using adjacent points
    # Here we approximate A,H via consecutive values as a simple proxy.
    # This retains the intended shape: closer consecutive values => higher "rate".
    rates: List[float] = []
    for i in range(len(values) - 1):
        a = float(values[i])
        h = float(values[i + 1])
        denom = abs(a) + abs(h) + 1e-8
        rate_i = 1.0 - abs(a - h) / denom
        # Clamp to [0,1] for interpretability
        rate_i = max(0.0, min(1.0, rate_i))
        rates.append(rate_i)

    rate = _safe_mean(rates, default=0.0)
    variance = _safe_var(rates, default=0.0)

    if len(rates) >= 5:
        x = np.arange(5, dtype=float)
        y = np.asarray(rates[-5:], dtype=float)
        # slope of linear fit
        coeffs = np.polyfit(x, y, 1)
        trend = float(coeffs[0])
    else:
        trend = 0.0

    if len(rates) >= 2:
        diffs = [abs(rates[i] - rates[i - 1]) for i in range(1, len(rates))]
        oscillation = _safe_var(diffs, default=0.0)
    else:
        oscillation = 0.0

    return {
        "rate": float(rate),
        "variance": float(variance),
        "trend": float(trend),
        "oscillation": float(oscillation),
    }


@dataclass
class UncertaintySummary:
    epistemic: float
    aleatoric: float
    total: float
    exploration_boost: float


def compute_uncertainty_metrics(
    energy_history: Sequence[float],
    recent_performance: Sequence[float] | None = None,
) -> UncertaintySummary:
    """Compute simple uncertainty signals for adaptive exploration/exploitation.

    Args:
        energy_history: recent scalar values (e.g., accepted energies).
        recent_performance: optional recent task success rates or metrics.
    """
    metrics = compute_agm_phase_metrics(energy_history)
    variance_perf = _safe_var(recent_performance or [], default=0.0)

    # Epistemic uncertainty: higher when rate is low, oscillation high
    epistemic = min(1.0, (1.0 - metrics["rate"]) + metrics["oscillation"])
    # Aleatoric uncertainty: proxy via performance variance
    aleatoric = min(1.0, variance_perf)
    total = float(epistemic + aleatoric)

    # Exploration recommendation
    exploration_boost = 0.0
    if total > 1.0:
        exploration_boost = 0.05
    elif total < 0.3:
        exploration_boost = -0.02
    return UncertaintySummary(
        epistemic=float(epistemic),
        aleatoric=float(aleatoric),
        total=float(total),
        exploration_boost=float(exploration_boost),
    )


__all__ = ["compute_agm_phase_metrics", "compute_uncertainty_metrics", "UncertaintySummary"]

