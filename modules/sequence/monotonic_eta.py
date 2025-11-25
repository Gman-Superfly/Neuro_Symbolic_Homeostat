"""Sequence order parameter via sublinear monotonicity sampling.

Provides:
- sample_monotonicity_score: approximate fraction of non-violations
- SequenceConsistencyModule: EnergyModule wrapper with local energy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import random

import numpy as np

from core.interfaces import EnergyModule, OrderParameter

__all__ = ["sample_monotonicity_score", "SequenceConsistencyModule"]


def sample_monotonicity_score(values: Sequence[float], samples: int = 512, seed: int | None = None) -> float:
    """Approximate monotonic non-decreasing score in [0, 1].
    
    Score is the fraction of sampled pairs (i < j) that satisfy v[i] <= v[j].
    Sublinear: O(samples).
    """
    assert samples > 0, "samples must be positive"
    n = len(values)
    assert n >= 2, "need at least two values"
    rng = random.Random(seed)
    ok = 0
    for _ in range(samples):
        i = rng.randrange(0, n - 1)
        j = rng.randrange(i + 1, n)
        if float(values[i]) <= float(values[j]):
            ok += 1
    return float(ok) / float(samples)


@dataclass
class SequenceConsistencyModule(EnergyModule):
    """Computes η as sublinear monotonicity and local energy around target=1.0."""

    alpha: float = 1.0
    beta: float = 1.0
    samples: int = 512
    seed: int | None = None

    def compute_eta(self, x: Any) -> OrderParameter:
        arr = np.asarray(x, dtype=float).tolist()
        assert isinstance(arr, list) and len(arr) >= 2, "sequence must have at least two numeric elements"
        eta = sample_monotonicity_score(arr, samples=self.samples, seed=self.seed)
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        return float(eta)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        # Landau-like around target order 1.0
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        a = float(constraints.get("seq_alpha", self.alpha))
        b = float(constraints.get("seq_beta", self.beta))
        assert a >= 0.0 and b >= 0.0, "alpha/beta must be non-negative"
        delta = (1.0 - float(eta))
        return float(a * (delta ** 2) + b * (delta ** 4))

    # Optional analytic derivative
    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        a = float(constraints.get("seq_alpha", self.alpha))
        b = float(constraints.get("seq_beta", self.beta))
        assert a >= 0.0 and b >= 0.0, "alpha/beta must be non-negative"
        delta = 1.0 - float(eta)
        return float(-2.0 * a * delta - 4.0 * b * (delta ** 3))


