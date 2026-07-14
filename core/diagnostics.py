"""Public diagnostic records for coordinator state inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class CoordinatorSnapshot:
    """Side-effect-free diagnostic view of one coordinator state."""

    etas: Tuple[float, ...]
    energy: float
    gradient: Tuple[float, ...]
    precision_diagonal: Tuple[float, ...]
    lipschitz_bound: float
    term_weights: Mapping[str, float]
    term_gradient_norms: Mapping[str, float]
    objective_version: int
