"""Unused redundancy-provider protocol archived from core.info_metrics."""

from typing import Any, Protocol


class SupportsRedundancy(Protocol):
    def compute_redundancy(self, source_eta: float, context: Any) -> float:
        ...
