"""Telemetry extensions archived after their providers left the active package."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def collect_legacy_scalar_metrics(coordinator: Any) -> Dict[str, float]:
    """Collect optional values used by the former observability branches."""
    metrics: Dict[str, float] = {}
    providers = {
        "sensitivity:dispersion": "last_probe_dispersion",
        "escape_event_count": "get_escape_event_count",
        "confidence:c": "last_confidence",
    }
    for output_key, method_name in providers.items():
        method = getattr(coordinator, method_name, None)
        if callable(method):
            value = method()
            if value is not None:
                metrics[output_key] = float(value)
    for output_key, attribute in {
        "homotopy_scale": "_homotopy_scale",
        "homotopy_backoffs": "_homotopy_backoffs",
    }.items():
        value = getattr(coordinator, attribute, None)
        if isinstance(value, (int, float)):
            metrics[output_key] = float(value)
    return metrics


def maximum_basis_correlation(history: Sequence[Sequence[float]]) -> float:
    """Return the largest absolute off-diagonal basis correlation."""
    values = np.asarray(history, dtype=float)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 2:
        return 0.0
    correlation = np.corrcoef(values, rowvar=False)
    np.fill_diagonal(correlation, 0.0)
    return float(np.max(np.abs(np.nan_to_num(correlation))))
