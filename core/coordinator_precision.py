"""Precision-cache helpers for energy relaxation."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .couplings import AsymmetricHingeCoupling, DirectedHingeCoupling, QuadraticCoupling
from .interfaces import OrderParameter, SupportsCouplingCurvature, SupportsPrecision

__all__ = ["update_precision_cache", "get_precision_diagonal"]


def update_precision_cache(coordinator: Any, etas: List[OrderParameter]) -> None:
    """Update diagonal curvature cache from modules and coupling curvature."""
    n = len(etas)
    diag = np.zeros(n, dtype=float)
    cw = coordinator._combined_term_weights()  # noqa: SLF001

    for idx, (module, eta) in enumerate(zip(coordinator.modules, etas)):
        w_loc = float(cw.get(f"local:{module.__class__.__name__}", 1.0))
        if isinstance(module, SupportsPrecision):
            try:
                curv = max(0.0, float(module.curvature(float(eta))))
            except Exception:
                curv = 0.0
        else:
            curv = 0.0
        if w_loc != 0.0 and curv != 0.0:
            diag[idx] += w_loc * curv

    for i, j, coupling in coordinator.couplings:
        key = f"coup:{coupling.__class__.__name__}"
        w_eff = float(cw.get(key, 1.0))
        if w_eff == 0.0:
            continue
        if isinstance(coupling, QuadraticCoupling):
            w = float(getattr(coupling, "weight", 0.0)) * w_eff
            add = 2.0 * w
            if add != 0.0:
                if 0 <= i < n:
                    diag[i] += add
                if 0 <= j < n:
                    diag[j] += add
        elif isinstance(coupling, DirectedHingeCoupling):
            w = float(getattr(coupling, "weight", 0.0)) * w_eff
            gap = float(etas[j]) - float(etas[i])
            if w != 0.0 and gap > 0.0:
                add = 2.0 * w
                if 0 <= i < n:
                    diag[i] += add
                if 0 <= j < n:
                    diag[j] += add
        elif isinstance(coupling, AsymmetricHingeCoupling):
            w = float(getattr(coupling, "weight", 0.0)) * w_eff
            alpha = float(getattr(coupling, "alpha_i", 1.0))
            beta = float(getattr(coupling, "beta_j", 1.0))
            gap = beta * float(etas[j]) - alpha * float(etas[i])
            if w != 0.0 and gap > 0.0:
                add_i = 2.0 * w * (alpha * alpha)
                add_j = 2.0 * w * (beta * beta)
                if 0 <= i < n:
                    diag[i] += add_i
                if 0 <= j < n:
                    diag[j] += add_j
        elif isinstance(coupling, SupportsCouplingCurvature):
            diag_i, diag_j, _ = coupling.coupling_curvature_bounds(float(etas[i]), float(etas[j]), coordinator.constraints)
            if 0 <= i < n:
                diag[i] += w_eff * max(0.0, float(diag_i))
            if 0 <= j < n:
                diag[j] += w_eff * max(0.0, float(diag_j))

    coordinator._precision_cache = {int(idx): float(val) for idx, val in enumerate(diag)}  # noqa: SLF001


def get_precision_diagonal(coordinator: Any) -> List[float]:
    """Return the currently cached diagonal precision vector."""
    if coordinator._precision_cache is None:  # noqa: SLF001
        return [0.0] * len(coordinator.modules)
    return [coordinator._precision_cache.get(i, 0.0) for i in range(len(coordinator.modules))]  # noqa: SLF001
