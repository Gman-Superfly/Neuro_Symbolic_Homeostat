"""Stability and Lipschitz-bound helpers for energy relaxation."""

from __future__ import annotations

from typing import Any, List

import math
import numpy as np

from .couplings import AsymmetricHingeCoupling, DirectedHingeCoupling, QuadraticCoupling
from .interfaces import OrderParameter, SupportsCouplingCurvature

__all__ = ["estimate_lipschitz_bound"]


def estimate_lipschitz_bound(coordinator: Any, etas: List[OrderParameter]) -> float:
    """Estimate a conservative Gershgorin-style gradient Lipschitz bound."""
    n = len(etas)
    if n == 0:
        return 0.0
    diag = np.zeros(n, dtype=float)
    offsum = np.zeros(n, dtype=float)
    eps = max(coordinator.grad_eps * 0.5, 1e-6)

    for i in range(n):
        eta_i = float(etas[i])
        # Clamp the probe points to the [0, 1] box, then normalize by the actual
        # window width. At a box edge the window shrinks to one side; dividing by
        # the true width keeps the curvature estimate accurate. Dividing by the
        # nominal 2*eps there would halve the estimate, underbound L, and break
        # the descent precondition for the 0.9 * 2/L step cap.
        lo = max(0.0, min(1.0, eta_i - eps))
        hi = max(0.0, min(1.0, eta_i + eps))
        width = hi - lo
        if width <= 0.0:
            continue
        g_m = coordinator._local_grad(i, lo)  # noqa: SLF001
        g_p = coordinator._local_grad(i, hi)  # noqa: SLF001
        curv = (g_p - g_m) / width
        if math.isfinite(curv) and curv > 0.0:
            diag[i] += float(curv)

    combined_weights = coordinator._combined_term_weights()  # noqa: SLF001
    for i, j, coupling in coordinator.couplings:
        key = f"coup:{coupling.__class__.__name__}"
        w_eff = float(combined_weights.get(key, 1.0))
        if isinstance(coupling, QuadraticCoupling):
            w = float(getattr(coupling, "weight", 0.0)) * w_eff
            diag[i] += 2.0 * w
            diag[j] += 2.0 * w
            offsum[i] += 2.0 * w
            offsum[j] += 2.0 * w
        elif isinstance(coupling, DirectedHingeCoupling):
            w = float(getattr(coupling, "weight", 0.0)) * w_eff
            gap = float(etas[j]) - float(etas[i])
            if gap > 0.0:
                diag[i] += 2.0 * w
                diag[j] += 2.0 * w
                offsum[i] += 2.0 * w
                offsum[j] += 2.0 * w
        elif isinstance(coupling, AsymmetricHingeCoupling):
            w = float(getattr(coupling, "weight", 0.0)) * w_eff
            alpha = float(getattr(coupling, "alpha_i", 1.0))
            beta = float(getattr(coupling, "beta_j", 1.0))
            gap = beta * float(etas[j]) - alpha * float(etas[i])
            if gap > 0.0:
                diag[i] += 2.0 * w * (alpha * alpha)
                diag[j] += 2.0 * w * (beta * beta)
                offsum[i] += 2.0 * w * abs(alpha * beta)
                offsum[j] += 2.0 * w * abs(alpha * beta)
        elif isinstance(coupling, SupportsCouplingCurvature):
            diag_i, diag_j, off_ij = coupling.coupling_curvature_bounds(
                float(etas[i]),
                float(etas[j]),
                coordinator.constraints,
            )
            diag[i] += w_eff * max(0.0, float(diag_i))
            diag[j] += w_eff * max(0.0, float(diag_j))
            off = w_eff * max(0.0, abs(float(off_ij)))
            offsum[i] += off
            offsum[j] += off

    estimate = float(np.max(diag + offsum))
    if not math.isfinite(estimate) or estimate <= 0.0:
        return 0.0
    return estimate
