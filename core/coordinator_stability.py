"""Stability and Lipschitz-bound helpers for energy relaxation."""

from __future__ import annotations

from typing import Any, List, Sequence

import math
import numpy as np

from .couplings import AsymmetricHingeCoupling, DirectedHingeCoupling, QuadraticCoupling
from .interfaces import OrderParameter, SupportsCouplingCurvature

__all__ = [
    "curvature_bound_components",
    "estimate_lipschitz_bound",
    "estimate_preconditioned_lipschitz_bound",
    "validate_curvature_bound_triplet",
]


def validate_curvature_bound_triplet(
    values: Sequence[float],
) -> tuple[float, float, float]:
    """Validate one coupling's non-negative curvature-bound report.

    Positive infinity is retained to represent a genuinely unbounded global
    curvature contract. NaN and negative values are invalid because silently
    clipping either could turn an underreported bound into a false certificate.
    """
    bounds = np.asarray(values, dtype=float)
    if bounds.shape != (3,):
        raise ValueError("coupling curvature bounds must contain exactly three values")
    if np.any(np.isnan(bounds)) or np.any(bounds < 0.0):
        raise ValueError("coupling curvature bounds must be non-negative and not NaN")
    return float(bounds[0]), float(bounds[1]), float(bounds[2])


def curvature_bound_components(
    coordinator: Any,
    etas: List[OrderParameter],
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Return diagonal and absolute off-diagonal Hessian bounds.

    The returned edge entries may contain repeated ``(i, j)`` pairs. Summing
    their contributions produces a Gershgorin row bound without materializing
    a dense Hessian.
    """
    n = len(etas)
    if n == 0:
        return np.zeros(0, dtype=float), []
    diag = np.zeros(n, dtype=float)
    offdiag: list[tuple[int, int, float]] = []
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
        if math.isnan(curv):
            raise ValueError("local curvature estimate must not be NaN")
        diag[i] += abs(float(curv))

    combined_weights = coordinator._combined_term_weights()  # noqa: SLF001
    for i, j, coupling in coordinator.couplings:
        key = f"coup:{coupling.__class__.__name__}"
        w_eff = float(combined_weights.get(key, 1.0))
        if w_eff == 0.0:
            continue
        if isinstance(coupling, QuadraticCoupling):
            w = abs(float(getattr(coupling, "weight", 0.0)) * w_eff)
            diag[i] += 2.0 * w
            diag[j] += 2.0 * w
            offdiag.append((i, j, 2.0 * w))
        elif isinstance(coupling, DirectedHingeCoupling):
            w = abs(float(getattr(coupling, "weight", 0.0)) * w_eff)
            # The proposal can cross the hinge boundary, so use the maximum
            # curvature over both active regions rather than the starting set.
            diag[i] += 2.0 * w
            diag[j] += 2.0 * w
            offdiag.append((i, j, 2.0 * w))
        elif isinstance(coupling, AsymmetricHingeCoupling):
            w = abs(float(getattr(coupling, "weight", 0.0)) * w_eff)
            alpha = float(getattr(coupling, "alpha_i", 1.0))
            beta = float(getattr(coupling, "beta_j", 1.0))
            diag[i] += 2.0 * w * (alpha * alpha)
            diag[j] += 2.0 * w * (beta * beta)
            offdiag.append((i, j, 2.0 * w * abs(alpha * beta)))
        elif isinstance(coupling, SupportsCouplingCurvature):
            diag_i, diag_j, off_ij = validate_curvature_bound_triplet(
                coupling.coupling_curvature_bounds(
                    float(etas[i]),
                    float(etas[j]),
                    coordinator.constraints,
                )
            )
            weight_scale = abs(w_eff)
            diag[i] += weight_scale * diag_i
            diag[j] += weight_scale * diag_j
            offdiag.append((i, j, weight_scale * off_ij))
        else:
            # An unreported custom edge has unknown curvature. Represent that
            # uncertainty as an infinite bound so a guarded fixed step fails
            # closed while projected Armijo remains available.
            diag[i] = math.inf
            diag[j] = math.inf

    return diag, offdiag


def estimate_lipschitz_bound(coordinator: Any, etas: List[OrderParameter]) -> float:
    """Estimate a Gershgorin row bound for the unpreconditioned Hessian."""
    diag, offdiag = curvature_bound_components(coordinator, etas)
    if diag.size == 0:
        return 0.0
    row_sums = diag.copy()
    for i, j, off in offdiag:
        row_sums[i] += off
        row_sums[j] += off

    estimate = float(np.max(row_sums))
    if math.isnan(estimate):
        raise ValueError("Lipschitz estimate must not be NaN")
    if estimate <= 0.0:
        return 0.0
    return estimate


def estimate_preconditioned_lipschitz_bound(
    coordinator: Any,
    etas: List[OrderParameter],
    preconditioner: Sequence[float],
) -> float:
    """Bound ``lambda_max(P^-1/2 H P^-1/2)`` by Gershgorin rows.

    ``preconditioner`` must be the positive diagonal actually used by the
    update. For an SPD quadratic, ``0 < alpha < 2 / estimate`` makes the
    preconditioned affine step contractive in the P norm.
    """
    diag, offdiag = curvature_bound_components(coordinator, etas)
    if diag.size == 0:
        return 0.0
    precision = np.asarray(preconditioner, dtype=float)
    if precision.shape != diag.shape:
        raise ValueError("preconditioner shape must match the coordinator state")
    if not np.all(np.isfinite(precision)) or np.any(precision <= 0.0):
        raise ValueError("preconditioner entries must be finite and positive")

    row_sums = diag / precision
    for i, j, off in offdiag:
        normalized_off = off / math.sqrt(float(precision[i] * precision[j]))
        row_sums[i] += normalized_off
        row_sums[j] += normalized_off

    estimate = float(np.max(row_sums))
    if math.isnan(estimate):
        raise ValueError("preconditioned Lipschitz estimate must not be NaN")
    if estimate <= 0.0:
        return 0.0
    return estimate
