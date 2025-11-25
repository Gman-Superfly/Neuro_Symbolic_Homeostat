from __future__ import annotations

from typing import Tuple


def prox_quadratic_pair(x0: float, y0: float, weight: float, tau: float) -> Tuple[float, float]:
    """Closed-form prox for w*(x-y)^2 + (1/(2τ))||[x;y]-[x0;y0]||^2 with [0,1] box projection."""
    a = 2.0 * weight + (1.0 / tau)
    b = -2.0 * weight
    c = -2.0 * weight
    d = 2.0 * weight + (1.0 / tau)
    det = a * d - b * c
    if det == 0.0:
        return float(max(0.0, min(1.0, x0))), float(max(0.0, min(1.0, y0)))
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    rhs_x = x0 / tau
    rhs_y = y0 / tau
    x = inv_a * rhs_x + inv_b * rhs_y
    y = inv_c * rhs_x + inv_d * rhs_y
    return float(max(0.0, min(1.0, x))), float(max(0.0, min(1.0, y)))


def prox_asym_hinge_pair(
    x0: float,
    y0: float,
    weight: float,
    alpha: float,
    beta: float,
    tau: float,
) -> Tuple[float, float]:
    """Closed-form prox for w*max(0, β y - α x)^2 + (1/(2τ))||[x;y]-[x0;y0]||^2 with [0,1] box."""
    assert alpha >= 0.0 and beta >= 0.0
    gap0 = beta * y0 - alpha * x0
    if gap0 <= 0.0 or weight == 0.0:
        return float(max(0.0, min(1.0, x0))), float(max(0.0, min(1.0, y0)))
    lam = 2.0 * weight
    denom = 1.0 + tau * lam * (beta * beta + alpha * alpha)
    g_star = (beta * y0 - alpha * x0) / denom
    x = x0 + tau * lam * alpha * g_star
    y = y0 - tau * lam * beta * g_star
    return float(max(0.0, min(1.0, x))), float(max(0.0, min(1.0, y)))


def prox_linear_gate(eta0: float, coeff: float, tau: float) -> float:
    """Prox for linear term -coeff * eta with box projection: minimize -c*η + (1/2τ)(η-η0)^2."""
    # derivative: -(coeff) + (1/τ)(η - η0) = 0 => η = η0 + τ*coeff
    eta = eta0 + tau * coeff
    return float(max(0.0, min(1.0, eta)))


