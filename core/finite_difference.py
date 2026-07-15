"""Box-aware finite-difference primitives shared by all solver paths."""

from __future__ import annotations

from typing import Callable

import math


def box_derivative(function: Callable[[float], float], value: float, epsilon: float) -> float:
    """Differentiate a scalar function without evaluating outside ``[0, 1]``.

    The interior stencil is centered.  Within one configured step of a
    boundary, a second-order three-point one-sided stencil avoids both an
    out-of-domain probe and the first-order bias of a simple forward or
    backward difference.  Each stencil is exact for scalar quadratics up to
    floating-point error.
    """
    x = float(value)
    requested_h = float(epsilon)
    if not math.isfinite(x) or x < 0.0 or x > 1.0:
        raise ValueError(f"finite-difference point must lie in [0, 1], got {value!r}")
    if not math.isfinite(requested_h) or requested_h <= 0.0:
        raise ValueError(f"finite-difference epsilon must be positive and finite, got {epsilon!r}")
    h = min(requested_h, 0.5)
    if x >= h and x <= 1.0 - h:
        return float((function(x + h) - function(x - h)) / (2.0 * h))
    if x < h:
        forward_h = min(h, (1.0 - x) / 2.0)
        if forward_h <= 0.0:
            raise ValueError("cannot form a forward finite-difference stencil")
        f0 = function(x)
        f1 = function(x + forward_h)
        f2 = function(x + 2.0 * forward_h)
        return float((-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * forward_h))
    backward_h = min(h, x / 2.0)
    if backward_h <= 0.0:
        raise ValueError("cannot form a backward finite-difference stencil")
    f0 = function(x)
    f1 = function(x - backward_h)
    f2 = function(x - 2.0 * backward_h)
    return float((3.0 * f0 - 4.0 * f1 + f2) / (2.0 * backward_h))
