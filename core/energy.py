"""Landau-style free energy utilities and total energy helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

import numpy as np

from .interfaces import EnergyModule, EnergyCoupling, OrderParameter

__all__ = [
    "total_energy",
    "project_noise_orthogonal",
    "project_noise_metric_orthogonal",
]





def total_energy(
    etas: List[OrderParameter],
    modules: List[EnergyModule],
    couplings: List[tuple[int, int, EnergyCoupling]],
    constraints: Mapping[str, Any],
) -> float:
    """Total energy F_total = Σ F_local + Σ F_couple."""
    assert len(etas) == len(modules), "Mismatch between etas and modules"
    total = 0.0
    # Optional term weights: {'local:ClassName': w, 'coup:ClassName': w}
    weights: Dict[str, float] = {}
    tw = constraints.get("term_weights", None)
    if isinstance(tw, dict):
        # best-effort copy of float-like values
        for k, v in tw.items():
            try:
                weights[str(k)] = float(v)  # type: ignore[arg-type]
            except Exception:
                continue
    for m, eta in zip(modules, etas):
        f = float(m.local_energy(eta, constraints))
        key = f"local:{m.__class__.__name__}"
        w = float(weights.get(key, 1.0))
        total += (w * f)
    for i, j, coup in couplings:
        assert 0 <= i < len(etas) and 0 <= j < len(etas), "Invalid coupling indices"
        fc = float(coup.coupling_energy(etas[i], etas[j], constraints))
        key = f"coup:{coup.__class__.__name__}"
        w = float(weights.get(key, 1.0))
        total += (w * fc)
    return float(total)


def project_noise_orthogonal(
    noise: np.ndarray,
    grad: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """Project noise vector onto the subspace orthogonal to the gradient.
    
    z_orth = z - (z · g) * g / ||g||²
    
    This ensures exploration happens along the level sets of the energy function
    (iso-energy contours), avoiding ascent/descent directions.
    """
    # Compute gradient norm squared
    grad_norm_sq = np.sum(grad * grad)
    
    if grad_norm_sq < eps:
        # Gradient is zero (at min/max/saddle) => all directions are valid
        return noise
        
    # Compute projection scalar: (z · g) / ||g||²
    projection_scalar = np.sum(noise * grad) / grad_norm_sq
    
    # Subtract component parallel to gradient
    noise_orth = noise - projection_scalar * grad
    
    return noise_orth


def project_noise_metric_orthogonal(
    noise: np.ndarray,
    grad: np.ndarray,
    *,
    M: np.ndarray | None = None,
    metric_solve: Callable[[np.ndarray], np.ndarray] | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return the M-orthogonal projection onto the energy tangent plane.

    ``grad`` is the ordinary gradient covector, so first-order tangency requires
    ``grad.T @ delta == 0``. The corresponding metric gradient is
    ``metric_grad = solve(M, grad)``. Projecting along that vector gives

        delta = z - ((grad.T @ z) / (grad.T @ metric_grad)) * metric_grad.

    The result is M-orthogonal to the metric gradient and therefore has zero
    first-order directional derivative. ``metric_solve`` provides a matrix-free
    implementation of ``solve(M, vector)``. With no metric input, this function
    falls back to the Euclidean projection.
    """
    g = np.asarray(grad, dtype=float)
    z = np.asarray(noise, dtype=float)
    assert g.shape == z.shape, "noise and grad must have the same shape"
    assert g.ndim == 1, "metric projection expects one-dimensional vectors"
    assert M is None or metric_solve is None, "provide M or metric_solve, not both"
    if M is None and metric_solve is None:
        return project_noise_orthogonal(z, g, eps=eps)

    grad_norm_sq = float(np.dot(g, g))
    if grad_norm_sq < eps:
        return z

    if metric_solve is not None:
        metric_grad = np.asarray(metric_solve(g), dtype=float)
    else:
        metric = np.asarray(M, dtype=float)
        assert metric.shape == (g.size, g.size), "M shape must match the gradient dimension"
        metric_grad = np.linalg.solve(metric, g)

    assert metric_grad.shape == g.shape, "metric_solve must preserve vector shape"
    denominator = float(np.dot(g, metric_grad))
    assert np.isfinite(denominator) and denominator > 0.0, "metric solve must define a positive SPD geometry"
    alpha = float(np.dot(g, z)) / denominator
    return z - alpha * metric_grad



