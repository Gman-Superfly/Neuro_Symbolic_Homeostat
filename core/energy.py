"""Landau-style free energy utilities and total energy helpers."""

from __future__ import annotations

from typing import List, Mapping, Any, Tuple, Dict
import math

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
    Mv: callable | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Project noise onto the subspace orthogonal to the gradient under metric M.
    
    When M is None and Mv is None, falls back to Euclidean projection.
    
    Uses:
        z_perp = z - ((z^T M g) / (g^T M g)) g
    where M g is computed via Mv(g) if provided, else M @ g.
    """
    g = np.asarray(grad, dtype=float)
    z = np.asarray(noise, dtype=float)
    if M is None and Mv is None:
        return project_noise_orthogonal(z, g, eps=eps)
    if Mv is not None:
        Mg = np.asarray(Mv(g), dtype=float)
    else:
        Mg = np.asarray(M @ g, dtype=float)  # type: ignore[operator]
    gT_M_g = float(np.sum(g * Mg))
    if abs(gT_M_g) < eps:
        return z
    zT_M_g = float(np.sum(z * Mg))
    alpha = zT_M_g / gT_M_g
    return z - alpha * g



