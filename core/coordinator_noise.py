"""Noise-mode helpers for energy relaxation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .energy import project_noise_metric_orthogonal, project_noise_orthogonal
from .noise_controller import PrecisionNoiseController

__all__ = ["apply_box_feasible_noise", "build_noise_vector", "resolved_noise_mode"]


def resolved_noise_mode(coordinator: Any) -> str:
    """Resolve legacy noise flags into an explicit noise mode."""
    if coordinator.noise_mode is not None:
        mode = str(coordinator.noise_mode).lower()
        if mode == "none":
            return "none"
        return mode
    if coordinator.precision_aware_noise_controller and coordinator.metric_aware_noise_controller:
        return "metric_precision_orthogonal"
    if coordinator.precision_aware_noise_controller:
        return "precision_orthogonal"
    if coordinator.metric_aware_noise_controller:
        return "metric_orthogonal"
    if coordinator.enable_orthogonal_noise:
        return "orthogonal"
    return "isotropic"


def build_noise_vector(
    coordinator: Any,
    raw_noise: np.ndarray,
    grad_vector: np.ndarray,
    current_noise_mag: float,
) -> np.ndarray:
    """Build a noise vector for the coordinator's configured noise mode."""
    assert raw_noise.shape == grad_vector.shape, "raw_noise and grad_vector shape mismatch"
    assert current_noise_mag >= 0.0, "current_noise_mag must be non-negative"

    mode = resolved_noise_mode(coordinator)
    if mode == "none" or current_noise_mag <= 1e-9:
        return np.zeros_like(grad_vector)

    if mode == "isotropic":
        noise_vector = np.asarray(raw_noise, dtype=float)
    elif mode in {"metric_orthogonal", "metric_precision_orthogonal"}:
        noise_vector = project_noise_metric_orthogonal(
            raw_noise,
            grad_vector,
            M=coordinator.metric_matrix,
            metric_solve=coordinator.metric_solve,
        )
    elif mode in {"orthogonal", "precision_orthogonal"}:
        noise_vector = project_noise_orthogonal(raw_noise, grad_vector)
    else:
        raise ValueError(f"Unknown noise_mode: {coordinator.noise_mode!r}")

    if mode in {"precision_orthogonal", "metric_precision_orthogonal"}:
        curv_diag = np.asarray(coordinator.get_precision_diagonal(), dtype=float)
        if isinstance(coordinator._noise_controller, PrecisionNoiseController) and hasattr(  # noqa: SLF001
            coordinator._noise_controller,  # noqa: SLF001
            "weights_for_curvatures",
        ):
            weights = coordinator._noise_controller.weights_for_curvatures(  # noqa: SLF001
                curv_diag,
                eps=coordinator.precision_epsilon,
            )
        else:
            inv = 1.0 / (float(coordinator.precision_epsilon) + np.maximum(curv_diag, 0.0))
            inv_norm = float(np.linalg.norm(inv))
            weights = inv / inv_norm if inv_norm > 0.0 else inv
        weighted_noise = weights * noise_vector
        if mode == "metric_precision_orthogonal":
            noise_vector = project_noise_metric_orthogonal(
                weighted_noise,
                grad_vector,
                M=coordinator.metric_matrix,
                metric_solve=coordinator.metric_solve,
            )
        else:
            noise_vector = project_noise_orthogonal(weighted_noise, grad_vector)

    noise_norm = np.linalg.norm(noise_vector)
    if noise_norm <= 1e-9:
        return np.zeros_like(grad_vector)
    return noise_vector * (current_noise_mag / noise_norm)


def apply_box_feasible_noise(
    state: np.ndarray | list[float],
    noise: np.ndarray,
) -> np.ndarray:
    """Apply the largest uniform noise scaling that remains in the unit box.

    Uniform scaling preserves any first-order orthogonality established by the
    noise builder. Per-coordinate clipping would generally destroy it.
    """
    values = np.asarray(state, dtype=float)
    direction = np.asarray(noise, dtype=float)
    if values.shape != direction.shape:
        raise ValueError("state and noise must have matching shapes")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(direction)):
        raise ValueError("state and noise must contain finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("state must lie in the closed unit box")

    scale = 1.0
    for value, component in zip(values, direction):
        if component > 0.0:
            scale = min(scale, float((1.0 - value) / component))
        elif component < 0.0:
            scale = min(scale, float(-value / component))
    scale = max(0.0, min(1.0, scale))
    proposal = values + scale * direction
    return np.clip(proposal, 0.0, 1.0)
