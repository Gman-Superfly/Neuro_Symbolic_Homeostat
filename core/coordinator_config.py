"""Configuration groups for the energy coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = [
    "StepConfig",
    "StabilityConfig",
    "NoiseConfig",
    "ProxConfig",
    "AdmmConfig",
    "WeightConfig",
]


@dataclass(frozen=True)
class StepConfig:
    """Gradient-step and backtracking controls."""

    grad_eps: float = 1e-4
    step_size: float = 0.05
    normalize_grads: bool = False
    max_grad_norm: Optional[float] = None
    line_search: bool = False
    backtrack_factor: float = 0.5
    max_backtrack: int = 5
    armijo_c: float = 1e-6


@dataclass(frozen=True)
class StabilityConfig:
    """Stability guard, contraction margin, and Lipschitz-step controls."""

    stability_guard: bool = True
    stability_cap_fraction: float = 0.9
    log_contraction_margin: bool = False
    warn_on_margin_shrink: bool = False
    margin_warn_threshold: float = 1e-6
    expose_lipschitz_details: bool = False
    auto_step_from_lipschitz: bool = False


@dataclass(frozen=True)
class NoiseConfig:
    """Noise projection and precision-aware exploration controls."""

    noise_mode: Optional[str] = None
    enable_orthogonal_noise: bool = True
    noise_magnitude: float = 0.0
    noise_schedule_decay: float = 0.99
    auto_noise_controller: bool = False
    metric_aware_noise_controller: bool = False
    metric_matrix: Optional[np.ndarray] = None
    metric_vector_product: Optional[Callable[[np.ndarray], np.ndarray]] = None
    precision_aware_noise_controller: bool = False
    precision_epsilon: float = 1e-8


@dataclass(frozen=True)
class ProxConfig:
    """Operator-splitting and proximal relaxation controls."""

    operator_splitting: bool = False
    prox_tau: float = 0.05
    prox_steps: int = 50
    prox_block_mode: Optional[str] = None


@dataclass(frozen=True)
class AdmmConfig:
    """ADMM-style splitting controls."""

    use_admm: bool = False
    admm_rho: float = 1.0
    admm_steps: int = 50
    admm_step_size: float = 0.05
    admm_gate_prox: bool = True
    admm_gate_damping: float = 0.5


@dataclass(frozen=True)
class WeightConfig:
    """Term-weight calibration controls."""

    term_weight_floor: float = 0.0
    term_weight_ceiling: Optional[float] = None
    auto_balance_term_weights: bool = False
    term_norm_target: float = 1.0
    max_term_norm_ratio: float = 10.0
