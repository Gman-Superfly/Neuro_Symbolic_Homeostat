"""Adaptive controller for orthogonal noise magnitude based on optimization signals.

Implements the logic:
    s_t = clamp(w1*(1-rate) + w2*backtrack + w3*(1-cos_theta) + w4*excess_L, 0, 1)
    noise_magnitude = s_t * noise_max
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OrthogonalNoiseController:
    """Adapts noise magnitude based on stability and curvature signals.
    
    Signals used:
    - Descent rate: 1 - (F_new / F_old) (if < threshold, boost noise)
    - Backtracks: if line search backtracks, boost noise (stuck/bad direction)
    - Gradient rotation: 1 - cos(g_t, g_{t-1}) (high rotation = curved valley => boost noise)
    - Contraction margin: if small, boost noise (near stability limit)
    
    Outputs a scaler in [0, 1] to multiply with base noise_magnitude.
    """
    
    base_magnitude: float = 0.0
    min_scale: float = 0.0
    max_scale: float = 1.0
    decay: float = 0.99  # Annealing factor per step
    
    # Weights for signals
    w_rate: float = 0.5
    w_backtrack: float = 0.2
    w_rotation: float = 0.3
    
    # Internal state
    _current_scale: float = 0.0
    _last_grad: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    
    def step(
        self, 
        grad: np.ndarray, 
        energy_drop_ratio: float, 
        backtracks: int, 
        iter_idx: int
    ) -> float:
        """Compute noise magnitude for the *next* step based on current signals.
        
        Args:
            grad: Current gradient vector.
            energy_drop_ratio: (F_old - F_new) / F_old (or similar relative drop).
            backtracks: Number of backtracks in the last step.
            iter_idx: Current iteration index (for annealing).
            
        Returns:
            float: Noise magnitude to use.
        """
        # 1. Gradient rotation (1 - cosine similarity)
        rotation_signal = 0.0
        if self._last_grad is not None:
            norm_curr = np.linalg.norm(grad)
            norm_last = np.linalg.norm(self._last_grad)
            if norm_curr > 1e-9 and norm_last > 1e-9:
                cos_theta = np.dot(grad, self._last_grad) / (norm_curr * norm_last)
                # cos in [-1, 1] -> rotation in [0, 2]
                # We map 1 (aligned) -> 0, -1 (opposed) -> 1
                rotation_signal = 0.5 * (1.0 - cos_theta)
        
        self._last_grad = grad.copy()
        
        # 2. Rate signal: if drop is small, boost noise
        # drop ratio approx 0 => stall => signal 1.0
        # drop ratio large => progress => signal 0.0
        rate_signal = max(0.0, 1.0 - energy_drop_ratio * 10.0) # scaling factor heuristic
        
        # 3. Backtrack signal: boolean-ish
        backtrack_signal = 1.0 if backtracks > 0 else 0.0
        
        # Combine
        raw_score = (
            self.w_rate * rate_signal +
            self.w_backtrack * backtrack_signal +
            self.w_rotation * rotation_signal
        )
        
        # Clamp and anneal
        self._current_scale = max(self.min_scale, min(self.max_scale, raw_score))
        
        # Apply decay schedule to the base magnitude, modulated by current scale
        effective_mag = self.base_magnitude * (self.decay ** iter_idx) * self._current_scale
        
        return float(effective_mag)

    def reset(self) -> None:
        self._last_grad = None
        self._current_scale = 0.0


@dataclass
class MetricAwareNoiseController(OrthogonalNoiseController):
    """Metric-aware variant (same signals; pairs with M-orthogonal projection).
    
    This controller reuses the same signal mapping as OrthogonalNoiseController.
    It is intended to be used when a problem-specific metric M is available
    and noise is projected with that metric (see project_noise_metric_orthogonal).
    """


@dataclass
class PrecisionNoiseController(OrthogonalNoiseController):
    """Precision-aware controller that redistributes noise toward low-curvature directions.

    Extends OrthogonalNoiseController with a curvature-to-weight mapper:
      w_i ∝ 1 / (eps + curvature_i)
    The weights can be used to reweight an already orthogonalized noise vector
    before re-projecting to ensure orthogonality is preserved.
    """

    precision_epsilon: float = 1e-8

    def weights_for_curvatures(self, curvatures: np.ndarray, *, eps: Optional[float] = None) -> np.ndarray:
        """Return per-dimension weights inversely proportional to curvature.

        Args:
            curvatures: 1D array of non-negative curvature (stiffness) values.
            eps: Optional override for small positive epsilon to avoid division by zero.

        Returns:
            1D array of non-negative weights normalized to unit ℓ2 norm (if possible).
        """
        curv = np.asarray(curvatures, dtype=float)
        local_eps = float(self.precision_epsilon if eps is None else eps)
        inv = 1.0 / (local_eps + np.maximum(curv, 0.0))
        # If all zeros, fall back to uniform weights
        if not np.any(inv > 0.0):
            inv = np.ones_like(inv)
        # Normalize to unit ℓ2 to separate direction (weights) from magnitude
        norm = float(np.linalg.norm(inv))
        if norm > 0.0:
            inv = inv / norm
        return inv