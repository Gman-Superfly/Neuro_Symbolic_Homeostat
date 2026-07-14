"""Information metrics used by optional relaxation telemetry."""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


class InformationMetrics:
    """Calculator for information structure metrics."""

    @staticmethod
    def compute_alignment(
        current: Sequence[float],
        reference: Sequence[float]
    ) -> float:
        """Compute cosine alignment between current state and reference concept.
        
        a = (current . reference) / (|current| * |reference|)
        """
        # Input validation
        assert current is not None and reference is not None, "inputs required"
        curr = np.asarray(current, dtype=float)
        ref = np.asarray(reference, dtype=float)
        assert curr.shape == ref.shape, f"shape mismatch: {curr.shape} vs {ref.shape}"
        assert curr.ndim == 1 or 1 in curr.shape, "expected 1D vectors or broadcastable 1D"
        if curr.ndim != 1:
            curr = curr.ravel()
            ref = ref.ravel()
        
        norm_c = np.linalg.norm(curr)
        norm_r = np.linalg.norm(ref)
        
        if norm_c < 1e-9 or norm_r < 1e-9:
            return 0.0
            
        cos_val = float(np.dot(curr, ref) / (norm_c * norm_r))
        # Numeric safety: clip to valid cosine range
        cos_val = max(-1.0, min(1.0, cos_val))
        # Output validation
        assert -1.0 - 1e-9 <= cos_val <= 1.0 + 1e-9, "alignment out of range"
        return cos_val

    @staticmethod
    def compute_drift(
        current: Sequence[float],
        reference: Sequence[float]
    ) -> float:
        """Compute Euclidean drift from reference trajectory.
        
        Δ = |current - reference|
        """
        assert current is not None and reference is not None, "inputs required"
        curr = np.asarray(current, dtype=float)
        ref = np.asarray(reference, dtype=float)
        assert curr.shape == ref.shape, f"shape mismatch: {curr.shape} vs {ref.shape}"
        drift = float(np.linalg.norm(curr - ref))
        assert drift >= 0.0, "negative drift impossible"
        return drift

    @staticmethod
    def compute_redundancy(
        gain: float,
        uncertainty: float,
        epsilon: float = 1e-8
    ) -> float:
        """Compute redundancy score.
        
        ρ = gain / (uncertainty + ε)
        High ρ implies the signal provides high gain relative to uncertainty.
        """
        assert isinstance(gain, (int, float)), "gain must be numeric"
        assert isinstance(uncertainty, (int, float)), "uncertainty must be numeric"
        assert isinstance(epsilon, (int, float)) and epsilon > 0.0, "epsilon must be > 0"
        assert uncertainty >= 0.0, "uncertainty must be non-negative"
        val = float(gain / (uncertainty + epsilon))
        assert math.isfinite(val), "redundancy not finite"
        return val

    @staticmethod
    def compute_constraint_violation_rate(
        violations: int,
        total_constraints: int
    ) -> float:
        """Compute constraint violation rate.

        rate = violations / total_constraints
        This corresponds to what our theory documents previously called
        'hallucination rate'. The new name is used for clarity.
        """
        assert isinstance(violations, (int, float)), "violations must be numeric"
        assert isinstance(total_constraints, (int, float)), "total_constraints must be numeric"
        if total_constraints <= 0:
            return 0.0
        v = max(0.0, float(violations))
        t = float(total_constraints)
        rate = v / t
        # Clamp to [0,1] to handle accidental overcounts gracefully
        rate = max(0.0, min(1.0, rate))
        return float(rate)

    # Backward-compatible alias (deprecated name)
    compute_hallucination_rate = compute_constraint_violation_rate

