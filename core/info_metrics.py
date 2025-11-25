"""Information Structure Metrics for Energy Landscapes.

Implements validation signals from 'Information_Structures':
- ρ (Redundancy): Semantic alignment/mutual information proxy
- a (Alignment): Hidden-state concept matching
- h (Hallucination): Constraint violation or unsupported output rate
"""

from __future__ import annotations

import math
from typing import List, Optional, Protocol, Any, Dict, Sequence
import numpy as np


class SupportsRedundancy(Protocol):
    """Protocol for modules that can measure redundancy with a source."""
    def compute_redundancy(self, source_eta: float, context: Any) -> float:
        ...


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
        curr = np.asarray(current, dtype=float)
        ref = np.asarray(reference, dtype=float)
        
        norm_c = np.linalg.norm(curr)
        norm_r = np.linalg.norm(ref)
        
        if norm_c < 1e-9 or norm_r < 1e-9:
            return 0.0
            
        return float(np.dot(curr, ref) / (norm_c * norm_r))

    @staticmethod
    def compute_drift(
        current: Sequence[float],
        reference: Sequence[float]
    ) -> float:
        """Compute Euclidean drift from reference trajectory.
        
        Δ = |current - reference|
        """
        curr = np.asarray(current, dtype=float)
        ref = np.asarray(reference, dtype=float)
        return float(np.linalg.norm(curr - ref))

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
        return float(gain / (uncertainty + epsilon))

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
        if total_constraints <= 0:
            return 0.0
        return float(violations / total_constraints)

    # Backward-compatible alias (deprecated name)
    compute_hallucination_rate = compute_constraint_violation_rate

