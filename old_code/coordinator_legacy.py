"""Coordinator paths archived because the active repository had no call sites."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from core.prox_utils import prox_asym_hinge_pair, prox_quadratic_pair


def is_gate_module(module: Any) -> bool:
    return hasattr(module, "cost") and module.__class__.__name__ == "EnergyGatingModule"


class LegacyCoordinatorMethods:
    """Reference-only mixin containing methods removed from EnergyCoordinator."""

    def prox_quadratic_pair(
        self, x0: float, y0: float, weight: float, tau: float
    ) -> tuple[float, float]:
        return prox_quadratic_pair(x0, y0, weight, tau)

    def prox_asym_hinge_pair(
        self,
        x0: float,
        y0: float,
        weight: float,
        alpha: float,
        beta: float,
        tau: float,
    ) -> tuple[float, float]:
        return prox_asym_hinge_pair(x0, y0, weight, alpha, beta, tau)

    def quadratic_energy_vectorized(self, etas: List[float], weights: dict[str, float]) -> float:
        cache = self._vectorized_cache
        if cache is None or cache.quadratic_i.size == 0:
            return 0.0
        eta_array = np.asarray(etas, dtype=float)
        term_weights = np.asarray(
            [float(weights.get(key, 1.0)) for key in cache.quadratic_term_keys],
            dtype=float,
        )
        effective_weights = cache.quadratic_weights * term_weights
        difference = eta_array[cache.quadratic_i] - eta_array[cache.quadratic_j]
        return float(np.sum(effective_weights * difference * difference))

    def coordinate_backtracking(
        self,
        etas: List[float],
        index: int,
        gradient: float,
        initial_step: float,
    ) -> List[float]:
        initial_energy = self._energy_value(etas)
        step = float(initial_step)
        gradient_sq = float(gradient * gradient)
        for _ in range(self.max_backtrack + 1):
            trial = list(etas)
            trial[index] = float(max(0.0, min(1.0, trial[index] - step * gradient)))
            if self._energy_value(trial) <= initial_energy - self.armijo_c * step * gradient_sq:
                return trial
            step *= self.backtrack_factor
        return list(etas)

    def update_uncertainty_gate_scale(self) -> None:
        """Archived incomplete path that depended on removed gate-cost settings."""
        raise NotImplementedError(
            "The uncertainty-gated cost path was never wired to EnergyCoordinator."
        )

    def relax_etas_coordinate(
        self,
        etas0: List[float],
        steps: int = 200,
        active_tol: float = 1e-4,
    ) -> List[float]:
        """Archived Gauss-Southwell-style coordinate update path."""
        etas = [float(value) for value in etas0]
        self._ensure_adjacency(len(etas))
        gradients = self._grads(etas)
        for _ in range(steps):
            index = int(np.argmax(np.abs(np.asarray(gradients, dtype=float))))
            gradient = float(gradients[index])
            if abs(gradient) < active_tol:
                break
            next_eta = float(max(0.0, min(1.0, etas[index] - self.step_size * gradient)))
            if next_eta == etas[index]:
                break
            etas[index] = next_eta
            gradients = self._grads(etas)
        return etas
