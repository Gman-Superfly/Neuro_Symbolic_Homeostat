"""Grouped public configuration for :class:`EnergyCoordinator`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .interfaces import WeightAdapter
from .solver_config import SolverConfig


@dataclass(frozen=True)
class GradientConfig:
    grad_eps: float = 1e-4
    step_size: float = 0.05
    use_analytic: bool = True
    normalize_grads: bool = False
    max_grad_norm: Optional[float] = None
    line_search: bool = False
    backtrack_factor: float = 0.5
    max_backtrack: int = 5
    armijo_c: float = 1e-6
    use_stiffness_updates: bool = False
    stiffness_epsilon: float = 1e-8


@dataclass(frozen=True)
class ExecutionConfig:
    use_vectorized_quadratic: bool = True
    use_vectorized_hinges: bool = True
    use_vectorized_gate_benefits: bool = True
    neighbor_gradients_only: bool = True
    enforce_invariants: bool = True


@dataclass(frozen=True)
class GuardConfig:
    stability_guard: bool = True
    stability_cap_fraction: float = 0.9
    log_contraction_margin: bool = False
    warn_on_margin_shrink: bool = False
    margin_warn_threshold: float = 1e-6
    expose_lipschitz_details: bool = False
    auto_step_from_lipschitz: bool = False
    use_free_energy_guard: bool = False
    free_energy_temperature: float = 1.0
    free_energy_epsilon: float = 1e-6
    enable_early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_delta_threshold: float = 1e-6
    assert_monotonic_energy: bool = True
    monotonic_energy_tol: float = 1e-10
    continue_after_rejection: bool = False


@dataclass(frozen=True)
class NoiseConfig:
    mode: Optional[str] = None
    enable_orthogonal: bool = True
    magnitude: float = 0.0
    schedule_decay: float = 0.99
    auto_controller: bool = False
    metric_aware: bool = False
    metric_matrix: Optional[np.ndarray] = None
    metric_solve: Optional[Callable[[np.ndarray], np.ndarray]] = None
    precision_aware: bool = False
    use_precision_preconditioning: bool = True
    precision_epsilon: float = 1e-8


@dataclass(frozen=True)
class WeightConfig:
    term_weight_floor: float = 0.0
    term_weight_ceiling: Optional[float] = None
    auto_balance_term_weights: bool = False
    term_norm_target: float = 1.0
    max_term_norm_ratio: float = 10.0
    adapter: Optional[WeightAdapter] = None


@dataclass(frozen=True)
class CoordinatorConfig:
    """Preferred grouped configuration surface for new callers."""

    solver: SolverConfig = field(default_factory=SolverConfig.gradient_solver)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)

    @classmethod
    def deterministic(cls, *, solver: Optional[SolverConfig] = None) -> "CoordinatorConfig":
        return cls(
            solver=solver or SolverConfig.gradient_solver(),
            noise=NoiseConfig(mode="none", magnitude=0.0, auto_controller=False),
        )

    def to_kwargs(self) -> dict[str, object]:
        """Translate grouped settings to the coordinator's internal fields."""
        return {
            "solver": self.solver,
            "grad_eps": self.gradient.grad_eps,
            "step_size": self.gradient.step_size,
            "use_analytic": self.gradient.use_analytic,
            "normalize_grads": self.gradient.normalize_grads,
            "max_grad_norm": self.gradient.max_grad_norm,
            "line_search": self.gradient.line_search,
            "backtrack_factor": self.gradient.backtrack_factor,
            "max_backtrack": self.gradient.max_backtrack,
            "armijo_c": self.gradient.armijo_c,
            "use_stiffness_updates": self.gradient.use_stiffness_updates,
            "stiffness_epsilon": self.gradient.stiffness_epsilon,
            "use_vectorized_quadratic": self.execution.use_vectorized_quadratic,
            "use_vectorized_hinges": self.execution.use_vectorized_hinges,
            "use_vectorized_gate_benefits": self.execution.use_vectorized_gate_benefits,
            "neighbor_gradients_only": self.execution.neighbor_gradients_only,
            "enforce_invariants": self.execution.enforce_invariants,
            "stability_guard": self.guards.stability_guard,
            "stability_cap_fraction": self.guards.stability_cap_fraction,
            "log_contraction_margin": self.guards.log_contraction_margin,
            "warn_on_margin_shrink": self.guards.warn_on_margin_shrink,
            "margin_warn_threshold": self.guards.margin_warn_threshold,
            "expose_lipschitz_details": self.guards.expose_lipschitz_details,
            "auto_step_from_lipschitz": self.guards.auto_step_from_lipschitz,
            "use_free_energy_guard": self.guards.use_free_energy_guard,
            "free_energy_temperature": self.guards.free_energy_temperature,
            "free_energy_epsilon": self.guards.free_energy_epsilon,
            "enable_early_stop": self.guards.enable_early_stop,
            "early_stop_patience": self.guards.early_stop_patience,
            "early_stop_delta_threshold": self.guards.early_stop_delta_threshold,
            "assert_monotonic_energy": self.guards.assert_monotonic_energy,
            "monotonic_energy_tol": self.guards.monotonic_energy_tol,
            "continue_after_rejection": self.guards.continue_after_rejection,
            "noise_mode": self.noise.mode,
            "enable_orthogonal_noise": self.noise.enable_orthogonal,
            "noise_magnitude": self.noise.magnitude,
            "noise_schedule_decay": self.noise.schedule_decay,
            "auto_noise_controller": self.noise.auto_controller,
            "metric_aware_noise_controller": self.noise.metric_aware,
            "metric_matrix": self.noise.metric_matrix,
            "metric_solve": self.noise.metric_solve,
            "precision_aware_noise_controller": self.noise.precision_aware,
            "use_precision_preconditioning": self.noise.use_precision_preconditioning,
            "precision_epsilon": self.noise.precision_epsilon,
            "term_weight_floor": self.weights.term_weight_floor,
            "term_weight_ceiling": self.weights.term_weight_ceiling,
            "auto_balance_term_weights": self.weights.auto_balance_term_weights,
            "term_norm_target": self.weights.term_norm_target,
            "max_term_norm_ratio": self.weights.max_term_norm_ratio,
            "weight_adapter": self.weights.adapter,
        }
