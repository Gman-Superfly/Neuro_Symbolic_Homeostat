"""Coordinate energy evaluation and guarded relaxation across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import math
import warnings
from types import MappingProxyType
import numpy as np

from .interfaces import (
    EnergyModule,
    EnergyCoupling,
    OrderParameter,
    SupportsLocalEnergyGrad,
    SupportsCouplingGrads,
    SupportsCouplingCurvature,
    WeightAdapter,
)
from .couplings import (
    QuadraticCoupling,
    DirectedHingeCoupling,
    AsymmetricHingeCoupling,
    GateBenefitCoupling,
    DampedGateBenefitCoupling,
)
from .energy import total_energy
from .finite_difference import box_derivative
from .diagnostics import CoordinatorSnapshot
from .noise_controller import OrthogonalNoiseController, PrecisionNoiseController
from .coordinator_noise import apply_box_feasible_noise, build_noise_vector, resolved_noise_mode
from .coordinator_precision import get_precision_diagonal, get_update_preconditioner, update_precision_cache
from .coordinator_stability import (
    estimate_lipschitz_bound,
    estimate_preconditioned_lipschitz_bound,
    validate_curvature_bound_triplet,
)
from .config import CoordinatorConfig
from .solver_config import ADMMSolverConfig, ProximalSolverConfig, SolverConfig, SolverMode
from .solvers import solve_admm, solve_proximal


@dataclass
class _VectorizedCouplingCache:
    quadratic_i: np.ndarray
    quadratic_j: np.ndarray
    quadratic_weights: np.ndarray
    quadratic_term_keys: Tuple[str, ...]

    directed_i: np.ndarray
    directed_j: np.ndarray
    directed_weights: np.ndarray
    directed_term_keys: Tuple[str, ...]

    asymmetric_i: np.ndarray
    asymmetric_j: np.ndarray
    asymmetric_weights: np.ndarray
    asymmetric_term_keys: Tuple[str, ...]
    asymmetric_alpha: np.ndarray
    asymmetric_beta: np.ndarray

    gate_idx: np.ndarray
    gate_weights: np.ndarray
    gate_term_keys: Tuple[str, ...]
    gate_delta_keys: Tuple[str, ...]

    damped_idx: np.ndarray
    damped_weights: np.ndarray
    damped_term_keys: Tuple[str, ...]
    damped_delta_keys: Tuple[str, ...]
    damped_damping: np.ndarray
    damped_eta_power: np.ndarray
    damped_positive_scale: np.ndarray
    damped_negative_scale: np.ndarray

EtaUpdateCallback = Callable[[List[OrderParameter]], None]
EnergyUpdateCallback = Callable[[float], None]


@dataclass
class EnergyCoordinator:
    """Compose local and coupling energies under guarded update rules."""

    modules: List[EnergyModule]
    couplings: List[tuple[int, int, EnergyCoupling]]
    constraints: Mapping[str, Any]
    grad_eps: float = 1e-4
    step_size: float = 0.05
    # Gradient/optimization controls
    use_analytic: bool = True
    normalize_grads: bool = False
    max_grad_norm: Optional[float] = None
    line_search: bool = False
    backtrack_factor: float = 0.5
    max_backtrack: int = 5
    armijo_c: float = 1e-6
    use_vectorized_quadratic: bool = True
    use_vectorized_hinges: bool = True
    use_vectorized_gate_benefits: bool = True
    # Deprecated compatibility flag. Full-objective finite differences must
    # evaluate every coordinate because isolated nodes still own local energy.
    neighbor_gradients_only: bool = True
    enforce_invariants: bool = True
    solver: SolverConfig = field(default_factory=SolverConfig)
    # Stiffness-based weighted-Jacobi updates.
    # The update is Δη_i = -α (∂F/∂η_i) / P_i, where P is the same
    # positive diagonal used to normalize the stability bound.
    use_stiffness_updates: bool = False
    stiffness_epsilon: float = 1e-8
    # Free-energy guard: F = U - T*S acceptance (Phase 2)
    use_free_energy_guard: bool = False
    free_energy_temperature: float = 1.0
    free_energy_epsilon: float = 1e-6
    # Early-stop with patience (Phase 2)
    enable_early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_delta_threshold: float = 1e-6
    # (homotopy/sensitivity features removed)
    # Term-weight calibration
    term_weight_floor: float = 0.0
    term_weight_ceiling: Optional[float] = None
    auto_balance_term_weights: bool = False
    term_norm_target: float = 1.0
    max_term_norm_ratio: float = 10.0
    # Optional term-weight adapter
    weight_adapter: Optional[WeightAdapter] = None
    # Curvature-based stability guard (optional)
    stability_guard: bool = True
    stability_cap_fraction: float = 0.9  # cap step to this fraction of 2/L estimate
    log_step_cap_slack: bool = False
    # Deprecated name retained for configuration compatibility.
    log_contraction_margin: bool = False
    # (stability coupling auto-cap removed)
    warn_on_margin_shrink: bool = False  # emit Python warnings when margin drops below threshold
    margin_warn_threshold: float = 1e-6  # threshold for margin warnings
    # Lipschitz/allocator details (instrumentation for adapters/telemetry)
    expose_lipschitz_details: bool = False
    # Noise / Exploration controls
    noise_mode: Optional[str] = None  # None maps legacy flags to: isotropic, orthogonal, or precision_orthogonal.
    enable_orthogonal_noise: bool = True  # Inject noise orthogonal to gradient (structure-preserving)
    # Note: default magnitude is 0.0 to preserve determinism unless explicitly enabled in experiments.
    noise_magnitude: float = 0.0
    noise_schedule_decay: float = 0.99  # Simple exponential decay for noise magnitude
    auto_noise_controller: bool = False  # Adapt noise magnitude using orthogonal-noise controller
    # Metric-aware projection (optional)
    metric_aware_noise_controller: bool = False
    metric_matrix: Optional[np.ndarray] = None
    metric_solve: Optional[Callable[[np.ndarray], np.ndarray]] = None
    # Precision-aware noise & steps
    precision_aware_noise_controller: bool = False
    use_precision_preconditioning: bool = True
    precision_epsilon: float = 1e-8
    # Auto step selection from Lipschitz bound (optional; requires stability_guard)
    auto_step_from_lipschitz: bool = False
    # (uncertainty-gated gate costs removed)
    # Monotone acceptance assertion for deterministic accepted steps.
    assert_monotonic_energy: bool = True
    monotonic_energy_tol: float = 1e-10  # Tolerance for numeric jitter
    continue_after_rejection: bool = False  # Restore and attempt the remaining schedule instead of stopping.
    # (escape/confidence logging removed)

    on_eta_updated: List[EtaUpdateCallback] = field(default_factory=list)
    on_energy_updated: List[EnergyUpdateCallback] = field(default_factory=list)

    _adjacency: Optional[List[List[Tuple[int, EnergyCoupling]]]] = field(default=None, init=False, repr=False)
    _term_weights: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _grad_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _trial_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _local_energy_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _local_grad_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _total_backtracks: int = field(default=0, init=False, repr=False)
    _last_step_backtracks: int = field(default=0, init=False, repr=False)
    _last_acceptance_reason: Optional[str] = field(default=None, init=False, repr=False)
    _last_step_cap_slack: Optional[float] = field(default=None, init=False, repr=False)
    _last_contraction_margin: Optional[float] = field(default=None, init=False, repr=False)
    # (homotopy/coupling auto-cap internals removed)
    _last_lipschitz_details: Optional[dict] = field(default=None, init=False, repr=False)
    _noise_controller: Optional[OrthogonalNoiseController] = field(default=None, init=False, repr=False)
    _last_energy_drop_ratio: float = field(default=1.0, init=False, repr=False)
    _accepted_energy_history: List[float] = field(default_factory=list, init=False, repr=False)
    _guard_energy_transitions: List[Tuple[int, float, float]] = field(default_factory=list, init=False, repr=False)
    _attempt_energy_history: List[float] = field(default_factory=list, init=False, repr=False)
    _acceptance_reason_history: List[str] = field(default_factory=list, init=False, repr=False)
    _objective_version: int = field(default=0, init=False, repr=False)
    _step_cap_slack_history: List[float] = field(default_factory=list, init=False, repr=False)
    _contraction_margin_history: List[float] = field(default_factory=list, init=False, repr=False)
    _rejected_steps: int = field(default=0, init=False, repr=False)
    _early_stop_stable_count: int = field(default=0, init=False, repr=False)
    _vectorized_cache: Optional[_VectorizedCouplingCache] = field(default=None, init=False, repr=False)
    _precision_cache: Optional[Dict[int, float]] = field(default=None, init=False, repr=False)
    _last_solver_metrics: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        modules: List[EnergyModule],
        couplings: List[tuple[int, int, EnergyCoupling]],
        constraints: Mapping[str, Any],
        config: CoordinatorConfig,
    ) -> "EnergyCoordinator":
        """Construct a coordinator from the grouped public configuration."""
        return cls(modules=modules, couplings=couplings, constraints=constraints, **config.to_kwargs())

    def __post_init__(self) -> None:
        self._validate_configuration()
        self._ensure_adjacency(len(self.modules))
        self._build_vectorized_cache()
        noise_mode = self._resolved_noise_mode()
        if self.auto_noise_controller and noise_mode in {
            "orthogonal",
            "precision_orthogonal",
            "metric_orthogonal",
            "metric_precision_orthogonal",
        }:
            if self.precision_aware_noise_controller:
                self._noise_controller = PrecisionNoiseController(
                    base_magnitude=float(self.noise_magnitude),
                    decay=float(self.noise_schedule_decay),
                    precision_epsilon=float(self.precision_epsilon),
                )
            else:
                self._noise_controller = OrthogonalNoiseController(
                    base_magnitude=float(self.noise_magnitude),
                    decay=float(self.noise_schedule_decay),
                )
        else:
            self._noise_controller = None
        self._last_energy_drop_ratio = 1.0

    def _resolved_noise_mode(self) -> str:
        """Resolve legacy noise flags into an explicit noise mode."""
        return resolved_noise_mode(self)

    def _build_noise_vector(
        self,
        raw_noise: np.ndarray,
        grad_vector: np.ndarray,
        current_noise_mag: float,
    ) -> np.ndarray:
        """Compatibility wrapper for the public noise-vector method."""
        return self.build_noise_vector(raw_noise, grad_vector, magnitude=current_noise_mag)

    def build_noise_vector(
        self,
        raw_noise: np.ndarray,
        gradient: np.ndarray,
        *,
        magnitude: Optional[float] = None,
    ) -> np.ndarray:
        """Transform a raw draw according to the configured noise geometry."""
        current_magnitude = self.noise_magnitude if magnitude is None else float(magnitude)
        return build_noise_vector(self, raw_noise, gradient, current_magnitude)

    def compute_etas(self, inputs: List[Any]) -> List[OrderParameter]:
        assert len(inputs) == len(self.modules), "inputs/modules length mismatch"
        etas: List[OrderParameter] = []
        for module, x in zip(self.modules, inputs):
            eta = float(module.compute_eta(x))
            etas.append(eta)
        self._emit_eta(etas)
        return etas

    def energy(self, etas: List[OrderParameter]) -> float:
        F = self._energy_value(etas)
        self._emit_energy(F)
        return F

    def inspect_state(self, etas: List[OrderParameter]) -> CoordinatorSnapshot:
        """Return public energy, gradient, curvature, and weight diagnostics."""
        values = [float(value) for value in etas]
        gradient = tuple(float(value) for value in self._grads(values))
        self._update_precision_cache(values)
        precision_diagonal = tuple(float(value) for value in self.get_precision_diagonal())
        raw_lipschitz_bound = float(self._estimate_lipschitz_bound(values))
        if self.use_stiffness_updates or self.use_precision_preconditioning:
            preconditioner = np.asarray(get_update_preconditioner(self), dtype=float)
            update_lipschitz_bound = float(
                self._estimate_preconditioned_lipschitz_bound(values, preconditioner)
            )
            preconditioner_diagonal = tuple(float(value) for value in preconditioner)
        else:
            update_lipschitz_bound = raw_lipschitz_bound
            preconditioner_diagonal = tuple(1.0 for _ in values)
        return CoordinatorSnapshot(
            etas=tuple(values),
            energy=float(self._energy_value(values)),
            gradient=gradient,
            precision_diagonal=precision_diagonal,
            lipschitz_bound=raw_lipschitz_bound,
            term_weights=dict(self._combined_term_weights()),
            term_gradient_norms=dict(self._term_grad_norms(values)),
            objective_version=int(self._objective_version),
            update_lipschitz_bound=update_lipschitz_bound,
            preconditioner_diagonal=preconditioner_diagonal,
        )

    def _energy_value(self, etas: List[OrderParameter]) -> float:
        # Merge term weights (constraints.term_weights overridden by adapter-maintained _term_weights)
        merged_constraints: dict[str, Any] = dict(self.constraints)
        calibrated_weights = self._combined_term_weights()
        if calibrated_weights:
            merged_constraints["term_weights"] = calibrated_weights
        return total_energy(etas, self.modules, self.couplings, merged_constraints)

    def relax_etas(self, etas0: List[OrderParameter], steps: int = 50) -> List[OrderParameter]:
        """Run one relaxation against a fixed snapshot of external constraints.

        Top-level constraint values are sampled at the run boundary. In
        particular, gate-benefit deltas cannot change between an energy
        evaluation and its gradient or acceptance check. Adapter-owned term
        weights remain an explicit, versioned source of between-step changes.
        """
        return self._run_with_constraint_snapshot(
            lambda: self._relax_etas_with_snapshot(etas0, steps)
        )

    def _run_with_constraint_snapshot(
        self,
        operation: Callable[[], List[OrderParameter]],
    ) -> List[OrderParameter]:
        """Run an optimization entry point with fixed top-level constraints."""
        original_constraints = self.constraints
        snapshot: dict[str, Any] = dict(original_constraints)
        term_weights = snapshot.get("term_weights")
        if isinstance(term_weights, Mapping):
            snapshot["term_weights"] = MappingProxyType(dict(term_weights))
        for _, _, coupling in self.couplings:
            if isinstance(coupling, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                delta = float(snapshot.get(coupling.delta_key, 0.0))
                if not math.isfinite(delta):
                    raise ValueError(
                        f"gate-benefit constraint {coupling.delta_key!r} must be finite"
                    )
                snapshot[coupling.delta_key] = delta
        self.constraints = MappingProxyType(snapshot)
        try:
            return operation()
        finally:
            self.constraints = original_constraints

    def _relax_etas_with_snapshot(
        self,
        etas0: List[OrderParameter],
        steps: int,
    ) -> List[OrderParameter]:
        """Implement ``relax_etas`` after the run constraint snapshot is installed."""
        if self.solver.mode == SolverMode.PROXIMAL:
            return solve_proximal(self, etas0, self.solver.proximal)
        if self.solver.mode == SolverMode.ADMM:
            return solve_admm(self, etas0, self.solver.admm)
        etas = [float(e) for e in etas0]
        # Initialize noise controller (if enabled)
        noise_mode = self._resolved_noise_mode()
        controller = self._noise_controller if (
            self.auto_noise_controller and noise_mode in {
                "orthogonal",
                "precision_orthogonal",
                "metric_orthogonal",
                "metric_precision_orthogonal",
            }
        ) else None
        # Controller feedback is defined within one relaxation run. The first
        # proposal has no completed predecessor; later proposals consume the
        # backtrack count produced by the immediately preceding proposal.
        controller_backtracks = 0
        if controller is not None:
            controller.base_magnitude = float(self.noise_magnitude)
            controller.decay = float(self.noise_schedule_decay)
            controller.reset()
        self._last_energy_drop_ratio = 1.0
        energy_value = self._energy_value(etas)
        prev_energy_value: Optional[float] = energy_value
        self._accepted_energy_history = []
        self._guard_energy_transitions = []
        self._attempt_energy_history = []
        self._acceptance_reason_history = []
        self._objective_version = 0
        self._last_step_cap_slack = None
        self._last_contraction_margin = None
        self._step_cap_slack_history = []
        self._contraction_margin_history = []
        self._rejected_steps = 0
        for iter_idx in range(steps):
            self._last_acceptance_reason = None
            self._last_step_backtracks = 0
            line_search_failed = False
            # Weight adaptation changes the objective between iterations. Build
            # the guard baseline under the same weights used for this proposal.
            prev_energy_value = self._energy_value(etas)
            etas_prev = list(etas)
            L_est = None
            
            # Phase 2: Update precision cache before gradient step
            self._update_precision_cache(etas)

            uses_preconditioning = self.use_stiffness_updates or self.use_precision_preconditioning
            preconditioner = (
                np.asarray(get_update_preconditioner(self), dtype=float)
                if uses_preconditioning
                else None
            )

            objective_grads = self._grads(etas)
            grads = list(objective_grads)
            # optional normalization/clipping
            if self.normalize_grads:
                norm = float(np.linalg.norm(np.asarray(grads, dtype=float)))
                if norm > 0.0:
                    grads = [g / norm for g in grads]
            if self.max_grad_norm is not None:
                norm = float(np.linalg.norm(np.asarray(grads, dtype=float)))
                if norm > self.max_grad_norm and norm > 0.0:
                    scale = self.max_grad_norm / norm
                    grads = [g * scale for g in grads]
            # Stability guard: cap step size if enabled
            step_to_use = self.step_size
            need_L = self.stability_guard and not self.normalize_grads
            if need_L:
                L_est = (
                    self._estimate_preconditioned_lipschitz_bound(etas, preconditioner)
                    if preconditioner is not None
                    else self._estimate_lipschitz_bound(etas)
                )
                if math.isinf(L_est) and not self.line_search:
                    raise ValueError(
                        "the update curvature bound is infinite; enable line_search "
                        "or disable stability_guard explicitly"
                    )
            if self.stability_guard and L_est and L_est > 0.0 and math.isfinite(L_est):
                # Optional: set step directly from Lipschitz estimate (2/L) with safety fraction
                if self.auto_step_from_lipschitz:
                    step_to_use = self.stability_cap_fraction * (2.0 / L_est)
                cap = self.stability_cap_fraction * (2.0 / L_est)
                if cap > 0.0:
                    step_to_use = min(step_to_use, cap)
                    if self.log_step_cap_slack or self.log_contraction_margin:
                        margin = (2.0 / L_est) - step_to_use
                        self._last_step_cap_slack = margin
                        self._step_cap_slack_history.append(float(margin))
                        # Backward-compatible aliases. This value is cap slack,
                        # not a spectral contraction rate.
                        self._last_contraction_margin = margin
                        self._contraction_margin_history.append(float(margin))
                        # Emit warning if margin shrinks below threshold
                        if self.warn_on_margin_shrink and margin < self.margin_warn_threshold:
                            warnings.warn(
                                f"Step-cap slack ({margin:.2e}) below threshold ({self.margin_warn_threshold:.2e}). "
                                f"Consider reducing step_size or coupling weights. "
                                f"Lipschitz bound L={L_est:.2e}, safe step=2/L={2.0/L_est:.2e}, current step={step_to_use:.2e}",
                                UserWarning,
                                stacklevel=2
                            )
            elif self.stability_guard and (self.log_step_cap_slack or self.log_contraction_margin):
                self._last_step_cap_slack = None
                self._step_cap_slack_history.append(float("nan"))
                self._last_contraction_margin = None
                self._contraction_margin_history.append(float("nan"))
            
            # Inject orthogonal noise if enabled (structure-preserving exploration)
            # Tangency and the stationary threshold are defined by the ordinary
            # objective gradient, not by a normalized or clipped update vector.
            noise_grad_vector = np.asarray(objective_grads, dtype=float)
            noise_vector = np.zeros_like(noise_grad_vector)
            current_noise_mag = 0.0
            if noise_mode in {
                "orthogonal",
                "precision_orthogonal",
                "metric_orthogonal",
                "metric_precision_orthogonal",
            }:
                if controller is not None:
                    current_noise_mag = controller.step(
                        noise_grad_vector,
                        energy_drop_ratio=getattr(self, "_last_energy_drop_ratio", 1.0),
                        backtracks=controller_backtracks,
                        iter_idx=iter_idx,
                    )
                else:
                    current_noise_mag = self.noise_magnitude * (self.noise_schedule_decay ** iter_idx)
            elif noise_mode == "isotropic":
                current_noise_mag = self.noise_magnitude * (self.noise_schedule_decay ** iter_idx)
            else:
                current_noise_mag = 0.0
            if current_noise_mag > 1e-9:
                raw_noise = np.random.normal(0, 1, size=noise_grad_vector.shape)
                noise_vector = self._build_noise_vector(
                    raw_noise,
                    noise_grad_vector,
                    current_noise_mag,
                )
            
            # step
            if self.line_search:
                grads_eff = list(grads)
                if preconditioner is not None:
                    grads_eff = [g / d for g, d in zip(grads_eff, preconditioner)]

            # (stability coupling auto-cap removed)
            # Optional: prepare Lipschitz details for allocator/telemetry
            self._last_lipschitz_details = None
            need_details = (
                self.expose_lipschitz_details
                or (self.weight_adapter is not None and any(
                    hasattr(self.weight_adapter, attr) for attr in ("edge_costs", "row_margins", "global_margin")
                ))
            )
            if need_details:
                target_L = (
                    2.0 * float(self.stability_cap_fraction) / float(step_to_use)
                    if step_to_use > 0.0
                    else L_est
                )
                self._last_lipschitz_details = self._estimate_lipschitz_details(
                    etas,
                    smoothing_epsilon=max(self.grad_eps * 0.5, 1e-6),
                    target_L=target_L,
                    preconditioner=preconditioner,
                )
            # step
            if self.line_search:
                # Projected Armijo along the selected descent direction.
                etas = self._step_with_backtracking(
                    etas,
                    objective_grads,
                    grads_eff,
                    step_to_use,
                )
                line_search_failed = self._last_acceptance_reason == "armijo_failed_no_step"
                if not line_search_failed and np.any(noise_vector):
                    etas = apply_box_feasible_noise(etas, noise_vector).tolist()
            else:
                # Apply the deterministic projected update first.
                for i in range(len(etas)):
                    # Use the same diagonal consumed by the stability bound.
                    if preconditioner is not None:
                        g_eff = float(grads[i]) / float(preconditioner[i])
                    else:
                        g_eff = float(grads[i])
                    update = -step_to_use * g_eff
                    etas[i] = float(max(0.0, min(1.0, etas[i] + update)))
                if np.any(noise_vector):
                    etas = apply_box_feasible_noise(etas, noise_vector).tolist()

            # Save the completed proposal's line-search work for the next
            # controller decision. `_last_step_backtracks` remains the public
            # metric for the most recently completed proposal.
            controller_backtracks = int(self._last_step_backtracks)
            
            energy_value = self._energy_value(etas)
            if self.enforce_invariants:
                self._check_invariants(etas, energy_value)
            if prev_energy_value is not None:
                drop = max(prev_energy_value - energy_value, 0.0)
                denom = max(abs(prev_energy_value), 1e-12)
                self._last_energy_drop_ratio = drop / denom
            else:
                self._last_energy_drop_ratio = 1.0
            # Early stop on non-monotonic energy (guard against oscillations).
            # We reject steps that don't improve our objective to prevent instability.
            should_reject = line_search_failed

            # Option A: Free-energy guard (Phase 2)
            # Accept based on F = U - T*S decrease instead of U alone.
            if self.use_free_energy_guard and prev_energy_value is not None:
                # Compute free energy for current and previous states
                prev_free_energy = self._compute_free_energy(etas_prev)
                curr_free_energy = self._compute_free_energy(etas)
                delta_F = curr_free_energy - prev_free_energy
                
                # Accept non-increase within the configured numerical tolerance.
                if delta_F > self.free_energy_epsilon:
                    should_reject = True
                    self._last_acceptance_reason = "free_energy_insufficient_decrease"

            # Option B: Standard energy guard (Phase 1)
            # Strictly enforce monotonic decrease in total energy U.
            elif prev_energy_value is not None and energy_value > prev_energy_value + 1e-12:
                should_reject = True
                self._last_acceptance_reason = "non_monotonic_rejected"
            
            if should_reject:
                if not self._last_acceptance_reason:
                    self._last_acceptance_reason = "non_monotonic_rejected"
                self._rejected_steps += 1
                etas = list(etas_prev)
                energy_value = float(prev_energy_value) if prev_energy_value is not None else self._energy_value(etas)
                self._attempt_energy_history.append(float(energy_value))
                self._acceptance_reason_history.append(str(self._last_acceptance_reason))
                if self.continue_after_rejection:
                    continue
                break
            # Validate the accepted deterministic state. Rejected proposals are
            # restored above and are not part of the monotonic energy contract.
            if (
                self.assert_monotonic_energy
                and self.noise_magnitude <= 1e-12
                and not self.line_search
                and not self.use_free_energy_guard
                and prev_energy_value is not None
            ):
                assert energy_value <= prev_energy_value + self.monotonic_energy_tol, (
                    f"Accepted energy increased: {prev_energy_value:.12e} -> {energy_value:.12e} "
                    f"(delta={energy_value - prev_energy_value:.3e})."
                )
            self._guard_energy_transitions.append(
                (int(self._objective_version), float(prev_energy_value), float(energy_value))
            )
            self._attempt_energy_history.append(float(energy_value))
            # Emit only after acceptance
            # Record acceptance reason for standard/coordinate steps
            if not self.line_search and not self._last_acceptance_reason:
                if self.use_free_energy_guard:
                    self._last_acceptance_reason = "free_energy_accepted"
                elif prev_energy_value is not None and energy_value <= prev_energy_value:
                    self._last_acceptance_reason = "monotone_decrease"
                else:
                    self._last_acceptance_reason = "initial_step"
            self._acceptance_reason_history.append(str(self._last_acceptance_reason))
            self._emit_eta(etas)
            self._emit_energy(energy_value)
            self._record_energy_history(energy_value)
            
            # Early-stop with patience: stop if energy stabilizes
            if self.enable_early_stop and prev_energy_value is not None:
                delta_E = abs(prev_energy_value - energy_value)
                if delta_E < self.early_stop_delta_threshold:
                    self._early_stop_stable_count += 1
                    if self._early_stop_stable_count >= self.early_stop_patience:
                        self._last_acceptance_reason = "early_stop_converged"
                        self._acceptance_reason_history[-1] = self._last_acceptance_reason
                        break
                else:
                    self._early_stop_stable_count = 0
            
            weights_before_update = dict(self._term_weights)
            term_norms = self._term_grad_norms(etas)
            if self.auto_balance_term_weights:
                self._auto_balance_term_weights(term_norms)
            if self.weight_adapter is not None:
                # If adapter supports allocator fields, inject details snapshot
                if self._last_lipschitz_details is not None:
                    if hasattr(self.weight_adapter, "edge_costs"):
                        # The current adapter allocates one weight per coupling
                        # family, so its cost keys must match the term-norm keys.
                        family_costs = self._last_lipschitz_details.get("family_costs", {})
                        try:
                            # type: ignore[attr-defined]
                            self.weight_adapter.edge_costs = {
                                str(k): float(v) for k, v in family_costs.items()
                            }
                        except Exception:
                            pass
                    if hasattr(self.weight_adapter, "row_margins"):
                        row_margins = self._last_lipschitz_details.get("row_margins", {})
                        try:
                            # type: ignore[attr-defined]
                            self.weight_adapter.row_margins = {int(k): float(v) for k, v in row_margins.items()}
                        except Exception:
                            pass
                    if hasattr(self.weight_adapter, "global_margin"):
                        gm = float(self._last_lipschitz_details.get("global_margin", 0.0))
                        try:
                            # type: ignore[attr-defined]
                            self.weight_adapter.global_margin = gm
                        except Exception:
                            pass
                updated = self.weight_adapter.step(term_norms, energy_value, dict(self._term_weights))
                self._term_weights = {
                    str(k): float(v) for k, v in updated.items() if isinstance(k, str)
                }
            if self._term_weights != weights_before_update:
                self._objective_version += 1
                self._early_stop_stable_count = 0
        return etas

    def relax_etas_proximal(
        self,
        etas0: List[OrderParameter],
        steps: int = 50,
        tau: float = 0.05,
    ) -> List[OrderParameter]:
        """Compatibility wrapper for the proximal solver."""
        warnings.warn(
            "relax_etas_proximal is deprecated; configure SolverConfig and call relax_etas",
            DeprecationWarning,
            stacklevel=2,
        )
        config = ProximalSolverConfig(
            steps=steps,
            tau=tau,
            block_mode=self.solver.proximal.block_mode,
        )
        return self._run_with_constraint_snapshot(
            lambda: solve_proximal(self, etas0, config)
        )

    def relax_etas_admm(
        self,
        etas0: List[OrderParameter],
        steps: int = 50,
        rho: float = 1.0,
        step_size: float = 0.05,
    ) -> List[OrderParameter]:
        """Compatibility wrapper for the ADMM-like solver."""
        warnings.warn(
            "relax_etas_admm is deprecated; configure SolverConfig and call relax_etas",
            DeprecationWarning,
            stacklevel=2,
        )
        config = ADMMSolverConfig(
            steps=steps,
            rho=rho,
            step_size=step_size,
            gate_prox=self.solver.admm.gate_prox,
            gate_damping=self.solver.admm.gate_damping,
        )
        return self._run_with_constraint_snapshot(
            lambda: solve_admm(self, etas0, config)
        )

    def _box_derivative(self, function: Callable[[float], float], value: float) -> float:
        return box_derivative(function, value, self.grad_eps)

    def _finite_diff_grads(self, etas: List[OrderParameter]) -> List[float]:
        # Every coordinate owns a local objective, including isolated nodes in
        # partially coupled graphs.  Coupling adjacency therefore cannot be
        # used to prune full-objective finite differences safely.
        grads: List[float] = [0.0 for _ in etas]
        for i, eta in enumerate(etas):
            def energy_at(value: float, *, index: int = i) -> float:
                trial = list(etas)
                trial[index] = float(value)
                return float(self._energy_value(trial))

            grads[i] = self._box_derivative(energy_at, float(eta))
        return grads

    def _analytic_grads(self, etas: List[OrderParameter]) -> List[float]:
        """Analytic grads using optional module/coupling derivatives; finite-diff per term as fallback (no double-count)."""
        n = len(etas)
        grad_arr = self._grad_buffer_for(n)
        # Local terms (apply term weights)
        cw = self._combined_term_weights()
        for idx, (m, eta) in enumerate(zip(self.modules, etas)):
            w = float(cw.get(f"local:{m.__class__.__name__}", 1.0))
            if isinstance(m, SupportsLocalEnergyGrad):
                grad_arr[idx] += w * float(m.d_local_energy_d_eta(float(eta), self.constraints))
            else:
                grad_arr[idx] += w * self._box_derivative(
                    lambda value, module=m: float(module.local_energy(value, self.constraints)),
                    float(eta),
                )
        if self.use_vectorized_quadratic:
            q_grads = self._quadratic_coupling_gradients_vectorized(etas, cw)
            grad_arr += np.asarray(q_grads, dtype=float)
        if self.use_vectorized_hinges:
            hinge_grads = self._hinge_coupling_gradients_vectorized(etas, cw)
            grad_arr += np.asarray(hinge_grads, dtype=float)
        if self.use_vectorized_gate_benefits:
            grad_arr += self._gate_benefit_gradients_vectorized(etas, cw)
        for i, j, coup in self.couplings:
            if self.use_vectorized_quadratic and isinstance(coup, QuadraticCoupling):
                continue
            if self.use_vectorized_hinges and isinstance(coup, (DirectedHingeCoupling, AsymmetricHingeCoupling)):
                continue
            if self.use_vectorized_gate_benefits and isinstance(coup, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                continue
            w = float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
            if isinstance(coup, SupportsCouplingGrads):
                gi, gj = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                grad_arr[i] += w * float(gi)
                grad_arr[j] += w * float(gj)
            else:
                gi = self._box_derivative(
                    lambda value, coupling=coup, other=etas[j]: float(
                        coupling.coupling_energy(value, other, self.constraints)
                    ),
                    float(etas[i]),
                )
                gj = self._box_derivative(
                    lambda value, coupling=coup, other=etas[i]: float(
                        coupling.coupling_energy(other, value, self.constraints)
                    ),
                    float(etas[j]),
                )
                grad_arr[i] += w * gi
                grad_arr[j] += w * gj
        return grad_arr.tolist()

    def _grads(self, etas: List[OrderParameter]) -> List[float]:
        if self.use_analytic:
            try:
                grads = self._analytic_grads(etas)
            except Exception:
                grads = self._finite_diff_grads(etas)
        else:
            grads = self._finite_diff_grads(etas)
        return grads

    def _update_precision_cache(self, etas: List[OrderParameter]) -> None:
        """Update diagonal precision/curvature cache (locals + coupling contributions).
        
        Locals: Uses SupportsPrecision.curvature if available, scaled by local term weight.
        Couplings: Adds curvature from quadratic and active hinge couplings, scaled by coupling term weights.
        """
        update_precision_cache(self, etas)

    def get_precision_diagonal(self) -> List[float]:
        """Return the currently cached diagonal precision vector."""
        return get_precision_diagonal(self)
    
    def _compute_entropy(self, etas: List[OrderParameter]) -> float:
        """Compute Shannon-like entropy for order parameters: S = -Σ[η*log(η) + (1-η)*log(1-η)]."""
        S = 0.0
        for eta_val in etas:
            eta_f = float(max(1e-9, min(1.0 - 1e-9, eta_val)))  # Clamp away from boundaries
            s_i = -(eta_f * math.log(eta_f) + (1.0 - eta_f) * math.log(1.0 - eta_f))
            S += s_i
        return S
    
    def _compute_free_energy(self, etas: List[OrderParameter]) -> float:
        """Compute free energy F = U - T*S where U is internal energy and S is entropy."""
        U = self._energy_value(etas)
        S = self._compute_entropy(etas)
        F = U - float(self.free_energy_temperature) * S
        return F
    
    def _quadratic_coupling_gradients_vectorized(self, etas: List[OrderParameter], cw: dict[str, float]) -> List[float]:
        """Vectorized accumulation of gradients for quadratic couplings."""
        n = len(etas)
        grads = np.zeros(n, dtype=float)
        cache = self._vectorized_cache
        if cache is None or cache.quadratic_i.size == 0:
            return grads.tolist()
        eta_arr = np.asarray(etas, dtype=float)
        term_weights = (
            np.asarray([float(cw.get(key, 1.0)) for key in cache.quadratic_term_keys], dtype=float)
            if cache.quadratic_term_keys
            else np.ones_like(cache.quadratic_weights)
        )
        weights = cache.quadratic_weights * term_weights
        if weights.size == 0:
            return grads.tolist()
        diff = eta_arr[cache.quadratic_i] - eta_arr[cache.quadratic_j]
        gi = 2.0 * weights * diff
        gj = -2.0 * weights * diff
        np.add.at(grads, cache.quadratic_i, gi)
        np.add.at(grads, cache.quadratic_j, gj)
        return grads.tolist()

    def _hinge_coupling_gradients_vectorized(self, etas: List[OrderParameter], cw: dict[str, float]) -> List[float]:
        """Vectorized gradients for directed/asymmetric hinge couplings."""
        n = len(etas)
        grads = np.zeros(n, dtype=float)
        cache = self._vectorized_cache
        if cache is None:
            return grads.tolist()
        eta_arr = np.asarray(etas, dtype=float)

        if cache.directed_i.size > 0:
            weights = cache.directed_weights * (
                np.asarray([float(cw.get(key, 1.0)) for key in cache.directed_term_keys], dtype=float)
                if cache.directed_term_keys
                else 1.0
            )
            gap = eta_arr[cache.directed_j] - eta_arr[cache.directed_i]
            mask = gap > 0.0
            contrib = 2.0 * weights * gap * mask
            np.add.at(grads, cache.directed_i, -contrib)
            np.add.at(grads, cache.directed_j, contrib)

        if cache.asymmetric_i.size > 0:
            weights = cache.asymmetric_weights * (
                np.asarray([float(cw.get(key, 1.0)) for key in cache.asymmetric_term_keys], dtype=float)
                if cache.asymmetric_term_keys
                else 1.0
            )
            gap = cache.asymmetric_beta * eta_arr[cache.asymmetric_j] - cache.asymmetric_alpha * eta_arr[cache.asymmetric_i]
            mask = gap > 0.0
            gi = -2.0 * weights * gap * cache.asymmetric_alpha * mask
            gj = 2.0 * weights * gap * cache.asymmetric_beta * mask
            np.add.at(grads, cache.asymmetric_i, gi)
            np.add.at(grads, cache.asymmetric_j, gj)
        return grads.tolist()

    def _gate_benefit_gradients_vectorized(self, etas: List[OrderParameter], cw: dict[str, float]) -> np.ndarray:
        n = len(etas)
        grads = np.zeros(n, dtype=float)
        cache = self._vectorized_cache
        if cache is None:
            return grads
        eta_arr = np.asarray(etas, dtype=float)

        if cache.gate_idx.size > 0:
            weights = cache.gate_weights * (
                np.asarray([float(cw.get(key, 1.0)) for key in cache.gate_term_keys], dtype=float)
                if cache.gate_term_keys
                else 1.0
            )
            delta = np.asarray(
                [float(self.constraints.get(key, 0.0)) for key in cache.gate_delta_keys],
                dtype=float,
            )
            contrib = -weights * delta
            np.add.at(grads, cache.gate_idx, contrib)

        if cache.damped_idx.size > 0:
            weights = cache.damped_weights * (
                np.asarray([float(cw.get(key, 1.0)) for key in cache.damped_term_keys], dtype=float)
                if cache.damped_term_keys
                else 1.0
            )
            delta = np.asarray(
                [float(self.constraints.get(key, 0.0)) for key in cache.damped_delta_keys],
                dtype=float,
            )
            scaled = np.where(
                delta >= 0.0,
                cache.damped_positive_scale * delta,
                cache.damped_negative_scale * delta,
            )
            gate_vals = eta_arr[cache.damped_idx]
            grad_vals = np.zeros_like(gate_vals)
            mask_nonzero = (scaled != 0.0) & (weights != 0.0) & (cache.damped_damping != 0.0)
            if np.any(mask_nonzero):
                mask_one = mask_nonzero & (cache.damped_eta_power == 1.0)
                grad_vals[mask_one] = (
                    -weights[mask_one] * cache.damped_damping[mask_one] * scaled[mask_one]
                )
                mask_pow = mask_nonzero & (cache.damped_eta_power != 1.0) & (gate_vals > 0.0)
                if np.any(mask_pow):
                    grad_vals[mask_pow] = (
                        -weights[mask_pow]
                        * cache.damped_damping[mask_pow]
                        * scaled[mask_pow]
                        * cache.damped_eta_power[mask_pow]
                        * (gate_vals[mask_pow] ** (cache.damped_eta_power[mask_pow] - 1.0))
                    )
            np.add.at(grads, cache.damped_idx, grad_vals)
        return grads

    def _local_energy_grad_batch(self, etas: List[OrderParameter]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(etas)
        cw = self._combined_term_weights()
        energy_buf = self._local_energy_buffer_for(n)
        grad_buf = self._local_grad_buffer_for(n)
        for idx, (m, eta) in enumerate(zip(self.modules, etas)):
            w = float(cw.get(f"local:{m.__class__.__name__}", 1.0))
            energy = w * float(m.local_energy(float(eta), self.constraints))
            energy_buf[idx] = energy
            if isinstance(m, SupportsLocalEnergyGrad):
                grad_buf[idx] = w * float(m.d_local_energy_d_eta(float(eta), self.constraints))
            else:
                grad_buf[idx] = w * self._box_derivative(
                    lambda value, module=m: float(module.local_energy(value, self.constraints)),
                    float(eta),
                )
        return energy_buf, grad_buf

    def _grad_buffer_for(self, n: int) -> np.ndarray:
        buf = self._grad_buffer
        if buf is None or buf.shape[0] != n:
            buf = np.zeros(n, dtype=float)
            self._grad_buffer = buf
        else:
            buf.fill(0.0)
        return buf

    def _local_energy_buffer_for(self, n: int) -> np.ndarray:
        buf = self._local_energy_buffer
        if buf is None or buf.shape[0] != n:
            buf = np.zeros(n, dtype=float)
            self._local_energy_buffer = buf
        else:
            buf.fill(0.0)
        return buf

    def _local_grad_buffer_for(self, n: int) -> np.ndarray:
        buf = self._local_grad_buffer
        if buf is None or buf.shape[0] != n:
            buf = np.zeros(n, dtype=float)
            self._local_grad_buffer = buf
        else:
            buf.fill(0.0)
        return buf

    def _trial_array_for(self, etas: List[OrderParameter]) -> np.ndarray:
        n = len(etas)
        buf = self._trial_buffer
        if buf is None or buf.shape[0] != n:
            buf = np.asarray(etas, dtype=float).copy()
            self._trial_buffer = buf
        else:
            buf[:] = np.asarray(etas, dtype=float)
        return buf

    def _step_with_backtracking(
        self,
        etas: List[OrderParameter],
        objective_grads: List[float],
        direction: List[float],
        step_init: float,
    ) -> List[float]:
        """Projected Armijo search using the realized box-constrained displacement."""
        F0 = self._energy_value(etas)
        step = float(step_init)
        state = np.asarray(etas, dtype=float)
        gradient = np.asarray(objective_grads, dtype=float)
        direction_vector = np.asarray(direction, dtype=float)
        if gradient.shape != state.shape or direction_vector.shape != state.shape:
            raise ValueError("state, gradient, and line-search direction must have matching shapes")
        for local_bk in range(self.max_backtrack + 1):
            trial_arr = self._trial_array_for(etas)
            trial_arr -= step * direction_vector
            np.clip(trial_arr, 0.0, 1.0, out=trial_arr)
            displacement = trial_arr - state
            directional_change = float(np.dot(gradient, displacement))
            trial = trial_arr.tolist()
            F1 = self._energy_value(trial)
            if directional_change <= 0.0 and F1 <= F0 + self.armijo_c * directional_change:
                self._last_step_backtracks = local_bk
                self._total_backtracks += local_bk
                self._last_acceptance_reason = "armijo_accepted"
                return trial
            if local_bk < self.max_backtrack:
                step *= self.backtrack_factor
        self._last_step_backtracks = self.max_backtrack
        self._total_backtracks += self.max_backtrack
        self._last_acceptance_reason = "armijo_failed_no_step"
        return list(etas)

    def _estimate_lipschitz_bound(self, etas: List[OrderParameter]) -> float:
        """Conservative Gershgorin-style bound on gradient Lipschitz constant.

        Approximates diagonal (local curvature) via finite differences of local gradient
        and adds coupling curvature contributions for quadratic/hinge families.
        """
        return estimate_lipschitz_bound(self, etas)

    def _estimate_preconditioned_lipschitz_bound(
        self,
        etas: List[OrderParameter],
        preconditioner: np.ndarray,
    ) -> float:
        """Bound the Hessian normalized by the diagonal used for this update."""
        return estimate_preconditioned_lipschitz_bound(self, etas, preconditioner)

    def _estimate_lipschitz_details(
        self,
        etas: List[OrderParameter],
        smoothing_epsilon: float = 1e-3,
        target_L: Optional[float] = None,
        preconditioner: Optional[np.ndarray] = None,
    ) -> dict:
        """Return row contributions in the same geometry as the guarded update.

        Produces:
          - L_est: current Lipschitz estimate (float)
          - row_sums: dict[row_index -> row_sum]
          - row_targets: dict[row_index -> target_row_sum] (proportional scaling if target_L < L_est)
          - row_margins: dict[row_index -> max(0, target - current)]
          - global_margin: max(0, target_L - L_est)
          - family_costs: dict['coup:ClassName' -> ΔL per unit relative scaling (max over rows)]

        Built-in hinge families use their maximum curvature across both active
        regions, so a proposal that crosses a hinge boundary remains covered.
        """
        del smoothing_epsilon  # retained for call-site compatibility
        n = len(etas)
        if n == 0:
            return {
                "L_est": 0.0,
                "row_sums": {},
                "row_targets": {},
                "row_margins": {},
                "global_margin": 0.0,
                "family_costs": {},
            }
        if preconditioner is None:
            precision = np.ones(n, dtype=float)
            geometry = "unpreconditioned"
        else:
            precision = np.asarray(preconditioner, dtype=float)
            if precision.shape != (n,):
                raise ValueError("preconditioner shape must match the coordinator state")
            if not np.all(np.isfinite(precision)) or np.any(precision <= 0.0):
                raise ValueError("preconditioner entries must be finite and positive")
            geometry = "preconditioned"

        row_sums = np.zeros(n, dtype=float)
        probe_epsilon = max(self.grad_eps * 0.5, 1e-6)
        for index, eta in enumerate(etas):
            lower = max(0.0, min(1.0, float(eta) - probe_epsilon))
            upper = max(0.0, min(1.0, float(eta) + probe_epsilon))
            width = upper - lower
            if width <= 0.0:
                continue
            curvature = (self._local_grad(index, upper) - self._local_grad(index, lower)) / width
            if math.isnan(curvature):
                raise ValueError("local curvature estimate must not be NaN")
            row_sums[index] += abs(float(curvature)) / float(precision[index])
        # Per-row, per-family contributions to row sum
        per_row_family = {}  # row -> {family_key: contrib}
        for r in range(n):
            per_row_family[r] = {}

        def _add_row_family(row: int, fam: str, amount: float) -> None:
            if amount == 0.0:
                return
            d = per_row_family[row]
            d[fam] = float(d.get(fam, 0.0) + amount)

        # Track per-edge costs in the same normalized row geometry.
        edge_costs: dict[int, float] = {}
        combined_weights = self._combined_term_weights()
        for edge_idx, (i, j, coup) in enumerate(self.couplings):
            key = f"coup:{coup.__class__.__name__}"
            w_eff = float(combined_weights.get(key, 1.0))
            if w_eff == 0.0:
                edge_costs[edge_idx] = 0.0
                continue
            diag_i = 0.0
            diag_j = 0.0
            off_ij = 0.0
            if isinstance(coup, QuadraticCoupling):
                add = 2.0 * abs(float(getattr(coup, "weight", 0.0)) * w_eff)
                diag_i = add
                diag_j = add
                off_ij = add
            elif isinstance(coup, DirectedHingeCoupling):
                add = 2.0 * abs(float(getattr(coup, "weight", 0.0)) * w_eff)
                diag_i = add
                diag_j = add
                off_ij = add
            elif isinstance(coup, AsymmetricHingeCoupling):
                w = abs(float(getattr(coup, "weight", 0.0)) * w_eff)
                alpha = float(getattr(coup, "alpha_i", 1.0))
                beta = float(getattr(coup, "beta_j", 1.0))
                diag_i = 2.0 * w * (alpha * alpha)
                diag_j = 2.0 * w * (beta * beta)
                off_ij = 2.0 * w * abs(alpha * beta)
            elif isinstance(coup, SupportsCouplingCurvature):
                reported_i, reported_j, reported_off = validate_curvature_bound_triplet(
                    coup.coupling_curvature_bounds(
                        float(etas[i]),
                        float(etas[j]),
                        self.constraints,
                    )
                )
                scale = abs(w_eff)
                diag_i = scale * reported_i
                diag_j = scale * reported_j
                off_ij = scale * reported_off
            else:
                diag_i = math.inf
                diag_j = math.inf

            normalized_off = off_ij / math.sqrt(float(precision[i] * precision[j]))
            contribution_i = diag_i / float(precision[i]) + normalized_off
            contribution_j = diag_j / float(precision[j]) + normalized_off
            row_sums[i] += contribution_i
            row_sums[j] += contribution_j
            _add_row_family(i, key, contribution_i)
            _add_row_family(j, key, contribution_j)
            edge_costs[edge_idx] = float(max(contribution_i, contribution_j))

        L_est = float(np.max(row_sums)) if row_sums.size > 0 else 0.0
        # Targets and margins
        if not target_L or not math.isfinite(target_L) or target_L <= 0.0:
            target_L = L_est
        row_targets = {}
        row_margins = {}
        for r in range(n):
            row_targets[r] = float(target_L)
            row_margins[r] = max(0.0, float(target_L - row_sums[r]))
        global_margin = max(0.0, float(target_L - L_est))

        # Family costs: max row contribution for each family
        family_costs: dict[str, float] = {}
        for r, fam_map in per_row_family.items():
            for fam, amount in fam_map.items():
                current = family_costs.get(fam, 0.0)
                if amount > current:
                    family_costs[fam] = float(amount)
        
        # Build compact row_sums dict
        row_sums_dict = {i: float(row_sums[i]) for i in range(n)}

        return {
            "L_est": L_est,
            "row_sums": row_sums_dict,
            "row_targets": row_targets,
            "row_margins": row_margins,
            "global_margin": global_margin,
            "family_costs": family_costs,
            "edge_costs": {int(k): float(v) for k, v in edge_costs.items()},
            "geometry": geometry,
            "preconditioner": {i: float(precision[i]) for i in range(n)},
        }

    def _record_energy_history(self, energy: float) -> None:
        self._accepted_energy_history.append(float(energy))
        if len(self._accepted_energy_history) > 256:
            self._accepted_energy_history = self._accepted_energy_history[-256:]

    def last_relaxation_metrics(self) -> Mapping[str, Any]:
        """Expose basic observability for the most recent relaxation run."""
        history = list(self._accepted_energy_history)
        guard_transitions = [
            {
                "objective_version": int(version),
                "energy_before": float(before),
                "energy_after": float(after),
            }
            for version, before, after in self._guard_energy_transitions
        ]
        return {
            "accepted_steps": len(history),
            "rejected_steps": int(self._rejected_steps),
            "attempted_steps": len(self._attempt_energy_history),
            "energy_trace": history,
            "attempt_energy_trace": list(self._attempt_energy_history),
            "last_acceptance_reason": self._last_acceptance_reason,
            "acceptance_reasons": list(self._acceptance_reason_history),
            "objective_version": int(self._objective_version),
            "guard_transitions": guard_transitions,
            "last_energy_drop_ratio": float(self._last_energy_drop_ratio),
            "last_step_cap_slack": self._last_step_cap_slack,
            "step_cap_slacks": list(self._step_cap_slack_history),
            # Deprecated compatibility names retained for existing log readers.
            "last_contraction_margin": self._last_contraction_margin,
            "contraction_margins": list(self._contraction_margin_history),
        }

    def last_solver_metrics(self) -> Mapping[str, Any]:
        """Return mode-specific metrics for the most recent solver run."""
        if self.solver.mode == SolverMode.GRADIENT:
            metrics = dict(self.last_relaxation_metrics())
            metrics["solver_mode"] = "gradient"
            return metrics
        return dict(self._last_solver_metrics)

    def _emit_eta(self, etas: List[OrderParameter]) -> None:
        for cb in self.on_eta_updated:
            cb(etas)

    def _emit_energy(self, F: float) -> None:
        for cb in self.on_energy_updated:
            cb(F)

    # --- Helpers for adjacency and local/edge terms ---
    def _ensure_adjacency(self, n: int) -> None:
        if self._adjacency is not None:
            return
        adj: List[List[Tuple[int, EnergyCoupling]]] = [[] for _ in range(n)]
        for i, j, coup in self.couplings:
            adj[i].append((j, coup))
            adj[j].append((i, coup))
        self._adjacency = adj

    def _build_vectorized_cache(self) -> None:
        """Pre-compute sparse index structures for vectorized kernels."""
        if not self.couplings:
            self._vectorized_cache = _VectorizedCouplingCache(
                quadratic_i=np.zeros(0, dtype=int),
                quadratic_j=np.zeros(0, dtype=int),
                quadratic_weights=np.zeros(0, dtype=float),
                quadratic_term_keys=tuple(),
                directed_i=np.zeros(0, dtype=int),
                directed_j=np.zeros(0, dtype=int),
                directed_weights=np.zeros(0, dtype=float),
                directed_term_keys=tuple(),
                asymmetric_i=np.zeros(0, dtype=int),
                asymmetric_j=np.zeros(0, dtype=int),
                asymmetric_weights=np.zeros(0, dtype=float),
                asymmetric_term_keys=tuple(),
                asymmetric_alpha=np.zeros(0, dtype=float),
                asymmetric_beta=np.zeros(0, dtype=float),
                gate_idx=np.zeros(0, dtype=int),
                gate_weights=np.zeros(0, dtype=float),
                gate_term_keys=tuple(),
                gate_delta_keys=tuple(),
                damped_idx=np.zeros(0, dtype=int),
                damped_weights=np.zeros(0, dtype=float),
                damped_term_keys=tuple(),
                damped_delta_keys=tuple(),
                damped_damping=np.zeros(0, dtype=float),
                damped_eta_power=np.zeros(0, dtype=float),
                damped_positive_scale=np.zeros(0, dtype=float),
                damped_negative_scale=np.zeros(0, dtype=float),
            )
            return

        def _tuple_keys(items: List[str]) -> Tuple[str, ...]:
            return tuple(items)

        quadratic = [(i, j, coup) for i, j, coup in self.couplings if isinstance(coup, QuadraticCoupling)]
        directed = [(i, j, coup) for i, j, coup in self.couplings if isinstance(coup, DirectedHingeCoupling)]
        asymmetric = [(i, j, coup) for i, j, coup in self.couplings if isinstance(coup, AsymmetricHingeCoupling)]
        gate = [(i, j, coup) for i, j, coup in self.couplings if isinstance(coup, GateBenefitCoupling)]
        damped = [(i, j, coup) for i, j, coup in self.couplings if isinstance(coup, DampedGateBenefitCoupling)]

        cache = _VectorizedCouplingCache(
            quadratic_i=np.asarray([i for i, _, _ in quadratic], dtype=int) if quadratic else np.zeros(0, dtype=int),
            quadratic_j=np.asarray([j for _, j, _ in quadratic], dtype=int) if quadratic else np.zeros(0, dtype=int),
            quadratic_weights=np.asarray(
                [float(getattr(coup, "weight", 0.0)) for _, _, coup in quadratic], dtype=float
            )
            if quadratic
            else np.zeros(0, dtype=float),
            quadratic_term_keys=_tuple_keys([f"coup:{coup.__class__.__name__}" for _, _, coup in quadratic]),
            directed_i=np.asarray([i for i, _, _ in directed], dtype=int) if directed else np.zeros(0, dtype=int),
            directed_j=np.asarray([j for _, j, _ in directed], dtype=int) if directed else np.zeros(0, dtype=int),
            directed_weights=np.asarray(
                [float(getattr(coup, "weight", 0.0)) for _, _, coup in directed], dtype=float
            )
            if directed
            else np.zeros(0, dtype=float),
            directed_term_keys=_tuple_keys([f"coup:{coup.__class__.__name__}" for _, _, coup in directed]),
            asymmetric_i=np.asarray([i for i, _, _ in asymmetric], dtype=int) if asymmetric else np.zeros(0, dtype=int),
            asymmetric_j=np.asarray([j for _, j, _ in asymmetric], dtype=int) if asymmetric else np.zeros(0, dtype=int),
            asymmetric_weights=np.asarray(
                [float(getattr(coup, "weight", 0.0)) for _, _, coup in asymmetric], dtype=float
            )
            if asymmetric
            else np.zeros(0, dtype=float),
            asymmetric_term_keys=_tuple_keys([f"coup:{coup.__class__.__name__}" for _, _, coup in asymmetric]),
            asymmetric_alpha=np.asarray([float(coup.alpha_i) for _, _, coup in asymmetric], dtype=float)
            if asymmetric
            else np.zeros(0, dtype=float),
            asymmetric_beta=np.asarray([float(coup.beta_j) for _, _, coup in asymmetric], dtype=float)
            if asymmetric
            else np.zeros(0, dtype=float),
            gate_idx=np.asarray([i for i, _, _ in gate], dtype=int) if gate else np.zeros(0, dtype=int),
            gate_weights=np.asarray([float(coup.weight) for _, _, coup in gate], dtype=float)
            if gate
            else np.zeros(0, dtype=float),
            gate_term_keys=_tuple_keys([f"coup:{coup.__class__.__name__}" for _, _, coup in gate]),
            gate_delta_keys=_tuple_keys([str(coup.delta_key) for _, _, coup in gate]),
            damped_idx=np.asarray([i for i, _, _ in damped], dtype=int) if damped else np.zeros(0, dtype=int),
            damped_weights=np.asarray([float(coup.weight) for _, _, coup in damped], dtype=float)
            if damped
            else np.zeros(0, dtype=float),
            damped_term_keys=_tuple_keys([f"coup:{coup.__class__.__name__}" for _, _, coup in damped]),
            damped_delta_keys=_tuple_keys([str(coup.delta_key) for _, _, coup in damped]),
            damped_damping=np.asarray([float(coup.damping) for _, _, coup in damped], dtype=float)
            if damped
            else np.zeros(0, dtype=float),
            damped_eta_power=np.asarray([float(coup.eta_power) for _, _, coup in damped], dtype=float)
            if damped
            else np.zeros(0, dtype=float),
            damped_positive_scale=np.asarray([float(coup.positive_scale) for _, _, coup in damped], dtype=float)
            if damped
            else np.zeros(0, dtype=float),
            damped_negative_scale=np.asarray([float(coup.negative_scale) for _, _, coup in damped], dtype=float)
            if damped
            else np.zeros(0, dtype=float),
        )
        self._vectorized_cache = cache

    def rebuild_vectorization_cache(self) -> None:
        """Public hook when couplings change at runtime."""
        self._vectorized_cache = None
        self._build_vectorized_cache()

    def _local_grad(self, idx: int, eta_i: float) -> float:
        m = self.modules[idx]
        w = float(self._combined_term_weights().get(f"local:{m.__class__.__name__}", 1.0))
        if isinstance(m, SupportsLocalEnergyGrad):
            return float(w * m.d_local_energy_d_eta(float(eta_i), self.constraints))
        return float(
            w
            * self._box_derivative(
                lambda value: float(m.local_energy(value, self.constraints)),
                float(eta_i),
            )
        )

    def _combined_term_weights(self) -> dict[str, float]:
        base_tw: dict[str, float] = {}
        tw = self.constraints.get("term_weights", None)
        if isinstance(tw, Mapping):
            for k, v in tw.items():
                try:
                    base_tw[str(k)] = float(v)  # type: ignore[arg-type]
                except Exception:
                    continue
        if self._term_weights:
            base_tw.update({str(k): float(v) for k, v in self._term_weights.items()})
        floor = float(self.term_weight_floor)
        ceiling = None if self.term_weight_ceiling is None else float(self.term_weight_ceiling)
        if floor < 0.0:
            raise ValueError("term_weight_floor must be >= 0")
        if ceiling is not None and ceiling < floor:
            raise ValueError("term_weight_ceiling must be >= floor")
        calibrated: dict[str, float] = {}
        for key, value in base_tw.items():
            v = float(value)
            if not math.isfinite(v):
                raise ValueError(f"term weight {key!r} must be finite")
            if floor:
                v = max(v, floor)
            if ceiling is not None:
                v = min(v, ceiling)
            calibrated[key] = v
        return calibrated

    def _term_grad_norms(self, etas: List[OrderParameter]) -> dict[str, float]:
        """Compute L2 norms of term-specific gradient contributions (weighted)."""
        norms_sq: dict[str, float] = {}
        cw = self._combined_term_weights()
        local_grads = self._local_energy_grad_batch(etas)[1]
        for idx, m in enumerate(self.modules):
            key = f"local:{m.__class__.__name__}"
            g = float(local_grads[idx])
            norms_sq[key] = float(norms_sq.get(key, 0.0) + g * g)
        # couplings
        for i, j, coup in self.couplings:
            key = f"coup:{coup.__class__.__name__}"
            w = float(cw.get(key, 1.0))
            if isinstance(coup, SupportsCouplingGrads):
                gi, gj = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                gi = w * float(gi)
                gj = w * float(gj)
            else:
                gi = w * self._box_derivative(
                    lambda value, coupling=coup, other=etas[j]: float(
                        coupling.coupling_energy(value, other, self.constraints)
                    ),
                    float(etas[i]),
                )
                gj = w * self._box_derivative(
                    lambda value, coupling=coup, other=etas[i]: float(
                        coupling.coupling_energy(other, value, self.constraints)
                    ),
                    float(etas[j]),
                )
            norms_sq[key] = float(norms_sq.get(key, 0.0) + gi * gi + gj * gj)
        # sqrt
        return {k: float(math.sqrt(v)) for k, v in norms_sq.items()}

    def _auto_balance_term_weights(self, term_norms: Mapping[str, float]) -> None:
        if not term_norms:
            return
        target = max(float(self.term_norm_target), 1e-9)
        ratio_cap = max(float(self.max_term_norm_ratio), 1.0)
        for key, norm in term_norms.items():
            norm = float(norm)
            if not math.isfinite(norm) or norm <= 0.0:
                continue
            ratio = norm / target
            if ratio <= ratio_cap:
                continue
            current = float(self._term_weights.get(key, 1.0))
            scale = target / norm
            new_weight = current * scale
            self._term_weights[key] = new_weight
            warnings.warn(
                f"Term '{key}' gradient norm {norm:.3f} exceeded target {target:.3f}; "
                f"auto-balancing weight from {current:.3f} to {new_weight:.3f}.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _validate_configuration(self) -> None:
        assert isinstance(self.modules, list) and len(self.modules) > 0, "at least one module required"
        assert self.grad_eps > 0.0, "grad_eps must be > 0"
        assert self.step_size > 0.0, "step_size must be > 0"
        assert self.stiffness_epsilon > 0.0, "stiffness_epsilon must be > 0"
        assert self.precision_epsilon > 0.0, "precision_epsilon must be > 0"
        assert 0.0 < self.stability_cap_fraction < 1.0, "stability_cap_fraction must be in (0, 1)"
        if self.noise_mode is not None:
            allowed_noise_modes = {
                "none",
                "isotropic",
                "orthogonal",
                "precision_orthogonal",
                "metric_orthogonal",
                "metric_precision_orthogonal",
            }
            assert str(self.noise_mode).lower() in allowed_noise_modes, f"noise_mode must be one of {allowed_noise_modes}"
        configured_noise_mode = (
            str(self.noise_mode).lower()
            if self.noise_mode is not None
            else (
                "metric_precision_orthogonal"
                if self.metric_aware_noise_controller and self.precision_aware_noise_controller
                else "metric_orthogonal" if self.metric_aware_noise_controller else None
            )
        )
        if configured_noise_mode in {"metric_orthogonal", "metric_precision_orthogonal"}:
            assert self.metric_matrix is not None or self.metric_solve is not None, (
                "metric-aware noise requires metric_matrix or metric_solve"
            )
            assert not (self.metric_matrix is not None and self.metric_solve is not None), (
                "provide metric_matrix or metric_solve, not both"
            )
        if self.metric_matrix is not None:
            metric = np.asarray(self.metric_matrix, dtype=float)
            expected_shape = (len(self.modules), len(self.modules))
            assert metric.shape == expected_shape, f"metric_matrix must have shape {expected_shape}"
            assert np.all(np.isfinite(metric)), "metric_matrix must contain finite values"
            assert np.allclose(metric, metric.T, rtol=0.0, atol=1e-12), "metric_matrix must be symmetric"
            try:
                np.linalg.cholesky(metric)
            except np.linalg.LinAlgError as exc:
                raise AssertionError("metric_matrix must be positive definite") from exc
        assert 0.0 < self.armijo_c < 1.0, "armijo_c must be between 0 and 1"
        assert 0.0 < self.backtrack_factor < 1.0, "backtrack_factor must be in (0,1)"
        assert self.max_backtrack >= 0, "max_backtrack must be non-negative"
        if self.normalize_grads and self.stability_guard:
            assert self.line_search, (
                "normalize_grads with stability_guard requires line_search because the spectral cap "
                "does not certify normalized-gradient dynamics"
            )
        if self.term_weight_ceiling is not None:
            assert self.term_weight_ceiling >= self.term_weight_floor >= 0.0
        for i, j, _ in self.couplings:
            assert 0 <= i < len(self.modules), "coupling index out of range"
            assert 0 <= j < len(self.modules), "coupling index out of range"

    def _check_invariants(self, etas: List[OrderParameter], energy_value: Optional[float] = None) -> None:
        tol = 1e-9
        for eta in etas:
            assert math.isfinite(eta), "η must be finite"
            assert -tol <= eta <= 1.0 + tol, "η out of bounds"
        if energy_value is not None:
            assert math.isfinite(energy_value), "Energy must be finite"
