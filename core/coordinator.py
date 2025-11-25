"""Coordinator for total energy evaluation and optional eta relaxation.

NOTE: CODE IS FROM CFC V0.4 There are many extra functions and 
classes that are not used in this paper demo repo
(to be cleaned up)

This coordinator can:
- compute etas from inputs via modules
- compute total energy with couplings
- optionally relax etas by gradient steps on F_total (finite-difference)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import warnings
import numpy as np

from .interfaces import (
    EnergyModule,
    EnergyCoupling,
    OrderParameter,
    SupportsLocalEnergyGrad,
    SupportsCouplingGrads,
    SupportsPrecision,
    WeightAdapter,
)
from .couplings import (
    QuadraticCoupling,
    DirectedHingeCoupling,
    AsymmetricHingeCoupling,
    GateBenefitCoupling,
    DampedGateBenefitCoupling,
)
from .energy import project_noise_orthogonal, project_noise_metric_orthogonal, total_energy
from .prox_utils import prox_asym_hinge_pair, prox_linear_gate, prox_quadratic_pair
from .agm_metrics import compute_agm_phase_metrics, compute_uncertainty_metrics
from .noise_controller import OrthogonalNoiseController, PrecisionNoiseController


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


def _is_gate_module(module: EnergyModule) -> bool:
    return hasattr(module, "cost") and module.__class__.__name__ == "EnergyGatingModule"


@dataclass
class EnergyCoordinator:
    """Energy coordinator with simple event hooks."""

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
    use_vectorized_quadratic: bool = False
    use_vectorized_hinges: bool = False
    use_vectorized_gate_benefits: bool = True
    neighbor_gradients_only: bool = True
    use_coordinate_descent: bool = False
    adaptive_coordinate_descent: bool = False
    coordinate_steps: int = 100
    coordinate_active_tol: float = 1e-4
    adaptive_switch_delta: float = 1e-5
    adaptive_switch_patience: int = 5
    enforce_invariants: bool = True
    # Mirror/logit parameterization for bounded η updates (gradient mode)
    use_logit_updates: bool = False
    logit_epsilon: float = 1e-8
    # Operator-splitting / proximal mode
    operator_splitting: bool = False
    prox_tau: float = 0.05
    prox_steps: int = 50
    prox_block_mode: Optional[str] = None  # e.g., "star"
    # ADMM (experimental, quadratic couplings focus)
    use_admm: bool = False
    admm_rho: float = 1.0
    admm_steps: int = 50
    admm_step_size: float = 0.05
    admm_gate_prox: bool = True  # apply prox-linear step for gate-benefit after gradient update
    admm_gate_damping: float = 0.5  # blend factor for prox step (0..1)
    # Free-energy guard: F = U - T*S acceptance (Phase 2)
    use_free_energy_guard: bool = False
    free_energy_temperature: float = 1.0
    free_energy_epsilon: float = 1e-6
    # Early-stop with patience (Phase 2)
    enable_early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_delta_threshold: float = 1e-6
    # Sensitivity probes / Uncertainty tracking (Phase 2)
    enable_sensitivity_probes: bool = False
    sensitivity_probe_window: int = 10
    # Homotopy / continuation
    homotopy_coupling_scale_start: Optional[float] = None  # scale applied to all coupling term weights
    homotopy_term_scale_starts: Optional[Mapping[str, float]] = None  # individual term keys -> start scale
    homotopy_gate_cost_scale_start: Optional[float] = None
    homotopy_steps: int = 0
    # Homotopy guards (oscillation/backoff)
    enable_homotopy_guards: bool = True
    homotopy_backoff_factor: float = 0.5
    homotopy_min_start_scale: float = 0.05
    homotopy_oscillation_patience: int = 1
    # Term-weight calibration
    term_weight_floor: float = 0.0
    term_weight_ceiling: Optional[float] = None
    auto_balance_term_weights: bool = False
    term_norm_target: float = 1.0
    max_term_norm_ratio: float = 10.0
    # Optional term-weight adapter
    weight_adapter: Optional[WeightAdapter] = None
    # Stability / small-gain guard (optional)
    stability_guard: bool = True
    stability_cap_fraction: float = 0.9  # cap step to this fraction of 2/L estimate
    log_contraction_margin: bool = False
    stability_coupling_auto_cap: bool = False
    stability_coupling_target: Optional[float] = None  # desired max Lipschitz (if None, auto)
    warn_on_margin_shrink: bool = False  # emit Python warnings when margin drops below threshold
    margin_warn_threshold: float = 1e-6  # threshold for margin warnings
    # Lipschitz/allocator details (instrumentation for adapters/telemetry)
    expose_lipschitz_details: bool = False
    # Noise / Exploration controls
    enable_orthogonal_noise: bool = True  # Inject noise orthogonal to gradient (structure-preserving)
    # Note: default magnitude is 0.0 to preserve determinism unless explicitly enabled in experiments.
    noise_magnitude: float = 0.0
    noise_schedule_decay: float = 0.99  # Simple exponential decay for noise magnitude
    auto_noise_controller: bool = False  # Adapt noise magnitude using orthogonal-noise controller
    # Metric-aware projection (optional)
    metric_aware_noise_controller: bool = False
    metric_matrix: Optional[np.ndarray] = None
    metric_vector_product: Optional[Callable[[np.ndarray], np.ndarray]] = None
    # Precision-aware noise & steps
    precision_aware_noise_controller: bool = False
    use_precision_preconditioning: bool = True
    precision_epsilon: float = 1e-8
    # Auto step selection from Lipschitz bound (optional; requires stability_guard)
    auto_step_from_lipschitz: bool = False
    # Uncertainty-gated thresholds for gate costs
    enable_uncertainty_gate: bool = False
    gate_cost_relax_scale: float = 0.85
    gate_cost_tighten_scale: float = 1.15
    gate_cost_floor: float = 1e-4
    gate_cost_ceiling: Optional[float] = None
    gate_rate_exploit_threshold: float = 0.7
    gate_rate_explore_threshold: float = 0.3
    gate_uncertainty_relax_threshold: float = 0.3
    gate_uncertainty_tight_threshold: float = 1.0
    gate_cost_smoothing: float = 0.25
    # Energy conservation check (enabled by default; aligns with repo's monotonic energy goal)
    assert_monotonic_energy: bool = True  # Assert F_t+1 ≤ F_t in deterministic mode; guards auto-skip for noise/line-search
    monotonic_energy_tol: float = 1e-10  # Tolerance for numeric jitter
    # Escape events (noise-triggered basin transitions)
    enable_escape_event_logging: bool = False
    escape_displacement_min_norm: float = 1e-3
    escape_alignment_max_cosine: float = 0.2
    escape_min_energy_drop: float = 1e-6
    escape_noise_min_magnitude: float = 1e-6
    # Confidence fusion (tracks confidence trajectory)
    enable_confidence_logging: bool = False
    confidence_a: float = 1.0
    confidence_b: float = 1.0
    confidence_rho_max: float = 1.0

    on_eta_updated: List[EtaUpdateCallback] = field(default_factory=list)
    on_energy_updated: List[EnergyUpdateCallback] = field(default_factory=list)

    _adjacency: Optional[List[List[Tuple[int, EnergyCoupling]]]] = field(default=None, init=False, repr=False)
    _term_weights: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _grad_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _trial_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _local_energy_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _local_grad_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _adaptive_switches: int = field(default=0, init=False, repr=False)
    _total_backtracks: int = field(default=0, init=False, repr=False)
    _last_step_backtracks: int = field(default=0, init=False, repr=False)
    _last_acceptance_reason: Optional[str] = field(default=None, init=False, repr=False)
    _last_contraction_margin: Optional[float] = field(default=None, init=False, repr=False)
    _stability_coupling_scale: Optional[float] = field(default=None, init=False, repr=False)
    _homotopy_term_scales: Optional[dict[str, float]] = field(default=None, init=False, repr=False)
    _homotopy_gate_bases: Optional[List[float]] = field(default=None, init=False, repr=False)
    _homotopy_runtime_start: Optional[float] = field(default=None, init=False, repr=False)
    _homotopy_backoffs: int = field(default=0, init=False, repr=False)
    _last_lipschitz_details: Optional[dict] = field(default=None, init=False, repr=False)
    _noise_controller: Optional[OrthogonalNoiseController] = field(default=None, init=False, repr=False)
    _last_energy_drop_ratio: float = field(default=1.0, init=False, repr=False)
    _gate_uncertainty_scale: float = field(default=1.0, init=False, repr=False)
    _accepted_energy_history: List[float] = field(default_factory=list, init=False, repr=False)
    _contraction_margin_history: List[float] = field(default_factory=list, init=False, repr=False)
    _early_stop_stable_count: int = field(default=0, init=False, repr=False)
    _probe_dispersion_history: List[float] = field(default_factory=list, init=False, repr=False)
    _probe_trajectory_window: List[List[float]] = field(default_factory=list, init=False, repr=False)
    _vectorized_cache: Optional[_VectorizedCouplingCache] = field(default=None, init=False, repr=False)
    _precision_cache: Optional[Dict[int, float]] = field(default=None, init=False, repr=False)
    _escape_events_count: int = field(default=0, init=False, repr=False)
    _confidence_history: List[float] = field(default_factory=list, init=False, repr=False)
    _initial_energy_value: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_configuration()
        self._ensure_adjacency(len(self.modules))
        self._build_vectorized_cache()
        if self.auto_noise_controller and self.enable_orthogonal_noise:
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

    def _energy_value(self, etas: List[OrderParameter]) -> float:
        # Merge term weights (constraints.term_weights overridden by adapter-maintained _term_weights)
        merged_constraints: dict[str, Any] = dict(self.constraints)
        calibrated_weights = self._combined_term_weights()
        if calibrated_weights:
            merged_constraints["term_weights"] = calibrated_weights
        return total_energy(etas, self.modules, self.couplings, merged_constraints)

    def relax_etas(self, etas0: List[OrderParameter], steps: int = 50) -> List[OrderParameter]:
        """Finite-difference gradient steps on F_total w.r.t. etas."""
        # Optional operator-splitting path
        if self.operator_splitting:
            return self.relax_etas_proximal(etas0, steps=self.prox_steps, tau=self.prox_tau)
        if self.use_admm:
            return self.relax_etas_admm(etas0, steps=self.admm_steps, rho=self.admm_rho, step_size=self.admm_step_size)
        etas = [float(e) for e in etas0]
        if self.use_coordinate_descent:
            return self.relax_etas_coordinate(
                etas,
                steps=self.coordinate_steps,
                active_tol=self.coordinate_active_tol,
            )
        self._homotopy_scale = None
        self._stability_coupling_scale = None
        self._homotopy_term_scales = None
        self._homotopy_gate_bases = None
        controller = self._noise_controller if (self.auto_noise_controller and self.enable_orthogonal_noise) else None
        if controller is not None:
            controller.base_magnitude = float(self.noise_magnitude)
            controller.decay = float(self.noise_schedule_decay)
            controller.reset()
        homotopy_active = (
            self.homotopy_coupling_scale_start is not None
            and self.homotopy_coupling_scale_start >= 0.0
            and self.homotopy_steps > 0
        )
        if homotopy_active and self._homotopy_runtime_start is None:
            self._homotopy_runtime_start = float(self.homotopy_coupling_scale_start)  # runtime backoff-able
        gate_modules = [m for m in self.modules if _is_gate_module(m)]
        if gate_modules and self._homotopy_gate_bases is None:
            self._homotopy_gate_bases = [float(getattr(m, "cost", 0.0)) for m in gate_modules]
        stalled_steps = 0
        energy_value = self._energy_value(etas)
        prev_energy_value: Optional[float] = energy_value
        self._gate_uncertainty_scale = 1.0
        self._accepted_energy_history = []
        self._contraction_margin_history = []
        oscillations = 0
        if self.adaptive_coordinate_descent:
            etas = self.relax_etas_coordinate(
                etas,
                steps=self.coordinate_steps,
                active_tol=self.coordinate_active_tol,
            )
            energy_value = self._energy_value(etas)
            prev_energy_value = energy_value
        for iter_idx in range(steps):
            etas_prev = list(etas)
            L_est = None
            gate_homotopy_scale = 1.0
            if homotopy_active:
                t = min(1.0, iter_idx / float(self.homotopy_steps))
                start = float(self._homotopy_runtime_start if self._homotopy_runtime_start is not None else self.homotopy_coupling_scale_start)  # type: ignore[arg-type]
                scale = start + (1.0 - start) * t
                self._homotopy_scale = max(0.0, scale)
            if self.homotopy_term_scale_starts and self.homotopy_steps > 0:
                t = min(1.0, iter_idx / float(self.homotopy_steps))
                term_scales = {}
                for key, start in self.homotopy_term_scale_starts.items():
                    start = float(start)
                    term_scales[str(key)] = max(0.0, start + (1.0 - start) * t)
                self._homotopy_term_scales = term_scales
            if self.homotopy_gate_cost_scale_start is not None and self.homotopy_steps > 0 and gate_modules:
                t = min(1.0, iter_idx / float(self.homotopy_steps))
                start = float(self.homotopy_gate_cost_scale_start)
                gate_homotopy_scale = max(0.0, start + (1.0 - start) * t)
            if gate_modules:
                self._apply_gate_costs(gate_modules, gate_homotopy_scale)
            
            # Phase 2: Update precision cache before gradient step
            self._update_precision_cache(etas)
            
            grads = self._grads(etas)
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
            need_L = self.stability_guard or self.stability_coupling_auto_cap
            if need_L:
                L_est = self._estimate_lipschitz_bound(etas)
            if self.stability_guard and L_est and L_est > 0.0 and math.isfinite(L_est):
                # Optional: set step directly from Lipschitz estimate (2/L) with safety fraction
                if self.auto_step_from_lipschitz:
                    step_to_use = self.stability_cap_fraction * (2.0 / L_est)
                cap = self.stability_cap_fraction * (2.0 / L_est)
                if cap > 0.0:
                    step_to_use = min(step_to_use, cap)
                    if self.log_contraction_margin:
                        margin = (2.0 / L_est) - step_to_use
                        self._last_contraction_margin = margin
                        self._contraction_margin_history.append(float(margin))
                        # Emit warning if margin shrinks below threshold
                        if self.warn_on_margin_shrink and margin < self.margin_warn_threshold:
                            warnings.warn(
                                f"Contraction margin ({margin:.2e}) below threshold ({self.margin_warn_threshold:.2e}). "
                                f"Consider reducing step_size or coupling weights. "
                                f"Lipschitz bound L={L_est:.2e}, safe step=2/L={2.0/L_est:.2e}, current step={step_to_use:.2e}",
                                UserWarning,
                                stacklevel=2
                            )
            elif self.stability_guard and self.log_contraction_margin:
                self._last_contraction_margin = None
                self._contraction_margin_history.append(float("nan"))
            
            # Inject orthogonal noise if enabled (structure-preserving exploration)
            grad_vector = np.array(grads, dtype=float)
            noise_vector = np.zeros_like(grad_vector)
            current_noise_mag = 0.0
            if self.enable_orthogonal_noise:
                if controller is not None:
                    current_noise_mag = controller.step(
                        grad_vector,
                        energy_drop_ratio=getattr(self, "_last_energy_drop_ratio", 1.0),
                        backtracks=int(self._last_step_backtracks),
                        iter_idx=iter_idx,
                    )
            else:
                current_noise_mag = self.noise_magnitude * (self.noise_schedule_decay ** iter_idx)
            if current_noise_mag > 1e-9:
                raw_noise = np.random.normal(0, 1, size=grad_vector.shape)
                if self.metric_aware_noise_controller and (self.metric_vector_product is not None or self.metric_matrix is not None):
                    noise_vector = project_noise_metric_orthogonal(
                        raw_noise,
                        grad_vector,
                        M=self.metric_matrix,
                        Mv=self.metric_vector_product,
                    )
                else:
                    noise_vector = project_noise_orthogonal(raw_noise, grad_vector)
                # Precision-aware redistribution toward low-curvature directions
                if self.precision_aware_noise_controller:
                    try:
                        curv_diag = np.asarray(self.get_precision_diagonal(), dtype=float)
                        # If controller supports weights_for_curvatures, use it; else compute locally
                        if isinstance(self._noise_controller, PrecisionNoiseController) and hasattr(self._noise_controller, "weights_for_curvatures"):
                            weights = self._noise_controller.weights_for_curvatures(curv_diag, eps=self.precision_epsilon)  # type: ignore[attr-defined]
                        else:
                            inv = 1.0 / (float(self.precision_epsilon) + np.maximum(curv_diag, 0.0))
                            inv_norm = float(np.linalg.norm(inv))
                            weights = inv / inv_norm if inv_norm > 0.0 else inv
                        # Reweight and re-project to preserve orthogonality
                        noise_weighted = weights * noise_vector
                        if self.metric_aware_noise_controller and (self.metric_vector_product is not None or self.metric_matrix is not None):
                            noise_vector = project_noise_metric_orthogonal(
                                noise_weighted,
                                grad_vector,
                                M=self.metric_matrix,
                                Mv=self.metric_vector_product,
                            )
                        else:
                            noise_vector = project_noise_orthogonal(noise_weighted, grad_vector)
                    except Exception:
                        pass
                noise_norm = np.linalg.norm(noise_vector)
                if noise_norm > 1e-9:
                    noise_vector = noise_vector * (current_noise_mag / noise_norm)
            
            # step
            if self.line_search:
                # Line search applies to the gradient direction
                # Noise is added *after* the descent step decision (Langevin-style) or *to* the direction?
                # Standard practice: add noise to the update. 
                # But line search needs a direction. Let's define direction d = -grad + noise
                # Then line search along d.
                # Mirror/logit option: use ∂F/∂ζ = ∂F/∂η · η(1−η) to define the descent direction
                if self.use_logit_updates:
                    grads_eff = []
                    for i, g in enumerate(grads):
                        eta_i = float(max(0.0, min(1.0, etas[i])))
                        grads_eff.append(float(g) * eta_i * (1.0 - eta_i))
                else:
                    grads_eff = list(grads)
                # Precision-aware diagonal preconditioning for line-search direction
                if self.use_precision_preconditioning:
                    curv_diag = self.get_precision_diagonal()
                    eps = float(self.precision_epsilon)
                    denom = [max(eps, float(c)) for c in curv_diag]
                    grads_eff = [g / d for g, d in zip(grads_eff, denom)]

            # Coupling auto-cap
            coupling_scale = None
            if self.stability_coupling_auto_cap and L_est and L_est > 0.0 and math.isfinite(L_est):
                target = self.stability_coupling_target
                if target is None:
                    target = L_est
                if target > 0.0 and L_est > 0.0:
                    coupling_scale = min(1.0, max(0.0, target / L_est))
                    if coupling_scale >= 0.999:
                        coupling_scale = None
            self._stability_coupling_scale = coupling_scale
            # Optional: prepare Lipschitz details for allocator/telemetry
            self._last_lipschitz_details = None
            need_details = (
                self.expose_lipschitz_details
                or (self.weight_adapter is not None and any(
                    hasattr(self.weight_adapter, attr) for attr in ("edge_costs", "row_margins", "global_margin")
                ))
            )
            if need_details:
                target_L = self.stability_coupling_target if (self.stability_coupling_target and self.stability_coupling_target > 0.0) else L_est
                self._last_lipschitz_details = self._estimate_lipschitz_details(
                    etas, smoothing_epsilon=max(self.grad_eps * 0.5, 1e-6), target_L=target_L
                )
            # step
            if self.line_search:
                # Standard Armijo along -grad direction
                # Apply line search on grads_eff (mirror-aware if enabled), then add noise
                etas = self._step_with_backtracking(etas, grads_eff, step_to_use)
                if self.enable_orthogonal_noise and np.any(noise_vector):
                    # Add noise (orthogonal to gradient, so doesn't fight the descent step to first order)
                    for i in range(len(etas)):
                        etas[i] = float(max(0.0, min(1.0, etas[i] + noise_vector[i])))
            else:
                # No line search: direct gradient update with noise blended in
                for i in range(len(etas)):
                    if self.use_logit_updates and not self.line_search:
                        # Mirror/logit update in ζ-space with η = σ(ζ); dF/dζ = dF/dη * η(1-η)
                        eta_i = float(max(0.0, min(1.0, etas[i])))
                        eps = float(max(self.logit_epsilon, 1e-12))
                        # Compute current logit
                        num = eta_i + eps
                        den = (1.0 - eta_i) + eps
                        z = math.log(num) - math.log(den)
                        dF_dz = float(grads[i]) * eta_i * (1.0 - eta_i)
                        z_new = z - step_to_use * dF_dz
                        # Map back: σ(z) = 1 / (1 + exp(-z))
                        if z_new >= 0.0:
                            ez = math.exp(-z_new)
                            eta_new = 1.0 / (1.0 + ez)
                        else:
                            ez = math.exp(z_new)
                            eta_new = ez / (1.0 + ez)
                        # Add orthogonal noise in η-space if enabled
                        if self.enable_orthogonal_noise:
                            eta_new = eta_new + float(noise_vector[i])
                        etas[i] = float(max(0.0, min(1.0, eta_new)))
                    else:
                        # Update: eta - step*grad + noise
                        # noise_vector is already scaled by noise_magnitude
                        # Precision-aware diagonal preconditioning if enabled
                        if self.use_precision_preconditioning:
                            curv_i = float(self._precision_cache.get(i, 0.0)) if self._precision_cache is not None else 0.0  # type: ignore[union-attr]
                            denom_i = float(self.precision_epsilon) + max(0.0, curv_i)
                            g_eff = float(grads[i]) / (denom_i if denom_i > 0.0 else 1.0)
                        else:
                            g_eff = float(grads[i])
                        update = -step_to_use * g_eff
                        if self.enable_orthogonal_noise:
                            update += noise_vector[i]
                        etas[i] = float(max(0.0, min(1.0, etas[i] + update)))
            
            self._emit_eta(etas)
            energy_value = self._energy_value(etas)
            if self.adaptive_coordinate_descent and prev_energy_value is not None:
                drop = prev_energy_value - energy_value
                if drop < self.adaptive_switch_delta:
                    stalled_steps += 1
                else:
                    stalled_steps = 0
                if stalled_steps >= self.adaptive_switch_patience:
                    etas = self.relax_etas_coordinate(
                        etas,
                        steps=self.coordinate_steps,
                        active_tol=self.coordinate_active_tol,
                    )
                    self._adaptive_switches += 1
                    stalled_steps = 0
                    prev_energy_value = self._energy_value(etas)
                    continue
                if self.enforce_invariants:
                    self._check_invariants(etas, energy_value)
            if prev_energy_value is not None:
                drop = max(prev_energy_value - energy_value, 0.0)
                denom = max(abs(prev_energy_value), 1e-12)
                self._last_energy_drop_ratio = drop / denom
            else:
                self._last_energy_drop_ratio = 1.0
            # Optional strict monotonicity assertion (for debugging/validation in deterministic mode)
            if (
                self.assert_monotonic_energy
                and self.noise_magnitude <= 1e-12
                and not self.line_search
                and self.weight_adapter is None
                and not homotopy_active
                and prev_energy_value is not None
            ):
                assert energy_value <= prev_energy_value + self.monotonic_energy_tol, (
                    f"Energy increased: {prev_energy_value:.12e} → {energy_value:.12e} "
                    f"(Δ={energy_value - prev_energy_value:.3e}). This indicates a gradient bug, "
                    f"numerical instability, or misconfigured coupling. Disable assert_monotonic_energy "
                    f"if using exploration noise, line search, homotopy/weight-adaptation, or exotic schedules."
                )
            # Early stop on non-monotonic energy (guard against oscillations).
            # We reject steps that don't improve our objective to prevent instability.
            should_reject = False

            # Option A: Free-energy guard (Phase 2)
            # Accept based on F = U - T*S decrease instead of U alone.
            if self.use_free_energy_guard and prev_energy_value is not None:
                # Compute free energy for current and previous states
                prev_free_energy = self._compute_free_energy(etas_prev)
                curr_free_energy = self._compute_free_energy(etas)
                delta_F = curr_free_energy - prev_free_energy
                
                # Accept only if ΔF < -ε (sufficient decrease in free energy)
                if delta_F >= -self.free_energy_epsilon:
                    should_reject = True
                    self._last_acceptance_reason = "free_energy_insufficient_decrease"

            # Option B: Standard energy guard (Phase 1)
            # Strictly enforce monotonic decrease in total energy U.
            elif prev_energy_value is not None and energy_value > prev_energy_value + 1e-12:
                should_reject = True
            
            if should_reject:
                # Homotopy guard: back off start scale and retry this iteration
                if homotopy_active and self.enable_homotopy_guards and self._homotopy_runtime_start is not None:
                    oscillations += 1
                    if oscillations >= int(max(1, self.homotopy_oscillation_patience)):
                        new_start = max(
                            float(self.homotopy_min_start_scale),
                            float(self._homotopy_runtime_start) * float(self.homotopy_backoff_factor),
                        )
                        if new_start < float(self._homotopy_runtime_start):
                            self._homotopy_runtime_start = new_start
                            self._homotopy_backoffs += 1
                        oscillations = 0
                    # Revert etas and continue (skip accept)
                    etas = etas_prev
                    energy_value = prev_energy_value
                    if not self._last_acceptance_reason:
                        self._last_acceptance_reason = "homotopy_oscillation_rejected"
                    continue
                else:
                    if not self._last_acceptance_reason:
                        self._last_acceptance_reason = "non_monotonic_rejected"
                    break
            # Emit only after acceptance
            # Record acceptance reason for standard/coordinate steps
            if not self.line_search and not self._last_acceptance_reason:
                if self.use_free_energy_guard:
                    self._last_acceptance_reason = "free_energy_accepted"
                elif prev_energy_value is not None and energy_value <= prev_energy_value:
                    self._last_acceptance_reason = "monotone_decrease"
                else:
                    self._last_acceptance_reason = "initial_step"
            # Update sensitivity-probe metrics before logging energy so dashboards can show it in the same row
            if self.enable_sensitivity_probes:
                self._update_probe_metrics(etas)
            # Escape event detection (noise-triggered basin transition heuristic)
            if self.enable_escape_event_logging:
                try:
                    prev_arr = np.asarray(etas_prev, dtype=float)
                    curr_arr = np.asarray(etas, dtype=float)
                    disp = curr_arr - prev_arr
                    disp_norm = float(np.linalg.norm(disp))
                    grad_vec = np.asarray(grads, dtype=float)
                    grad_norm = float(np.linalg.norm(grad_vec))
                    align_cos = 1.0
                    if disp_norm > 0.0 and grad_norm > 0.0:
                        align_cos = float(np.dot(disp, -grad_vec) / (disp_norm * grad_norm))
                    energy_drop = 0.0 if prev_energy_value is None else float(prev_energy_value - energy_value)
                    # Heuristic: large displacement, poorly aligned with gradient (noise-driven), sufficient drop, and non-trivial noise used
                    noise_ok = (current_noise_mag if 'current_noise_mag' in locals() else 0.0) >= self.escape_noise_min_magnitude
                    is_escape = (
                        noise_ok and
                        disp_norm >= self.escape_displacement_min_norm and
                        align_cos <= self.escape_alignment_max_cosine and
                        energy_drop >= self.escape_min_energy_drop
                    )
                    if is_escape:
                        self._escape_events_count += 1
                        self._last_acceptance_reason = "escape_event"
                except Exception:
                    pass
            # Confidence fusion: c = sigmoid(a*(rho_max - residual) - b*sensitivity)
            if self.enable_confidence_logging:
                try:
                    # Residual proxy: normalized energy relative to initial
                    E0 = float(self._initial_energy_value if self._initial_energy_value is not None else energy_value)
                    denom = max(abs(E0), 1e-12)
                    residual = float(max(0.0, energy_value / denom))
                    # Sensitivity proxy: last probe dispersion if available
                    sens = float(self.last_probe_dispersion() or 0.0) if hasattr(self, "last_probe_dispersion") else 0.0
                    # Redundancy rho: optional from constraints, else 0.0
                    rho = float(self.constraints.get("redundancy_rho", 0.0)) if isinstance(self.constraints, dict) else 0.0
                    a = float(self.confidence_a)
                    b = float(self.confidence_b)
                    rho_max = float(self.confidence_rho_max)
                    z = a * (rho_max - residual + rho) - b * sens
                    # Sigmoid
                    c = 1.0 / (1.0 + math.exp(-float(z)))
                    c = float(max(0.0, min(1.0, c)))
                    self._confidence_history.append(c)
                except Exception:
                    pass
            self._emit_energy(energy_value)
            self._record_energy_history(energy_value)
            
            # Early-stop with patience: stop if energy stabilizes
            if self.enable_early_stop and prev_energy_value is not None:
                delta_E = abs(prev_energy_value - energy_value)
                if delta_E < self.early_stop_delta_threshold:
                    self._early_stop_stable_count += 1
                    if self._early_stop_stable_count >= self.early_stop_patience:
                        self._last_acceptance_reason = "early_stop_converged"
                        break
                else:
                    self._early_stop_stable_count = 0
            
            if self.enable_uncertainty_gate and gate_modules:
                self._update_uncertainty_gate_scale()
            prev_energy_value = energy_value
            term_norms = self._term_grad_norms(etas)
            if self.auto_balance_term_weights:
                self._auto_balance_term_weights(term_norms)
            if self.weight_adapter is not None:
                # If adapter supports allocator fields, inject details snapshot
                if self._last_lipschitz_details is not None:
                    if hasattr(self.weight_adapter, "edge_costs"):
                        # Prefer true edge_costs when available; fall back to family_costs
                        edge_costs = self._last_lipschitz_details.get("edge_costs") or self._last_lipschitz_details.get("family_costs", {})
                        try:
                            # type: ignore[attr-defined]
                            self.weight_adapter.edge_costs = {str(k): float(v) for k, v in edge_costs.items()}
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
        self._homotopy_scale = None
        self._homotopy_term_scales = None
        if self._homotopy_gate_bases is not None:
            for m, base in zip(gate_modules, self._homotopy_gate_bases):
                m.cost = float(base)
        self._homotopy_gate_bases = None
        self._gate_uncertainty_scale = 1.0
        self._accepted_energy_history = []
        self._stability_coupling_scale = None
        return etas

    # --- Proximal / operator-splitting mode ---
    def relax_etas_proximal(self, etas0: List[OrderParameter], steps: int = 50, tau: float = 0.05) -> List[OrderParameter]:
        """Block-coordinate proximal updates on locals + incident couplings.

        Strategy (conservative):
          - For each iteration:
            1) Local proximal gradient for every module (project to [0,1]).
            2) Pairwise prox on couplings (closed-form for quadratic/hinge family; projected if needed).
          - Emit accepted energies only; stop if energy increases.
        """
        if self.prox_block_mode == "star":
            return self._relax_etas_proximal_star(etas0, steps=steps, tau=tau)
        assert tau > 0.0, "prox tau must be positive"
        etas = [float(e) for e in etas0]
        prev_energy = self._energy_value(etas)
        self._emit_eta(etas)
        self._emit_energy(prev_energy)
        for _ in range(steps):
            # 1) Local proximal gradient step
            for idx, m in enumerate(self.modules):
                # gradient of local (analytic if available)
                if isinstance(m, SupportsLocalEnergyGrad):
                    g = float(m.d_local_energy_d_eta(etas[idx], self.constraints))
                else:
                    base = float(m.local_energy(etas[idx], self.constraints))
                    bumped = float(m.local_energy(min(1.0, etas[idx] + self.grad_eps), self.constraints))
                    g = (bumped - base) / self.grad_eps
                # apply term weight
                w = float(self._combined_term_weights().get(f"local:{m.__class__.__name__}", 1.0))
                # proximal gradient with projection to [0,1]
                etas[idx] = float(max(0.0, min(1.0, etas[idx] - tau * w * g)))
            # 2) Pairwise coupling prox/updates
            for i, j, coup in self.couplings:
                key = f"coup:{coup.__class__.__name__}"
                w_c = float(self._combined_term_weights().get(key, 1.0))
                if isinstance(coup, QuadraticCoupling):
                    etas[i], etas[j] = prox_quadratic_pair(etas[i], etas[j], coup.weight * w_c, tau)
                elif isinstance(coup, DirectedHingeCoupling):
                    etas[i], etas[j] = prox_asym_hinge_pair(
                        etas[i], etas[j], weight=coup.weight * w_c, alpha=1.0, beta=1.0, tau=tau
                    )
                elif isinstance(coup, AsymmetricHingeCoupling):
                    etas[i], etas[j] = prox_asym_hinge_pair(
                        etas[i], etas[j], weight=coup.weight * w_c, alpha=coup.alpha_i, beta=coup.beta_j, tau=tau
                    )
                elif isinstance(coup, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                    # Prox for linear gate term on variable i (j unaffected)
                    gi, _ = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                    # For F = ... + gi*η_i (since gi = ∂F/∂η_i), prox uses coeff = -gi
                    coeff = -float(gi) * w_c
                    etas[i] = prox_linear_gate(etas[i], coeff, tau)
                # else: no-op for unknown types
            # Check invariants and acceptance
            if self.enforce_invariants:
                self._check_invariants(etas)
            F = self._energy_value(etas)
            if F > prev_energy + 1e-12:
                break
            self._emit_eta(etas)
            self._emit_energy(F)
            prev_energy = F
        return etas

    def _prox_quadratic_pair(self, x0: float, y0: float, weight: float, tau: float) -> tuple[float, float]:
        return prox_quadratic_pair(x0, y0, weight, tau)

    def _prox_asym_hinge_pair(self, x0: float, y0: float, weight: float, alpha: float, beta: float, tau: float) -> tuple[float, float]:
        return prox_asym_hinge_pair(x0, y0, weight, alpha, beta, tau)

    def _relax_etas_proximal_star(self, etas0: List[OrderParameter], steps: int, tau: float) -> List[OrderParameter]:
        assert tau > 0.0, "prox tau must be positive"
        etas = [float(e) for e in etas0]
        self._ensure_adjacency(len(etas))
        prev_energy = self._energy_value(etas)
        self._emit_eta(etas)
        self._emit_energy(prev_energy)
        for _ in range(steps):
            cw = self._combined_term_weights()
            for center in range(len(self.modules)):
                block_vals = self._prox_star_block(center, etas, tau, cw)
                for idx, val in block_vals.items():
                    etas[idx] = float(max(0.0, min(1.0, val)))
            if self.enforce_invariants:
                self._check_invariants(etas)
            F = self._energy_value(etas)
            if F > prev_energy + 1e-12:
                break
            self._emit_eta(etas)
            self._emit_energy(F)
            prev_energy = F
        return etas

    def _prox_star_block(self, center: int, etas: List[OrderParameter], tau: float, cw: dict[str, float]) -> dict[int, float]:
        assert self._adjacency is not None
        block: List[int] = [center]
        for neighbor, _ in self._adjacency[center]:
            if 0 <= neighbor < len(etas):
                block.append(int(neighbor))
        block = sorted(set(block))
        updated: dict[int, float] = {}
        for idx in block:
            updated[idx] = self._prox_local_single(idx, etas[idx], tau, cw)
        for i, j, coup in self.couplings:
            if i not in block or j not in block:
                continue
            key = f"coup:{coup.__class__.__name__}"
            w_c = float(cw.get(key, 1.0))
            xi = updated.get(i, etas[i])
            xj = updated.get(j, etas[j])
            if isinstance(coup, QuadraticCoupling):
                xi, xj = prox_quadratic_pair(xi, xj, coup.weight * w_c, tau)
            elif isinstance(coup, DirectedHingeCoupling):
                xi, xj = prox_asym_hinge_pair(xi, xj, coup.weight * w_c, alpha=1.0, beta=1.0, tau=tau)
            elif isinstance(coup, AsymmetricHingeCoupling):
                xi, xj = prox_asym_hinge_pair(xi, xj, coup.weight * w_c, alpha=coup.alpha_i, beta=coup.beta_j, tau=tau)
            elif isinstance(coup, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                gi, _ = coup.d_coupling_energy_d_etas(xi, xj, self.constraints)
                coeff = -float(gi) * w_c
                xi = prox_linear_gate(xi, coeff, tau)
            else:
                continue
            updated[i] = float(max(0.0, min(1.0, xi)))
            updated[j] = float(max(0.0, min(1.0, xj)))
        return updated

    def _prox_local_single(self, idx: int, eta_i: float, tau: float, cw: dict[str, float]) -> float:
        module = self.modules[idx]
        if isinstance(module, SupportsLocalEnergyGrad):
            grad = float(module.d_local_energy_d_eta(eta_i, self.constraints))
        else:
            base = float(module.local_energy(eta_i, self.constraints))
            bumped = float(module.local_energy(min(1.0, eta_i + self.grad_eps), self.constraints))
            grad = (bumped - base) / self.grad_eps
        key = f"local:{module.__class__.__name__}"
        weight = float(cw.get(key, 1.0))
        eta_new = eta_i - tau * weight * grad
        return float(max(0.0, min(1.0, eta_new)))

    # --- ADMM (experimental) for quadratic couplings ---
    def relax_etas_admm(
        self,
        etas0: List[OrderParameter],
        steps: int = 50,
        rho: float = 1.0,
        step_size: float = 0.05,
    ) -> List[OrderParameter]:
        """ADMM-like splitting primarily for quadratic couplings.

        - Introduces auxiliary variables s_k for each quadratic edge k = (i,j) to represent (η_i - η_j).
        - Alternates:
            s-update:  minimize w*s^2 + (ρ/2)*(s - (d_ij - u))^2   (closed form)
            η-update:  gradient step on locals + augmented term (ρ/2)*||s - (η_i-η_j) + u||^2
            u-update:  u ← u + s - (η_i - η_j)
        - Hinge-family couplings use s ≥ 0 on gaps g = β η_j − α η_i with the same augmented residual r = s − g + u.
        - Other couplings (e.g., gate-benefit) contribute via their gradients during η-update.
        """
        assert rho > 0.0 and step_size > 0.0
        n = len(etas0)
        etas = [float(e) for e in etas0]
        # Track quadratic couplings' indices for s/u
        quad_edges: list[tuple[int, int, float, int]] = []
        # Track hinge-family couplings (directed/asymmetric) for s/u with nonnegativity
        hinge_edges: list[tuple[int, int, float, float, float, int]] = []  # (i,j,weight,alpha,beta,edge_idx)
        cw = self._combined_term_weights()
        for idx, (i, j, coup) in enumerate(self.couplings):
            if isinstance(coup, QuadraticCoupling):
                w = float(coup.weight) * float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                quad_edges.append((i, j, w, idx))
            elif isinstance(coup, DirectedHingeCoupling):
                w = float(coup.weight) * float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                hinge_edges.append((i, j, w, 1.0, 1.0, idx))
            elif isinstance(coup, AsymmetricHingeCoupling):
                w = float(coup.weight) * float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                hinge_edges.append((i, j, w, float(coup.alpha_i), float(coup.beta_j), idx))
        m = len(quad_edges)
        s = [0.0] * m
        u = [0.0] * m
        mh = len(hinge_edges)
        sh = [0.0] * mh
        uh = [0.0] * mh
        prev_energy = self._energy_value(etas)
        self._emit_eta(etas)
        self._emit_energy(prev_energy)
        for _ in range(steps):
            # s-update (closed form): s = ρ*(d - u) / (ρ + 2w)
            for k, (i, j, w, _edge_idx) in enumerate(quad_edges):
                d_ij = float(etas[i] - etas[j])
                denom = rho + 2.0 * w
                s[k] = (rho * (d_ij - u[k])) / (denom if denom > 0.0 else rho)
            # hinge s-update with nonnegativity: sh = max(0, ρ*(gap - uh)/(ρ+2w))
            for k, (i, j, w, alpha, beta, _edge_idx) in enumerate(hinge_edges):
                gap = beta * float(etas[j]) - alpha * float(etas[i])
                denom = rho + 2.0 * w
                sh_k = (rho * (gap - uh[k])) / (denom if denom > 0.0 else rho)
                sh[k] = float(max(0.0, sh_k))
            # η-update: gradient step using locals + augmented quadratic terms (and non-quadratic couplings)
            grads = [0.0] * n
            # local grads
            for idx_m, (m_module, eta_val) in enumerate(zip(self.modules, etas)):
                w_loc = float(cw.get(f"local:{m_module.__class__.__name__}", 1.0))
                if isinstance(m_module, SupportsLocalEnergyGrad):
                    grads[idx_m] += w_loc * float(m_module.d_local_energy_d_eta(float(eta_val), self.constraints))
                else:
                    base = float(m_module.local_energy(float(eta_val), self.constraints))
                    bumped = float(m_module.local_energy(min(1.0, float(eta_val) + self.grad_eps), self.constraints))
                    grads[idx_m] += w_loc * ((bumped - base) / self.grad_eps)
            # augmented terms from quadratic edges
            for k, (i, j, _w, _edge_idx) in enumerate(quad_edges):
                r = s[k] - (float(etas[i]) - float(etas[j])) + u[k]
                grads[i] += -rho * r
                grads[j] += rho * r
            # augmented terms from hinge edges
            for k, (i, j, _w, alpha, beta, _edge_idx) in enumerate(hinge_edges):
                r = sh[k] - (beta * float(etas[j]) - alpha * float(etas[i])) + uh[k]
                # ∂r/∂η_i = +alpha, ∂r/∂η_j = -beta
                grads[i] += rho * r * alpha
                grads[j] += -rho * r * beta
            # non-quadratic couplings via gradients
            for i, j, coup in self.couplings:
                if isinstance(coup, (QuadraticCoupling, DirectedHingeCoupling, AsymmetricHingeCoupling)):
                    continue
                w_c = float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                # Optionally skip gate-benefit grads if we plan a prox-linear update instead
                if self.admm_gate_prox and isinstance(coup, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                    continue
                if isinstance(coup, SupportsCouplingGrads):
                    gi, gj = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                    grads[i] += w_c * float(gi)
                    grads[j] += w_c * float(gj)
                else:
                    base = float(coup.coupling_energy(etas[i], etas[j], self.constraints))
                    bi = float(coup.coupling_energy(min(1.0, etas[i] + self.grad_eps), etas[j], self.constraints))
                    bj = float(coup.coupling_energy(etas[i], min(1.0, etas[j] + self.grad_eps), self.constraints))
                    grads[i] += w_c * ((bi - base) / self.grad_eps)
                    grads[j] += w_c * ((bj - base) / self.grad_eps)
            # gradient step + projection
            for i in range(n):
                etas[i] = float(max(0.0, min(1.0, float(etas[i]) - step_size * float(grads[i]))))
            # optional prox-linear update for gate-benefit family on gate variable (i)
            if self.admm_gate_prox:
                damp = float(max(0.0, min(1.0, self.admm_gate_damping)))
                if damp > 0.0:
                    for i, j, coup in self.couplings:
                        if isinstance(coup, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                            w_c = float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                            # use analytic derivative if available to form linear prox coeff
                            if isinstance(coup, SupportsCouplingGrads):
                                gi, _ = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                                coeff = -float(gi) * w_c
                            else:
                                base_e = float(coup.coupling_energy(etas[i], etas[j], self.constraints))
                                bump_e = float(coup.coupling_energy(min(1.0, etas[i] + self.grad_eps), etas[j], self.constraints))
                                gi_fd = (bump_e - base_e) / self.grad_eps
                                coeff = -float(gi_fd) * w_c
                            x_old = float(etas[i])
                            x_prox = float(prox_linear_gate(x_old, coeff, step_size))
                            etas[i] = float(max(0.0, min(1.0, (1.0 - damp) * x_old + damp * x_prox)))
            # u-update
            for k, (i, j, _w, _edge_idx) in enumerate(quad_edges):
                u[k] += s[k] - (float(etas[i]) - float(etas[j]))
            for k, (i, j, _w, alpha, beta, _edge_idx) in enumerate(hinge_edges):
                uh[k] += sh[k] - (beta * float(etas[j]) - alpha * float(etas[i]))
            # acceptance guard
            if self.enforce_invariants:
                self._check_invariants(etas)
            F = self._energy_value(etas)
            if F > prev_energy + 1e-12:
                break
            self._emit_eta(etas)
            self._emit_energy(F)
            prev_energy = F
        return etas
    def _finite_diff_grads(self, etas: List[OrderParameter]) -> List[float]:
        base = self._energy_value(etas)
        grads: List[float] = [0.0 for _ in etas]
        indices: Iterable[int]
        if self.neighbor_gradients_only:
            self._ensure_adjacency(len(etas))
            indices = self._active_indices(etas)
        else:
            indices = range(len(etas))
        for i in indices:
            bumped = list(etas)
            bumped[i] += self.grad_eps
            Fb = self._energy_value(bumped)
            grad_i = (Fb - base) / self.grad_eps
            grads[i] = float(grad_i)
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
                base = float(m.local_energy(eta, self.constraints))
                b = float(m.local_energy(eta + self.grad_eps, self.constraints))
                grad_arr[idx] += w * ((b - base) / self.grad_eps)
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
                base = float(coup.coupling_energy(etas[i], etas[j], self.constraints))
                bi = float(coup.coupling_energy(etas[i] + self.grad_eps, etas[j], self.constraints))
                bj = float(coup.coupling_energy(etas[i], etas[j] + self.grad_eps, self.constraints))
                grad_arr[i] += w * ((bi - base) / self.grad_eps)
                grad_arr[j] += w * ((bj - base) / self.grad_eps)
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
        """Update the local diagonal precision/curvature cache for modules."""
        cache: Dict[int, float] = {}
        for idx, (m, eta) in enumerate(zip(self.modules, etas)):
            if isinstance(m, SupportsPrecision):
                try:
                    curv = float(m.curvature(eta))
                    # Ensure non-negative precision (convex approximation)
                    cache[idx] = max(0.0, curv)
                except Exception:
                    cache[idx] = 0.0
            else:
                cache[idx] = 0.0
        self._precision_cache = cache

    def get_precision_diagonal(self) -> List[float]:
        """Return the currently cached diagonal precision vector."""
        if self._precision_cache is None:
            return [0.0] * len(self.modules)
        return [self._precision_cache.get(i, 0.0) for i in range(len(self.modules))]
    
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
    
    def relax_etas_coordinate(
        self,
        etas0: List[OrderParameter],
        steps: int = 200,
        active_tol: float = 1e-4,
    ) -> List[OrderParameter]:
        """Coordinate descent: update the index with largest |grad| each iteration."""
        etas = [float(e) for e in etas0]
        # Build adjacency
        self._ensure_adjacency(len(etas))
        # Initialize gradients and energy once
        grads = self._grads(etas)
        F = self._energy_value(etas)
        for _ in range(steps):
            # pick active coordinate
            idx = int(np.argmax(np.abs(np.asarray(grads, dtype=float))))
            g_i = float(grads[idx])
            if abs(g_i) < active_tol:
                break
            # choose step length
            step = float(self.step_size)
            if self.normalize_grads:
                gabs = abs(g_i)
                if gabs > 0.0:
                    step = step / gabs
            eta_i_old = float(etas[idx])
            eta_i_new = float(max(0.0, min(1.0, eta_i_old - step * g_i)))
            if eta_i_new == eta_i_old:
                # nothing to update
                break
            # local gradient delta
            d_local_old = self._local_grad(idx, eta_i_old)
            d_local_new = self._local_grad(idx, eta_i_new)
            delta_gi = d_local_new - d_local_old
            # local energy delta
            f_local_old = self._local_energy(idx, eta_i_old)
            f_local_new = self._local_energy(idx, eta_i_new)
            delta_F = f_local_new - f_local_old
            # coupling deltas on neighbors
            for (j, coup) in self._adjacency[idx]:  # type: ignore[union-attr]
                eta_j = float(etas[j])
                gi_old, gj_old = self._pair_coupling_grads(coup, idx, j, eta_i_old, eta_j)
                gi_new, gj_new = self._pair_coupling_grads(coup, idx, j, eta_i_new, eta_j)
                delta_gi += (gi_new - gi_old)
                grads[j] = float(grads[j] + (gj_new - gj_old))
                # energy delta for this edge
                f_ij_old = self._pair_coupling_energy(coup, idx, j, eta_i_old, eta_j)
                f_ij_new = self._pair_coupling_energy(coup, idx, j, eta_i_new, eta_j)
                delta_F += (f_ij_new - f_ij_old)
            # commit update
            etas[idx] = eta_i_new
            grads[idx] = float(g_i + delta_gi)
            F = float(F + delta_F)
            self._emit_eta(etas)
            energy_value = self.energy(etas)
            if self.enforce_invariants:
                self._check_invariants(etas, energy_value)
        return etas

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
                base = float(m.local_energy(eta, self.constraints))
                bumped = float(m.local_energy(min(1.0, eta + self.grad_eps), self.constraints))
                grad_buf[idx] = w * ((bumped - base) / self.grad_eps)
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

    def _quadratic_energy_vectorized(self, etas: List[OrderParameter], cw: dict[str, float]) -> float:
        cache = self._vectorized_cache
        if cache is None or cache.quadratic_i.size == 0:
            return 0.0
        eta_arr = np.asarray(etas, dtype=float)
        term_weights = (
            np.asarray([float(cw.get(key, 1.0)) for key in cache.quadratic_term_keys], dtype=float)
            if cache.quadratic_term_keys
            else np.ones_like(cache.quadratic_weights)
        )
        weights = cache.quadratic_weights * term_weights
        if weights.size == 0:
            return 0.0
        diff = eta_arr[cache.quadratic_i] - eta_arr[cache.quadratic_j]
        return float(np.sum(weights * diff * diff))

    def _step_with_backtracking(self, etas: List[OrderParameter], grads: List[float], step_init: float) -> List[float]:
        F0 = self._energy_value(etas)
        step = float(step_init)
        gvec = np.asarray(grads, dtype=float)
        g2 = float(np.dot(gvec, gvec))
        local_bk = 0
        for _ in range(self.max_backtrack + 1):
            trial_arr = self._trial_array_for(etas)
            trial_arr -= step * gvec
            np.clip(trial_arr, 0.0, 1.0, out=trial_arr)
            trial = trial_arr.tolist()
            F1 = self._energy_value(trial)
            if F1 <= F0 - self.armijo_c * step * g2:
                self._last_step_backtracks = local_bk
                self._total_backtracks += local_bk
                self._last_acceptance_reason = "armijo_accepted"
                return trial
            step *= self.backtrack_factor
            local_bk += 1
        trial_arr = self._trial_array_for(etas)
        trial_arr -= step_init * gvec
        np.clip(trial_arr, 0.0, 1.0, out=trial_arr)
        self._last_step_backtracks = local_bk
        self._total_backtracks += local_bk
        self._last_acceptance_reason = "armijo_failed_fallback"
        return trial_arr.tolist()

    def _coordinate_backtracking(self, etas: List[OrderParameter], idx: int, grad_i: float, step_init: float) -> List[OrderParameter]:
        F0 = self._energy_value(etas)
        step = float(step_init)
        g2 = float(grad_i * grad_i)
        local_bk = 0
        for _ in range(self.max_backtrack + 1):
            trial_arr = self._trial_array_for(etas)
            trial_arr[idx] = float(max(0.0, min(1.0, trial_arr[idx] - step * grad_i)))
            trial = trial_arr.tolist()
            F1 = self._energy_value(trial)
            if F1 <= F0 - self.armijo_c * step * g2:
                self._last_step_backtracks = local_bk
                self._total_backtracks += local_bk
                self._last_acceptance_reason = "armijo_coord_accepted"
                return trial
            step *= self.backtrack_factor
            local_bk += 1
        trial_arr = self._trial_array_for(etas)
        trial_arr[idx] = float(max(0.0, min(1.0, trial_arr[idx] - step_init * grad_i)))
        self._last_step_backtracks = local_bk
        self._total_backtracks += local_bk
        self._last_acceptance_reason = "armijo_coord_failed_fallback"
        return trial_arr.tolist()

    def _estimate_lipschitz_bound(self, etas: List[OrderParameter]) -> float:
        """Conservative Gershgorin-style bound on gradient Lipschitz constant.

        Approximates diagonal (local curvature) via finite differences of local gradient
        and adds coupling curvature contributions for quadratic/hinge families.
        """
        n = len(etas)
        if n == 0:
            return 0.0
        diag = np.zeros(n, dtype=float)
        offsum = np.zeros(n, dtype=float)
        eps = max(self.grad_eps * 0.5, 1e-6)
        # Local curvature (finite-diff on local gradient)
        for i in range(n):
            eta_i = float(etas[i])
            g_m = self._local_grad(i, max(0.0, min(1.0, eta_i - eps)))
            g_p = self._local_grad(i, max(0.0, min(1.0, eta_i + eps)))
            curv = (g_p - g_m) / (2.0 * eps)
            if math.isfinite(curv) and curv > 0.0:
                diag[i] += float(curv)
        # Coupling curvature
        for i, j, coup in self.couplings:
            key = f"coup:{coup.__class__.__name__}"
            w_eff = float(self._combined_term_weights().get(key, 1.0))
            if isinstance(coup, QuadraticCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                diag[i] += 2.0 * w
                diag[j] += 2.0 * w
                offsum[i] += 2.0 * w
                offsum[j] += 2.0 * w
            elif isinstance(coup, DirectedHingeCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                gap = float(etas[j]) - float(etas[i])
                if gap > 0.0:
                    diag[i] += 2.0 * w
                    diag[j] += 2.0 * w
                    offsum[i] += 2.0 * w
                    offsum[j] += 2.0 * w
            elif isinstance(coup, AsymmetricHingeCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                alpha = float(getattr(coup, "alpha_i", 1.0))
                beta = float(getattr(coup, "beta_j", 1.0))
                gap = beta * float(etas[j]) - alpha * float(etas[i])
                if gap > 0.0:
                    diag[i] += 2.0 * w * (alpha * alpha)
                    diag[j] += 2.0 * w * (beta * beta)
                    offsum[i] += 2.0 * w * abs(alpha * beta)
                    offsum[j] += 2.0 * w * abs(alpha * beta)
            else:
                # GateBenefit are linear (no curvature); ignore others
                continue
        L_est = float(np.max(diag + offsum))
        if not math.isfinite(L_est) or L_est <= 0.0:
            return 0.0
        return L_est

    def _estimate_lipschitz_details(
        self,
        etas: List[OrderParameter],
        smoothing_epsilon: float = 1e-3,
        target_L: Optional[float] = None,
    ) -> dict:
        """Return detailed Gershgorin-like bound components and family costs.

        Produces:
          - L_est: current Lipschitz estimate (float)
          - row_sums: dict[row_index -> row_sum]
          - row_targets: dict[row_index -> target_row_sum] (proportional scaling if target_L < L_est)
          - row_margins: dict[row_index -> max(0, target - current)]
          - global_margin: max(0, target_L - L_est)
          - family_costs: dict['coup:ClassName' -> ΔL per unit relative scaling (max over rows)]

        Notes:
          - Hinge contributions near activation are smoothed with a simple linear ramp in [-ε, 0].
          - Family cost aggregates the maximum row contribution attributable to a family; this
            approximates impact on the max row sum that defines L_est.
        """
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
        diag = np.zeros(n, dtype=float)
        offsum = np.zeros(n, dtype=float)
        # Per-row, per-family contributions to row sum
        per_row_family = {}  # row -> {family_key: contrib}
        for r in range(n):
            per_row_family[r] = {}

        def _add_row_family(row: int, fam: str, amount: float) -> None:
            if amount == 0.0:
                return
            d = per_row_family[row]
            d[fam] = float(d.get(fam, 0.0) + amount)

        # Track per-edge costs (index-based), using the same smoothed contributions
        edge_costs: dict[int, float] = {}

        for edge_idx, (i, j, coup) in enumerate(self.couplings):
            key = f"coup:{coup.__class__.__name__}"
            w_eff = float(self._combined_term_weights().get(key, 1.0))
            if isinstance(coup, QuadraticCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                add = 2.0 * w  # diag and off-diag magnitude per row
                if add != 0.0:
                    diag[i] += add
                    diag[j] += add
                    offsum[i] += add
                    offsum[j] += add
                    _add_row_family(i, key, add + add)  # diag+offsum contribution to row i
                    _add_row_family(j, key, add + add)  # row j
                    # Per-edge cost: max row contribution from this edge
                    edge_costs[edge_idx] = float((add + add))
            elif isinstance(coup, DirectedHingeCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                gap = float(etas[j]) - float(etas[i])
                # Smoothed activity in [-ε, 0] → [0,1]
                if gap > 0.0:
                    s = 1.0
                elif -smoothing_epsilon < gap <= 0.0:
                    s = (gap + smoothing_epsilon) / smoothing_epsilon
                else:
                    s = 0.0
                if s > 0.0 and w != 0.0:
                    add = 2.0 * w * s
                    diag[i] += add
                    diag[j] += add
                    offsum[i] += add
                    offsum[j] += add
                    _add_row_family(i, key, add + add)
                    _add_row_family(j, key, add + add)
                    edge_costs[edge_idx] = float((add + add))
            elif isinstance(coup, AsymmetricHingeCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                alpha = float(getattr(coup, "alpha_i", 1.0))
                beta = float(getattr(coup, "beta_j", 1.0))
                gap = beta * float(etas[j]) - alpha * float(etas[i])
                if gap > 0.0:
                    s = 1.0
                elif -smoothing_epsilon < gap <= 0.0:
                    s = (gap + smoothing_epsilon) / smoothing_epsilon
                else:
                    s = 0.0
                if s > 0.0 and w != 0.0:
                    add_i = 2.0 * w * (alpha * alpha) * s
                    add_j = 2.0 * w * (beta * beta) * s
                    add_off = 2.0 * w * abs(alpha * beta) * s
                    diag[i] += add_i
                    diag[j] += add_j
                    offsum[i] += add_off
                    offsum[j] += add_off
                    _add_row_family(i, key, add_i + add_off)
                    _add_row_family(j, key, add_j + add_off)
                    edge_costs[edge_idx] = float(max(add_i + add_off, add_j + add_off))
            else:
                # GateBenefit (linear) and others: no curvature
                continue

        row_sums = (diag + offsum)
        L_est = float(np.max(row_sums)) if row_sums.size > 0 else 0.0
        # Targets and margins
        if not target_L or not math.isfinite(target_L) or target_L <= 0.0:
            target_L = L_est
        row_targets = {}
        row_margins = {}
        if L_est > 0.0 and target_L < L_est:
            scale = target_L / L_est
            for r in range(n):
                target_row = float(row_sums[r] * scale)
                row_targets[r] = target_row
                row_margins[r] = max(0.0, target_row - float(row_sums[r]))
        else:
            for r in range(n):
                row_targets[r] = float(row_sums[r])
                row_margins[r] = 0.0
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
        }

    def _apply_gate_costs(self, gate_modules: List[EnergyModule], homotopy_scale: float) -> None:
        if not gate_modules or not self._homotopy_gate_bases:
            return
        base_costs = self._homotopy_gate_bases
        homotopy_scale = max(homotopy_scale, 0.0)
        uncertainty_scale = max(self._gate_uncertainty_scale, 0.0)
        total_scale = homotopy_scale * uncertainty_scale
        if total_scale <= 0.0:
            total_scale = 0.0
        floor = max(self.gate_cost_floor, 0.0)
        ceiling = self.gate_cost_ceiling
        for module, base in zip(gate_modules, base_costs):
            cost = float(base) * total_scale if total_scale > 0.0 else 0.0
            cost = max(cost, floor)
            if ceiling is not None:
                cost = min(cost, ceiling)
            try:
                module.cost = float(cost)
            except Exception:
                continue

    def _record_energy_history(self, energy: float) -> None:
        self._accepted_energy_history.append(float(energy))
        if len(self._accepted_energy_history) > 256:
            self._accepted_energy_history = self._accepted_energy_history[-256:]

    def last_relaxation_metrics(self) -> Mapping[str, Any]:
        """Expose basic observability for the most recent relaxation run."""
        history = list(self._accepted_energy_history)
        return {
            "accepted_steps": len(history),
            "energy_trace": history,
            "last_energy_drop_ratio": float(self._last_energy_drop_ratio),
            "last_contraction_margin": self._last_contraction_margin,
            "contraction_margins": list(self._contraction_margin_history),
        }

    def _update_uncertainty_gate_scale(self) -> None:
        if not self._accepted_energy_history:
            return
        metrics = compute_agm_phase_metrics(self._accepted_energy_history)
        summary = compute_uncertainty_metrics(self._accepted_energy_history)
        target_scale = 1.0
        if (
            metrics["rate"] >= self.gate_rate_exploit_threshold
            and summary.total <= self.gate_uncertainty_relax_threshold
        ):
            target_scale = self.gate_cost_relax_scale
        elif (
            metrics["rate"] <= self.gate_rate_explore_threshold
            or summary.total >= self.gate_uncertainty_tight_threshold
        ):
            target_scale = self.gate_cost_tighten_scale
        smoothing = min(max(self.gate_cost_smoothing, 0.0), 1.0)
        self._gate_uncertainty_scale = (
            (1.0 - smoothing) * self._gate_uncertainty_scale + smoothing * target_scale
        )
        if self._gate_uncertainty_scale < 0.0:
            self._gate_uncertainty_scale = 0.0

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

    def _active_indices(self, etas: List[OrderParameter]) -> Iterable[int]:
        """Return indices participating in any coupling (plus their neighbors)."""
        if self._adjacency is None:
            return range(len(etas))
        active: set[int] = set()
        for idx, neighbors in enumerate(self._adjacency):
            if neighbors:
                active.add(idx)
            for j, _coup in neighbors:
                if 0 <= j < len(etas):
                    active.add(j)
        if not active:
            return range(len(etas))
        return tuple(sorted(active))

    def _local_energy(self, idx: int, eta_i: float) -> float:
        m = self.modules[idx]
        w = float(self._combined_term_weights().get(f"local:{m.__class__.__name__}", 1.0))
        return float(w * m.local_energy(float(eta_i), self.constraints))

    def _local_grad(self, idx: int, eta_i: float) -> float:
        m = self.modules[idx]
        w = float(self._combined_term_weights().get(f"local:{m.__class__.__name__}", 1.0))
        if isinstance(m, SupportsLocalEnergyGrad):
            return float(w * m.d_local_energy_d_eta(float(eta_i), self.constraints))
        base = float(m.local_energy(eta_i, self.constraints))
        b = float(m.local_energy(min(1.0, eta_i + self.grad_eps), self.constraints))
        return float(w * ((b - base) / self.grad_eps))

    def _pair_coupling_energy(self, coup: EnergyCoupling, i: int, j: int, eta_i: float, eta_j: float) -> float:
        w = float(self._combined_term_weights().get(f"coup:{coup.__class__.__name__}", 1.0))
        return float(w * coup.coupling_energy(float(eta_i), float(eta_j), self.constraints))

    def _pair_coupling_grads(
        self,
        coup: EnergyCoupling,
        i: int,
        j: int,
        eta_i: float,
        eta_j: float,
    ) -> Tuple[float, float]:
        if isinstance(coup, SupportsCouplingGrads):
            gi, gj = coup.d_coupling_energy_d_etas(float(eta_i), float(eta_j), self.constraints)
            w = float(self._combined_term_weights().get(f"coup:{coup.__class__.__name__}", 1.0))
            return float(w * gi), float(w * gj)
        base = float(coup.coupling_energy(eta_i, eta_j, self.constraints))
        bi = float(coup.coupling_energy(min(1.0, eta_i + self.grad_eps), eta_j, self.constraints))
        bj = float(coup.coupling_energy(eta_i, min(1.0, eta_j + self.grad_eps), self.constraints))
        w = float(self._combined_term_weights().get(f"coup:{coup.__class__.__name__}", 1.0))
        gi = (bi - base) / self.grad_eps
        gj = (bj - base) / self.grad_eps
        return float(w * gi), float(w * gj)

    def _combined_term_weights(self) -> dict[str, float]:
        base_tw: dict[str, float] = {}
        tw = self.constraints.get("term_weights", None)
        if isinstance(tw, dict):
            for k, v in tw.items():
                try:
                    base_tw[str(k)] = float(v)  # type: ignore[arg-type]
                except Exception:
                    continue
        if self._term_weights:
            base_tw.update({str(k): float(v) for k, v in self._term_weights.items()})
        homotopy_scale = getattr(self, "_homotopy_scale", None)
        term_scales = getattr(self, "_homotopy_term_scales", None)
        coupling_scale = getattr(self, "_stability_coupling_scale", None)
        floor = float(self.term_weight_floor)
        ceiling = None if self.term_weight_ceiling is None else float(self.term_weight_ceiling)
        if floor < 0.0:
            raise ValueError("term_weight_floor must be >= 0")
        if ceiling is not None and ceiling < floor:
            raise ValueError("term_weight_ceiling must be >= floor")
        calibrated: dict[str, float] = {}
        for key, value in base_tw.items():
            v = float(value)
            if homotopy_scale is not None and key.startswith("coup:"):
                v *= homotopy_scale
            if term_scales and key in term_scales:
                v *= float(term_scales[key])
            if coupling_scale is not None and key.startswith("coup:"):
                v *= coupling_scale
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
                base = float(coup.coupling_energy(etas[i], etas[j], self.constraints))
                bi = float(coup.coupling_energy(etas[i] + self.grad_eps, etas[j], self.constraints))
                bj = float(coup.coupling_energy(etas[i], etas[j] + self.grad_eps, self.constraints))
                gi = w * ((bi - base) / self.grad_eps)
                gj = w * ((bj - base) / self.grad_eps)
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
        assert 0.0 < self.armijo_c < 1.0, "armijo_c must be between 0 and 1"
        assert 0.0 < self.backtrack_factor < 1.0, "backtrack_factor must be in (0,1)"
        assert self.max_backtrack >= 0, "max_backtrack must be non-negative"
        if self.term_weight_ceiling is not None:
            assert self.term_weight_ceiling >= self.term_weight_floor >= 0.0
        if self.adaptive_coordinate_descent:
            assert self.adaptive_switch_delta > 0.0, "adaptive_switch_delta must be > 0 when adaptive coordinate descent is enabled"
            assert self.adaptive_switch_patience >= 1, "adaptive_switch_patience must be >= 1"
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

    def _update_probe_metrics(self, etas: List[OrderParameter]) -> None:
        """Update sensitivity-probe dispersion metrics from recent η trajectory.

        Dispersion is computed as the mean standard deviation across η dimensions
        over a sliding window of recent accepted states.
        """
        try:
            self._probe_trajectory_window.append([float(e) for e in etas])
            # Keep sliding window bounded
            win = int(max(1, self.sensitivity_probe_window))
            if len(self._probe_trajectory_window) > win:
                self._probe_trajectory_window = self._probe_trajectory_window[-win:]
            # Need at least 2 points for a std estimate
            if len(self._probe_trajectory_window) < 2:
                self._probe_dispersion_history.append(0.0)
                return
            import numpy as _np  # local import to avoid polluting module namespace
            window = _np.asarray(self._probe_trajectory_window, dtype=float)
            stds = _np.std(window, axis=0)
            dispersion = float(_np.mean(stds))
            self._probe_dispersion_history.append(dispersion)
        except Exception:
            # Best-effort; ignore probe metric errors
            pass

    def get_probe_dispersion_history(self) -> List[float]:
        """Return the history of probe dispersion values (may be empty)."""
        return list(self._probe_dispersion_history)

    def last_probe_dispersion(self) -> Optional[float]:
        """Return the most recent probe dispersion value, if available."""
        if not self._probe_dispersion_history:
            return None
        return float(self._probe_dispersion_history[-1])

    def get_escape_event_count(self) -> int:
        """Return the number of escape events detected in the current relaxation run."""
        return int(self._escape_events_count)

    def last_confidence(self) -> Optional[float]:
        """Return the most recent confidence value, if available."""
        if not self._confidence_history:
            return None
        return float(self._confidence_history[-1])

    def get_confidence_history(self) -> List[float]:
        """Return the history of confidence values (may be empty)."""
        return list(self._confidence_history)


