from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from core.agm_metrics import compute_agm_phase_metrics, compute_uncertainty_metrics

from core.coordinator import EnergyCoordinator
from cf_logging.metrics_log import log_records
from core.info_metrics import InformationMetrics


@dataclass
class RelaxationTracker:
    """Attach to EnergyCoordinator callbacks and log energy/η traces to Polars CSV.

    Usage:
        tracker = RelaxationTracker(name="relaxation_trace", run_id="demo")
        tracker.attach(coord)
        coord.relax_etas(...)
        tracker.flush()
    """
    name: str
    run_id: str
    buffer: List[Dict[str, Any]] = field(default_factory=list)
    step: int = 0
    prev_energy: Optional[float] = None
    last_etas: Optional[List[float]] = None
    last_timestamp: Optional[float] = None
    log_per_eta: bool = False

    def attach(self, coord: EnergyCoordinator) -> None:
        coord.on_eta_updated.append(self.on_eta)
        coord.on_energy_updated.append(self.on_energy)
        self.last_timestamp = time.perf_counter()

    def on_eta(self, etas: List[float]) -> None:
        self.last_etas = [float(e) for e in etas]

    def on_energy(self, energy: float) -> None:
        self.step += 1
        delta = None if self.prev_energy is None else float(energy - self.prev_energy)
        self.prev_energy = float(energy)
        now = time.perf_counter()
        compute_cost = None
        if self.last_timestamp is not None:
            compute_cost = float(now - self.last_timestamp)
        self.last_timestamp = now
        redemption_gain = float("nan")
        if compute_cost is not None and compute_cost > 0.0 and delta is not None and delta < 0.0:
            redemption_gain = (-delta) / compute_cost
        if self.last_etas:
            min_eta = float(min(self.last_etas))
            max_eta = float(max(self.last_etas))
            mean_eta = float(sum(self.last_etas) / float(len(self.last_etas)))
        else:
            min_eta = float("nan")
            max_eta = float("nan")
            mean_eta = float("nan")
        row: Dict[str, Any] = {
            "run_id": self.run_id,
            "step": int(self.step),
            "energy": float(energy),
            "delta_energy": float("nan") if delta is None else float(delta),
            "compute_cost": float("nan") if compute_cost is None else float(compute_cost),
            "redemption_gain": float(redemption_gain),
            "min_eta": min_eta,
            "max_eta": max_eta,
            "mean_eta": mean_eta,
        }
        if self.log_per_eta and self.last_etas:
            for idx, val in enumerate(self.last_etas):
                row[f"eta:{idx}"] = float(val)
        self.buffer.append(row)

    def flush(self) -> None:
        if self.buffer:
            log_records(self.name, self.buffer)
            self.buffer.clear()


@dataclass
class EnergyBudgetTracker:
    """Per-step energy budget and gradient norms with optional stability metrics."""

    name: str = "energy_budget"
    run_id: str = "default"
    buffer: List[Dict[str, Any]] = field(default_factory=list)
    coord: Optional[EnergyCoordinator] = None
    last_etas: Optional[List[float]] = None
    _energy_history: List[float] = field(default_factory=list, init=False, repr=False)
    _monotonicity_violations: int = field(default=0, init=False, repr=False)
    _last_timestamp: Optional[float] = field(default=None, init=False, repr=False)
    # Optional polynomial basis monitor
    _poly_history: Dict[str, List[List[float]]] = field(default_factory=dict, init=False, repr=False)
    poly_history_window: int = 64
    poly_corr_warn_threshold: float = 0.9
    # Margin warning
    warn_on_margin_shrink: bool = False
    margin_warn_threshold: float = 1e-6
    # Precision/stiffness per-η logging
    log_per_eta_precision: bool = False
    # Free energy decomposition F = U - T*S
    log_free_energy_decomposition: bool = False
    temperature: float = 1.0  # Temperature for F = U - T*S

    def attach(self, coord: EnergyCoordinator) -> None:
        self.coord = coord
        coord.on_eta_updated.append(self.on_eta)
        coord.on_energy_updated.append(self.on_energy)
        self._last_timestamp = time.perf_counter()

    def on_eta(self, etas: List[float]) -> None:
        self.last_etas = [float(e) for e in etas]

    def on_energy(self, energy: float) -> None:
        if self.coord is None or self.last_etas is None:
            return
        coord = self.coord
        etas = self.last_etas
        row: Dict[str, Any] = {"run_id": self.run_id, "energy": float(energy)}
        now = time.perf_counter()
        compute_cost = None
        if self._last_timestamp is not None:
            compute_cost = float(now - self._last_timestamp)
        self._last_timestamp = now
        # Track energy history and monotonicity flag (soft logging; not an assert)
        prev = self._energy_history[-1] if self._energy_history else None
        self._energy_history.append(float(energy))
        # Keep a modest history window
        if len(self._energy_history) > 256:
            self._energy_history = self._energy_history[-256:]
        redemption_gain = float("nan")
        if prev is not None and compute_cost is not None and compute_cost > 0.0:
            drop = float(prev) - float(energy)
            if drop > 0.0:
                redemption_gain = drop / compute_cost
        if prev is not None and float(energy) > float(prev) + 1e-12:
            self._monotonicity_violations += 1
            row["monotonicity_violation"] = 1
            row["monotonicity_violation_count"] = int(self._monotonicity_violations)
        else:
            row["monotonicity_violation"] = 0
            row["monotonicity_violation_count"] = int(self._monotonicity_violations)
        row["compute_cost"] = float("nan") if compute_cost is None else float(compute_cost)
        row["redemption_gain"] = float(redemption_gain)
        # Per-term energies
        cw = coord._combined_term_weights()  # instrumentation
        for idx, m in enumerate(coord.modules):
            key = f"local:{m.__class__.__name__}"
            w = float(cw.get(key, 1.0))
            e = float(m.local_energy(float(etas[idx]), coord.constraints)) * w
            row[f"energy:{key}"] = e
        for i, j, coup in coord.couplings:
            key = f"coup:{coup.__class__.__name__}"
            w = float(cw.get(key, 1.0))
            e = float(coup.coupling_energy(float(etas[i]), float(etas[j]), coord.constraints)) * w
            row[f"energy:{key}"] = float(row.get(f"energy:{key}", 0.0) + e)
        # Gradient norms per term
        norms = coord._term_grad_norms(etas)  # instrumentation
        for k, v in norms.items():
            row[f"grad_norm:{k}"] = float(v)
        
        # Free energy decomposition: F = U - T*S
        if self.log_free_energy_decomposition:
            # U (Internal energy) is the total energy
            U = float(energy)
            # S (Entropy): Shannon-like entropy for order parameters in [0,1]
            # S = -Σ[η*log(η) + (1-η)*log(1-η)] with safe guards for 0 and 1
            S = 0.0
            for eta_val in etas:
                eta_f = float(max(1e-9, min(1.0 - 1e-9, eta_val)))  # Clamp away from boundaries
                s_i = -(eta_f * float(__import__('math').log(eta_f)) + (1.0 - eta_f) * float(__import__('math').log(1.0 - eta_f)))
                S += s_i
            # F = U - T*S
            F = U - float(self.temperature) * S
            row["U_internal_energy"] = float(U)
            row["S_entropy"] = float(S)
            row["F_free_energy"] = float(F)
            row["T_temperature"] = float(self.temperature)
        
        # Precision (stiffness) summary stats, if available
        try:
            if hasattr(coord, "get_precision_diagonal"):
                diag = list(getattr(coord, "get_precision_diagonal")())
                if isinstance(diag, list) and len(diag) > 0:
                    vals = [float(max(0.0, v)) for v in diag]
                    row["precision:min"] = float(min(vals))
                    row["precision:max"] = float(max(vals))
                    row["precision:mean"] = float(sum(vals) / float(len(vals)))
                    # Per-η precision logging if requested
                    if self.log_per_eta_precision:
                        for idx, prec_val in enumerate(vals):
                            row[f"precision:{idx}"] = float(prec_val)
        except Exception:
            # best-effort; ignore precision logging errors
            pass
        # Sensitivity probes: dispersion measure (if available)
        try:
            if hasattr(coord, "last_probe_dispersion"):
                val = getattr(coord, "last_probe_dispersion")()
                if val is not None:
                    row["sensitivity:dispersion"] = float(val)
        except Exception:
            pass
        # Information structure metrics (if reference provided)
        try:
            # Alignment and drift relative to reference etas
            ref_etas = None
            if isinstance(coord.constraints, dict):
                ref_etas = coord.constraints.get("reference_etas", None)
            if isinstance(ref_etas, list) and len(ref_etas) == len(etas):
                align = InformationMetrics.compute_alignment(etas, ref_etas)
                drift = InformationMetrics.compute_drift(etas, ref_etas)
                row["info:alignment"] = float(align)
                row["info:drift"] = float(drift)
        except Exception:
            # Optional metrics; ignore errors
            pass
        try:
            # Constraint violation rate if counts are supplied in constraints
            if isinstance(coord.constraints, dict):
                v = coord.constraints.get("constraint_violation_count", None)
                t = coord.constraints.get("total_constraints_checked", None)
                if isinstance(v, (int, float)) and isinstance(t, (int, float)):
                    rate = InformationMetrics.compute_constraint_violation_rate(int(v), int(t))
                    row["info:constraint_violation_rate"] = float(rate)
        except Exception:
            pass
        # Escape events (if available)
        try:
            if hasattr(coord, "get_escape_event_count"):
                row["escape_event_count"] = int(getattr(coord, "get_escape_event_count")())
        except Exception:
            pass
        # Confidence (if available)
        try:
            if hasattr(coord, "last_confidence"):
                cval = getattr(coord, "last_confidence")()
                if cval is not None:
                    row["confidence:c"] = float(cval)
        except Exception:
            pass
        # AGM phase metrics and uncertainty (computed on recent energy history)
        try:
            recent = self._energy_history[-32:] if self._energy_history else []
            agm = compute_agm_phase_metrics(recent)
            row["agm:rate"] = float(agm.get("rate", 0.0))
            row["agm:variance"] = float(agm.get("variance", 0.0))
            row["agm:trend"] = float(agm.get("trend", 0.0))
            row["agm:oscillation"] = float(agm.get("oscillation", 0.0))
            unc = compute_uncertainty_metrics(recent, recent_performance=None)
            row["uncertainty:epistemic"] = float(unc.epistemic)
            row["uncertainty:aleatoric"] = float(unc.aleatoric)
            row["uncertainty:total"] = float(unc.total)
            row["uncertainty:exploration_boost"] = float(unc.exploration_boost)
        except Exception:
            # Best-effort; skip if metrics fail
            pass
        # Stability/backtracks (if available)
        margin = getattr(coord, "_last_contraction_margin", None)
        last_bk = getattr(coord, "_last_step_backtracks", None)
        total_bk = getattr(coord, "_total_backtracks", None)
        if margin is not None:
            row["contraction_margin"] = float(margin)
            if self.warn_on_margin_shrink and float(margin) < float(self.margin_warn_threshold):
                row["margin_warn"] = 1
            else:
                row["margin_warn"] = 0
        # Backtrack counts
        if last_bk is not None:
            row["last_backtracks"] = int(last_bk)
        if total_bk is not None:
            row["total_backtracks"] = int(total_bk)
        # Acceptance reason (if tracked by coordinator)
        accept_reason = getattr(coord, "_last_acceptance_reason", None)
        if accept_reason is not None:
            row["acceptance_reason"] = str(accept_reason)
        # Lipschitz allocator details (if available)
        details = getattr(coord, "_last_lipschitz_details", None)
        if isinstance(details, dict):
            gm = details.get("global_margin", None)
            if gm is not None:
                try:
                    row["margin:global"] = float(gm)
                except Exception:
                    pass
            row_m = details.get("row_margins", None)
            if isinstance(row_m, dict):
                for r, val in row_m.items():
                    try:
                        row[f"margin:row:{int(r)}"] = float(val)
                    except Exception:
                        continue
            fam_costs = details.get("family_costs", None)
            if isinstance(fam_costs, dict):
                for k, val in fam_costs.items():
                    try:
                        row[f"cost:{str(k)}"] = float(val)
                    except Exception:
                        continue
        # Adapter allocations/scores if present
        wa = getattr(coord, "weight_adapter", None)
        if wa is not None:
            scores = getattr(wa, "scores", None)
            if isinstance(scores, dict):
                for k, val in scores.items():
                    try:
                        row[f"score:{str(k)}"] = float(val)
                    except Exception:
                        continue
            allocs = getattr(wa, "last_allocations", None)
            if isinstance(allocs, dict):
                for k, val in allocs.items():
                    try:
                        row[f"alloc:{str(k)}"] = float(val)
                    except Exception:
                        continue
            spent_g = getattr(wa, "last_spent_global", None)
            if isinstance(spent_g, (int, float)):
                row["spent:global"] = float(spent_g)
        # Homotopy telemetry if available
        homo_scale = getattr(coord, "_homotopy_scale", None)
        if isinstance(homo_scale, (int, float)):
            row["homotopy_scale"] = float(homo_scale)
        homo_backoffs = getattr(coord, "_homotopy_backoffs", None)
        if isinstance(homo_backoffs, (int, float)):
            row["homotopy_backoffs"] = float(homo_backoffs)
        # Polynomial basis decorrelation monitor (aPC-style)
        try:
            for idx, (m, eta) in enumerate(zip(coord.modules, etas)):
                # Detect polynomial module by method presence
                if hasattr(m, "get_basis_values"):
                    key = f"poly:{idx}"
                    vals = m.get_basis_values(float(eta), coord.constraints)  # type: ignore[attr-defined]
                    hist = self._poly_history.get(key, [])
                    hist.append([float(v) for v in vals])
                    if len(hist) > int(self.poly_history_window):
                        hist = hist[-int(self.poly_history_window):]
                    self._poly_history[key] = hist
                    if len(hist) >= 4:
                        # Compute correlation matrix (features x features)
                        k = len(hist[0])
                        # Compute mean per feature
                        means = [0.0] * k
                        for rowv in hist:
                            for j in range(k):
                                means[j] += rowv[j]
                        n = float(len(hist))
                        means = [m / n for m in means]
                        # Compute covariance matrix
                        cov = [[0.0 for _ in range(k)] for _ in range(k)]
                        for rowv in hist:
                            for a in range(k):
                                da = rowv[a] - means[a]
                                for b in range(k):
                                    db = rowv[b] - means[b]
                                    cov[a][b] += da * db
                        for a in range(k):
                            for b in range(k):
                                cov[a][b] = cov[a][b] / max(1.0, n - 1.0)
                        # Convert to correlation; guard zero variance
                        var = [max(cov[j][j], 1e-12) for j in range(k)]
                        corr_max = 0.0
                        for a in range(k):
                            for b in range(k):
                                if a == b:
                                    continue
                                denom = (var[a] * var[b]) ** 0.5
                                r = cov[a][b] / denom if denom > 0.0 else 0.0
                                corr_max = max(corr_max, abs(float(r)))
                        row[f"poly_corr_max:{key}"] = float(corr_max)
                        # Emit thresholded warning flag
                        row[f"poly_corr_warn:{key}"] = 1 if corr_max >= float(self.poly_corr_warn_threshold) else 0
        except Exception:
            # Best-effort; ignore failures
            pass
        if last_bk is not None:
            row["last_backtracks"] = int(last_bk)
        if total_bk is not None:
            row["total_backtracks"] = int(total_bk)
        self.buffer.append(row)

    def flush(self) -> None:
        if not self.buffer:
            return
        log_records(self.name, self.buffer)
        self.buffer.clear()

@dataclass
class GatingMetricsLogger:
    """Lightweight logger for gating decisions (η_gate, hazard, redemption, etc.)."""

    name: str = "gating_metrics"
    run_id: str = "default"
    buffer: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, *, hazard: float, eta_gate: float, redemption: float, good: bool) -> None:
        self.buffer.append({
            "run_id": self.run_id,
            "hazard": float(hazard),
            "eta_gate": float(eta_gate),
            "redemption": float(redemption),
            "is_good": bool(good),
        })

    def flush(self) -> None:
        if not self.buffer:
            return
        log_records(self.name, self.buffer)
        self.buffer.clear()



