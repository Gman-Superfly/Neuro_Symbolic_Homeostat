"""Benchmark harness for ΔF90 comparisons across coordinator configs.

Usage:
    uv run python -m experiments.benchmark_delta_f90
    uv run python -m experiments.benchmark_delta_f90 --configs default analytic vect hinge coord

Outputs a CSV under logs/benchmark_delta_f90.csv with:
    run_id, config_name, steps, wall_time_sec, delta_f90_steps
"""

from __future__ import annotations

import argparse
import time
from typing import List, Dict, Any, Tuple

from modules.gating.energy_gating import EnergyGatingModule
from core.couplings import QuadraticCoupling, GateBenefitCoupling
from core.coordinator import EnergyCoordinator
from cf_logging.metrics_log import log_records
from cf_logging.observability import EnergyBudgetTracker
from core.weight_adapters import GradNormWeightAdapter, AGMPhaseWeightAdapter
try:
    from core.weight_adapters import SmallGainWeightAdapter  # type: ignore
except Exception:
    SmallGainWeightAdapter = None  # fallback if not present


def make_modules_and_couplings() -> Tuple[List[Any], List[Tuple[int, int, Any]], Dict[str, Any], List[Any]]:
    seq = [i / 63 for i in range(64)]
    seq_mod = EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3)
    gate_mod = EnergyGatingModule(gain_fn=lambda _: 0.1, cost=0.05, a=0.25, b=0.35)
    mods = [seq_mod, gate_mod]
    coups = [
        (0, 1, QuadraticCoupling(weight=0.7)),
        (1, 0, GateBenefitCoupling(weight=0.8, delta_key="delta_eta_domain")),
    ]
    constraints = {"delta_eta_domain": 0.08}
    return mods, coups, constraints, [seq, None]


def make_dense_modules_and_couplings(n: int = 16) -> Tuple[List[Any], List[Tuple[int, int, Any]], Dict[str, Any], List[Any]]:
    assert n >= 3, "dense scenario requires at least 3 modules"
    modules: List[EnergyGatingModule] = []
    inputs: List[Any] = []
    for idx in range(n):
        gain_offset = 0.02 * (idx % 5)
        modules.append(EnergyGatingModule(gain_fn=lambda _, go=gain_offset: go, a=0.15, b=0.05))
        inputs.append([idx / max(1, n - 1) for _ in range(32)])
    couplings: List[Tuple[int, int, Any]] = []
    for i in range(n):
        j = (i + 1) % n
        couplings.append((i, j, QuadraticCoupling(weight=0.6)))
        couplings.append((j, i, GateBenefitCoupling(weight=0.4, delta_key="delta_dense")))
        k = (i + 2) % n
        couplings.append((i, k, QuadraticCoupling(weight=0.3)))
    constraints = {"delta_dense": 0.05}
    return modules, couplings, constraints, inputs


def delta_f90(energies: List[float]) -> int:
    if len(energies) < 2:
        return len(energies)
    total_drop = energies[0] - min(energies)
    if total_drop <= 0.0:
        return len(energies)
    threshold = energies[0] - 0.9 * total_drop
    for idx, val in enumerate(energies):
        if val <= threshold:
            return idx + 1
    return len(energies)


def _per_term_breakdown(coord: EnergyCoordinator, etas: List[float]) -> Dict[str, float]:
    """Return per-term energy contributions and gradient norms keyed by term names."""
    out: Dict[str, float] = {}
    cw = coord._combined_term_weights()  # instrumentation
    # Local energies
    for idx, m in enumerate(coord.modules):
        key = f"local:{m.__class__.__name__}"
        w = float(cw.get(key, 1.0))
        e = float(m.local_energy(float(etas[idx]), coord.constraints)) * w
        out[f"energy:{key}"] = e
    # Coupling energies
    for i, j, coup in coord.couplings:
        key = f"coup:{coup.__class__.__name__}"
        w = float(cw.get(key, 1.0))
        e = float(coup.coupling_energy(float(etas[i]), float(etas[j]), coord.constraints)) * w
        out[f"energy:{key}"] = out.get(f"energy:{key}", 0.0) + e
    # Gradient norms per term
    norms = coord._term_grad_norms(etas)  # instrumentation
    for k, v in norms.items():
        out[f"grad_norm:{k}"] = float(v)
    return out


def run_config(
    name: str,
    coord_kwargs: Dict[str, Any],
    steps: int,
    scenario: str,
    dense_size: int,
    log_budget: bool,
    budget_name: str,
    run_id: str,
    warn_on_margin_shrink: bool = False,
    margin_warn_threshold: float | None = None,
) -> Dict[str, Any]:
    if scenario == "dense":
        mods, coups, constraints, inputs = make_dense_modules_and_couplings(dense_size)
        coord_kwargs = dict(coord_kwargs)
        coord_kwargs.setdefault("use_vectorized_quadratic", True)
        coord_kwargs.setdefault("use_vectorized_hinges", True)
        coord_kwargs.setdefault("use_vectorized_gate_benefits", True)
        coord_kwargs.setdefault("neighbor_gradients_only", False)
    else:
        mods, coups, constraints, inputs = make_modules_and_couplings()
    # Materialize adapter if requested via a sentinel in coord_kwargs
    adapter_name = coord_kwargs.pop("_adapter", None)
    # Optional SmallGain overrides (inserted by caller)
    sg_rho = coord_kwargs.pop("sg_rho", None)
    sg_dw = coord_kwargs.pop("sg_dw", None)
    adapter = None
    if adapter_name == "gradnorm":
        adapter = GradNormWeightAdapter()
    elif adapter_name == "agm":
        adapter = AGMPhaseWeightAdapter()
    elif adapter_name == "smallgain" and SmallGainWeightAdapter is not None:
        # Allow parameter overrides when provided
        if sg_rho is not None or sg_dw is not None:
            adapter = SmallGainWeightAdapter(
                budget_fraction=float(sg_rho) if sg_rho is not None else 0.7,
                max_step_change=float(sg_dw) if sg_dw is not None else 0.10,
            )
        else:
            adapter = SmallGainWeightAdapter()
        # ensure coordinator exposes details needed by the adapter
        coord_kwargs = dict(coord_kwargs)
        coord_kwargs["expose_lipschitz_details"] = True
        # keep stability guard on for a meaningful contraction margin
        coord_kwargs.setdefault("stability_guard", True)
        coord_kwargs.setdefault("log_contraction_margin", True)
    coord = EnergyCoordinator(mods, coups, constraints, weight_adapter=adapter, **coord_kwargs)
    tracker = None
    if log_budget:
        tracker = EnergyBudgetTracker(name=budget_name, run_id=f"{run_id}_{name}")
        # Apply margin warning configuration if requested
        if warn_on_margin_shrink:
            tracker.warn_on_margin_shrink = True
        if margin_warn_threshold is not None:
            try:
                tracker.margin_warn_threshold = float(margin_warn_threshold)
            except Exception:
                pass
        tracker.attach(coord)
    etas = coord.compute_etas(inputs)
    energies: List[float] = []
    coord.on_energy_updated.append(lambda F: energies.append(F))
    start = time.perf_counter()
    coord.relax_etas(etas, steps=steps)
    duration = time.perf_counter() - start
    if not energies:
        energies = [coord.energy(etas)]
    total_improvement = 0.0
    if energies:
        total_improvement = max(energies[0] - energies[-1], 0.0)
    redemption_gain = total_improvement / duration if duration > 0.0 else float("nan")
    row: Dict[str, Any] = {
        "config": name,
        "steps": steps,
        "wall_time_sec": duration,
        "compute_cost": duration,
        "redemption_gain": redemption_gain,
        "delta_f90_steps": delta_f90(energies),
        "energy_final": energies[-1],
    }
    # Backtracks (if tracked by coordinator)
    last_bk = getattr(coord, "_last_step_backtracks", None)
    total_bk = getattr(coord, "_total_backtracks", None)
    if isinstance(last_bk, int):
        row["last_backtracks"] = int(last_bk)
    if isinstance(total_bk, int):
        row["total_backtracks"] = int(total_bk)
    # If SmallGain overrides were used, include them in the log row for sweep analysis
    if adapter_name == "smallgain":
        if sg_rho is not None:
            row["sg_rho"] = float(sg_rho)
        if sg_dw is not None:
            row["sg_dw"] = float(sg_dw)
    # Append per-term breakdown at the end of relaxation
    breakdown = _per_term_breakdown(coord, etas)
    row.update(breakdown)
    # Mark mode flags for analysis
    row["operator_splitting"] = bool(getattr(coord, "operator_splitting", False))
    row["adapter"] = str(adapter_name or "none")
    if tracker is not None:
        tracker.flush()
    return row


PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "use_analytic": False,
        "use_vectorized_quadratic": False,
        "use_vectorized_hinges": False,
        "neighbor_gradients_only": False,
    },
    "analytic": {
        "use_analytic": True,
        "use_vectorized_quadratic": False,
        "use_vectorized_hinges": False,
        "neighbor_gradients_only": False,
    },
    "vect": {
        "use_analytic": True,
        "use_vectorized_quadratic": True,
        "use_vectorized_hinges": True,
        "neighbor_gradients_only": True,
        "line_search": True,
        "normalize_grads": True,
    },
    "coord": {
        "use_analytic": True,
        "use_vectorized_quadratic": True,
        "neighbor_gradients_only": True,
        "use_coordinate_descent": True,
        "coordinate_steps": 60,
    },
    "adaptive": {
        "use_analytic": True,
        "use_vectorized_quadratic": True,
        "use_vectorized_hinges": True,
        "neighbor_gradients_only": True,
        "adaptive_coordinate_descent": True,
        "coordinate_steps": 30,
        "line_search": True,
    },
    # Operator-splitting / proximal mode
    "prox": {
        "use_analytic": True,
        "operator_splitting": True,
        "prox_steps": 60,
        "prox_tau": 0.05,
    },
    "prox_star": {
        "use_analytic": True,
        "operator_splitting": True,
        "prox_steps": 60,
        "prox_tau": 0.05,
        "prox_block_mode": "star",
    },
    # Adapter comparisons (gradient-based relaxation path)
    "gradnorm": {
        "use_analytic": True,
        "_adapter": "gradnorm",
        "line_search": True,
        "normalize_grads": True,
    },
    "agm": {
        "use_analytic": True,
        "_adapter": "agm",
        "line_search": True,
        "normalize_grads": True,
    },
    "smallgain": {
        "use_analytic": True,
        "_adapter": "smallgain",
        "line_search": True,
        "normalize_grads": True,
        # helpful toggles
        "use_vectorized_quadratic": True,
        "use_vectorized_hinges": True,
        "neighbor_gradients_only": True,
        # stability and telemetry
        "stability_guard": True,
        "log_contraction_margin": True,
        "expose_lipschitz_details": True,
    },
    "admm": {
        "use_analytic": True,
        "use_admm": True,
        "admm_steps": 60,
        "admm_rho": 1.0,
        "admm_step_size": 0.05,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", choices=list(PRESETS.keys()), default=list(PRESETS.keys()))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--run_id", type=str, default="benchmark_delta_f90")
    parser.add_argument("--scenario", choices=["baseline", "dense"], default="baseline")
    parser.add_argument("--dense_size", type=int, default=16)
    parser.add_argument("--log_budget", action="store_true")
    parser.add_argument("--budget_name", type=str, default="benchmark_delta_f90_budget")
    # Budget tracker margin warning toggles
    parser.add_argument("--warn_on_margin_shrink", action="store_true", help="Emit margin_warn=1 when contraction_margin < threshold")
    parser.add_argument("--margin_warn_threshold", type=float, default=None, help="Threshold for contraction margin warnings (default 1e-4)")
    # SmallGain sweep knobs (optional)
    parser.add_argument("--sg_rho", type=float, default=None, help="SmallGain budget_fraction (ρ), e.g. 0.5/0.7/0.9")
    parser.add_argument("--sg_dw", type=float, default=None, help="SmallGain max_step_change (per-step Δweight), e.g. 0.05/0.10/0.20")
    # ADMM knobs (optional)
    parser.add_argument("--admm_gate_prox", action="store_true", help="Enable prox-linear update for gate-benefit in ADMM")
    parser.add_argument("--admm_gate_damping", type=float, default=None, help="Blend (0..1) for ADMM gate prox step; default 0.5")
    # Logit updates (optional)
    parser.add_argument("--use_logit_updates", action="store_true", help="Enable mirror/logit parameterization for η updates in gradient mode")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for cfg in args.configs:
        # Clone preset so we can inject optional SmallGain overrides
        preset = dict(PRESETS[cfg])
        if cfg == "smallgain":
            if args.sg_rho is not None:
                preset["sg_rho"] = float(args.sg_rho)
            if args.sg_dw is not None:
                preset["sg_dw"] = float(args.sg_dw)
        if cfg == "admm":
            if args.admm_gate_prox:
                preset["admm_gate_prox"] = True
            if args.admm_gate_damping is not None:
                preset["admm_gate_damping"] = float(args.admm_gate_damping)
        # apply logit updates only to gradient-based configs
        if args.use_logit_updates and cfg in ("default", "analytic", "vect", "coord", "adaptive", "gradnorm", "agm", "smallgain"):
            preset["use_logit_updates"] = True
        result = run_config(
            cfg,
            preset,
            args.steps,
            scenario=args.scenario,
            dense_size=args.dense_size,
            log_budget=args.log_budget,
            budget_name=args.budget_name,
            run_id=args.run_id,
            warn_on_margin_shrink=args.warn_on_margin_shrink,
            margin_warn_threshold=(args.margin_warn_threshold if args.margin_warn_threshold is not None else None),
        )
        result["run_id"] = args.run_id
        rows.append(result)
        print(
            f"[{cfg}] dF90 steps={result['delta_f90_steps']}, "
            f"wall_time={result['wall_time_sec']:.4f}s, energy_final={result['energy_final']:.6f}"
        )
    path = log_records("benchmark_delta_f90", rows)
    print(f"Logged {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()

