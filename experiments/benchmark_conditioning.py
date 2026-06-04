"""Measure the condition-number benefit of diagonal precision preconditioning.

The benchmark builds an uncoupled quadratic energy whose coordinates have a
large spread in curvature (stiffness). Plain gradient descent must use a single
global step small enough for the stiffest coordinate, so the slackest coordinate
converges slowly. Diagonal precision preconditioning divides each coordinate's
gradient by its curvature, so every coordinate converges on a similar timescale.

Both configurations use the same stability guard and the same Lipschitz-derived
step (auto_step_from_lipschitz), so the only difference is the preconditioner.
This is the fair comparison: plain gradient descent is given its near-optimal
fixed step, and the preconditioned run uses the identical global step.

Usage:
    uv run python -m experiments.benchmark_conditioning
    uv run python -m experiments.benchmark_conditioning --stiffness 200 1 0.02
    uv run python -m experiments.benchmark_conditioning --csv logs/conditioning_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision

__all__ = ["QuadraticWell", "run_single_config", "run_benchmark", "main"]


@dataclass(frozen=True)
class QuadraticWell(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Local quadratic well with a known, exposed curvature (stiffness)."""

    target: float
    stiffness: float

    def compute_eta(self, x: Any) -> OrderParameter:
        """Return the supplied initial order parameter, clamped to [0, 1]."""
        eta = float(x)
        assert 0.0 <= eta <= 1.0, f"eta out of bounds: {eta}"
        return eta

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        """Compute the quadratic energy 0.5 * stiffness * (eta - target)^2."""
        diff = float(eta) - self.target
        return 0.5 * self.stiffness * diff * diff

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        """Compute the analytic gradient stiffness * (eta - target)."""
        return self.stiffness * (float(eta) - self.target)

    def curvature(self, eta: OrderParameter) -> float:
        """Return the local curvature, which equals the stiffness for a quadratic."""
        return self.stiffness


def _max_abs_error(etas: List[float], targets: List[float]) -> float:
    """Return the largest absolute deviation from target across coordinates."""
    assert len(etas) == len(targets), "etas and targets length mismatch"
    return max(abs(float(e) - float(t)) for e, t in zip(etas, targets))


def run_single_config(
    targets: List[float],
    stiffness: List[float],
    init: List[float],
    *,
    precondition: bool,
    tol: float,
    max_iters: int,
) -> Dict[str, Any]:
    """Run one relaxation to convergence and count the iterations.

    Args:
        targets: Per-coordinate quadratic-well centers in [0, 1].
        stiffness: Per-coordinate curvature values (all positive).
        init: Initial order parameters in [0, 1].
        precondition: If True, divide each gradient by its diagonal curvature.
        tol: Convergence threshold on the max absolute error to target.
        max_iters: Iteration cap. Convergence is reported as False if reached.

    Returns:
        Dictionary with iteration count, convergence flag, final max error, and
        the step size selected by the stability guard on the first iteration.
    """
    assert len(targets) == len(stiffness) == len(init), "input length mismatch"
    assert all(s > 0.0 for s in stiffness), "stiffness must be positive"
    assert tol > 0.0, "tol must be positive"
    assert max_iters > 0, "max_iters must be positive"

    modules = [QuadraticWell(target=t, stiffness=s) for t, s in zip(targets, stiffness)]
    coordinator = EnergyCoordinator(
        modules=modules,
        couplings=[],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=False,
        use_precision_preconditioning=precondition,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        stability_cap_fraction=0.9,
        noise_mode="none",
        enable_orthogonal_noise=False,
        assert_monotonic_energy=False,
        enable_early_stop=False,
    )

    etas = [float(e) for e in init]
    # For an uncoupled quadratic energy the Gershgorin Lipschitz bound equals the
    # largest stiffness (off-diagonal terms are zero), so the guard selects this
    # step. Both configurations select the same step; only the per-coordinate
    # scaling differs.
    step_used = 0.9 * 2.0 / max(stiffness)
    iters = 0
    for iters in range(1, max_iters + 1):
        etas = [float(e) for e in coordinator.relax_etas(etas, steps=1)]
        if _max_abs_error(etas, targets) <= tol:
            return {
                "precondition": precondition,
                "iterations": iters,
                "converged": True,
                "final_max_error": _max_abs_error(etas, targets),
                "step_used": step_used,
            }

    return {
        "precondition": precondition,
        "iterations": max_iters,
        "converged": False,
        "final_max_error": _max_abs_error(etas, targets),
        "step_used": step_used,
    }


def run_benchmark(
    targets: List[float],
    stiffness: List[float],
    init: List[float],
    *,
    tol: float,
    max_iters: int,
) -> Dict[str, Any]:
    """Run plain gradient descent and the preconditioned variant on one problem.

    Args:
        targets: Per-coordinate quadratic-well centers in [0, 1].
        stiffness: Per-coordinate curvature values (all positive).
        init: Initial order parameters in [0, 1].
        tol: Convergence threshold on the max absolute error to target.
        max_iters: Iteration cap for each configuration.

    Returns:
        Dictionary with the condition number, both result records, and the
        iteration speedup ratio when both configurations converged.
    """
    condition_number = max(stiffness) / min(stiffness)
    plain = run_single_config(
        targets, stiffness, init, precondition=False, tol=tol, max_iters=max_iters
    )
    precond = run_single_config(
        targets, stiffness, init, precondition=True, tol=tol, max_iters=max_iters
    )
    speedup = None
    if plain["converged"] and precond["converged"] and precond["iterations"] > 0:
        speedup = plain["iterations"] / precond["iterations"]
    return {
        "condition_number": condition_number,
        "plain": plain,
        "precond": precond,
        "speedup": speedup,
    }


def _write_csv(path: Path, result: Dict[str, Any], stiffness: List[float]) -> None:
    """Write one benchmark result as two rows (plain and preconditioned)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["config", "condition_number", "iterations", "converged", "final_max_error", "step_used", "stiffness"]
        )
        stiff_repr = "|".join(f"{s:g}" for s in stiffness)
        for key in ("plain", "precond"):
            rec = result[key]
            writer.writerow(
                [
                    "plain_gd" if key == "plain" else "precond",
                    f"{result['condition_number']:.6g}",
                    rec["iterations"],
                    rec["converged"],
                    f"{rec['final_max_error']:.6e}",
                    f"{rec['step_used']:.6e}",
                    stiff_repr,
                ]
            )


def main() -> None:
    """Parse arguments, run the benchmark, and print a comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stiffness",
        type=float,
        nargs="+",
        default=[100.0, 1.0, 0.1],
        help="Per-coordinate curvature values (default: 100 1 0.1).",
    )
    parser.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance.")
    parser.add_argument("--max-iters", type=int, default=100000, help="Iteration cap per config.")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    stiffness = list(args.stiffness)
    n = len(stiffness)
    assert n >= 1, "need at least one coordinate"
    # Keep the trajectory in the interior of [0, 1]. Plain gradient descent uses a
    # step near 2/L_max, so the stiffest coordinate oscillates with a factor near
    # -1. With a uniform target of 0.5 and an initial offset of 0.3, the stiffest
    # coordinate lands at 0.5 - 0.8 * 0.3 = 0.26 and never reaches a box edge, so
    # the comparison measures convergence rate rather than a boundary-clamp
    # artifact in the finite-difference curvature estimate.
    targets = [0.5] * n
    init = [0.8] * n

    result = run_benchmark(targets, stiffness, init, tol=args.tol, max_iters=args.max_iters)

    print("=" * 72)
    print("DIAGONAL PRECISION PRECONDITIONING: CONDITION-NUMBER BENCHMARK")
    print("=" * 72)
    print(f"coordinates:       {n}")
    print(f"stiffness:         {stiffness}")
    print(f"targets:           {targets}")
    print(f"init:              {init}")
    print(f"condition number:  {result['condition_number']:.6g}")
    print(f"tolerance:         {args.tol:g}")
    print("-" * 72)
    plain = result["plain"]
    precond = result["precond"]
    print(
        f"plain gradient descent:  iterations={plain['iterations']:>8}  "
        f"converged={plain['converged']}  step={plain['step_used']:.4e}  "
        f"final_err={plain['final_max_error']:.3e}"
    )
    print(
        f"precision preconditioned:iterations={precond['iterations']:>8}  "
        f"converged={precond['converged']}  step={precond['step_used']:.4e}  "
        f"final_err={precond['final_max_error']:.3e}"
    )
    print("-" * 72)
    if result["speedup"] is not None:
        print(f"iteration speedup (plain / precond): {result['speedup']:.2f}x")
    else:
        print("speedup undefined: at least one configuration did not converge in max_iters")
    print("=" * 72)

    if args.csv is not None:
        out_path = Path(args.csv)
        _write_csv(out_path, result, stiffness)
        print(f"wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
