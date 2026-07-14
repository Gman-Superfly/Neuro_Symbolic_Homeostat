"""Measure controlled nonconvex escape under matched noise budgets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision
from experiments.ablate_pson_noise import QuadraticWell


MODES = ("none", "isotropic", "orthogonal", "precision_orthogonal")


@dataclass(frozen=True)
class AsymmetricDoubleWell(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Two stationary wells with a lower right-hand basin."""

    left: float = 0.2
    right: float = 0.8
    barrier_scale: float = 20.0
    left_energy: float = 0.1

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def _q(self, eta: float) -> float:
        return (eta - self.left) / (self.right - self.left)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        del constraints
        value = float(eta)
        q = self._q(value)
        smoothstep = 3.0 * q * q - 2.0 * q * q * q
        return (
            self.barrier_scale * (value - self.left) ** 2 * (value - self.right) ** 2
            + self.left_energy * (1.0 - smoothstep)
        )

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        del constraints
        value = float(eta)
        u = value - self.left
        v = value - self.right
        distance = self.right - self.left
        q = self._q(value)
        return (
            2.0 * self.barrier_scale * u * v * (u + v)
            - self.left_energy * (6.0 * q - 6.0 * q * q) / distance
        )

    def curvature(self, eta: OrderParameter) -> float:
        value = float(eta)
        u = value - self.left
        v = value - self.right
        distance = self.right - self.left
        q = self._q(value)
        second = (
            2.0 * self.barrier_scale * (u * u + 4.0 * u * v + v * v)
            - self.left_energy * (6.0 - 12.0 * q) / (distance * distance)
        )
        return max(abs(second), 1e-6)


def run_trial(mode: str, seed: int, steps: int, dimension: int, noise_magnitude: float) -> Dict[str, Any]:
    assert mode in MODES
    assert dimension >= 3
    np.random.seed(seed)
    modules: List[EnergyModule] = [AsymmetricDoubleWell()]
    modules.extend(
        QuadraticWell(target=0.5, curvature_value=float(value))
        for value in np.geomspace(200.0, 2000.0, num=dimension - 1)
    )
    inputs = [0.2, *([0.5] * (dimension - 1))]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=False,
        use_precision_preconditioning=False,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        noise_mode=mode,
        noise_magnitude=0.0 if mode == "none" else noise_magnitude,
        precision_aware_noise_controller=(mode == "precision_orthogonal"),
        enable_orthogonal_noise=(mode not in {"none", "isotropic"}),
        assert_monotonic_energy=False,
        continue_after_rejection=True,
    )
    etas = coord.compute_etas(inputs)
    initial_energy = coord.inspect_state(etas).energy
    accepted_states: List[List[float]] = []
    coord.on_eta_updated.append(lambda values: accepted_states.append([float(value) for value in values]))
    output = coord.relax_etas(etas, steps=steps)
    final_energy = coord.inspect_state(output).energy
    escaped_steps = [index + 1 for index, state in enumerate(accepted_states) if state[0] >= 0.6]
    metrics = coord.last_relaxation_metrics()
    return {
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "dimension": dimension,
        "noise_magnitude": 0.0 if mode == "none" else noise_magnitude,
        "escaped": int(bool(escaped_steps)),
        "escape_accepted_step": escaped_steps[0] if escaped_steps else "",
        "maximum_escape_coordinate": max([0.2, *[state[0] for state in accepted_states]]),
        "final_escape_coordinate": float(output[0]),
        "energy_initial": initial_energy,
        "energy_final": final_energy,
        "energy_drop": initial_energy - final_energy,
        "accepted_steps": int(metrics["accepted_steps"]),
        "rejected_steps": int(metrics["rejected_steps"]),
    }


def paired_escape_summary(
    rows: Sequence[Mapping[str, Any]],
    mode: str,
    baseline_mode: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    def by_seed(selected_mode: str) -> Dict[int, Mapping[str, Any]]:
        return {int(row["seed"]): row for row in rows if str(row["mode"]) == selected_mode}

    baseline = by_seed(baseline_mode)
    comparison = by_seed(mode)
    seeds = sorted(baseline)
    assert seeds == sorted(comparison) and seeds
    baseline_escape = np.asarray([float(baseline[seed]["escaped"]) for seed in seeds])
    comparison_escape = np.asarray([float(comparison[seed]["escaped"]) for seed in seeds])
    differences = comparison_escape - baseline_escape
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    bootstrap = np.mean(differences[indices], axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mode": mode,
        "baseline_mode": baseline_mode,
        "trials": len(seeds),
        "escape_rate": float(np.mean(comparison_escape)),
        "baseline_escape_rate": float(np.mean(baseline_escape)),
        "paired_escape_rate_difference": float(np.mean(differences)),
        "difference_ci_low": float(interval[0]),
        "difference_ci_high": float(interval[1]),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--noise-magnitude", type=float, default=0.55)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("logs/pson_escape_trials.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("logs/pson_escape_summary.csv"))
    args = parser.parse_args()

    rows = [
        run_trial(mode, seed, int(args.steps), int(args.dimension), float(args.noise_magnitude))
        for mode in MODES
        for seed in range(int(args.trials))
    ]
    summaries = [
        paired_escape_summary(
            rows,
            mode,
            baseline,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=20260714 + mode_index * 10 + baseline_index,
        )
        for mode_index, mode in enumerate(("isotropic", "orthogonal", "precision_orthogonal"))
        for baseline_index, baseline in enumerate(("none", "isotropic"))
        if mode != baseline
    ]
    _write(args.output, rows)
    _write(args.summary_output, summaries)
    print(f"wrote {len(rows)} trials to {args.output}")
    print(f"wrote {len(summaries)} paired summaries to {args.summary_output}")
    for mode in MODES:
        subset = [row for row in rows if row["mode"] == mode]
        print(
            f"{mode:20s} escape_rate={np.mean([row['escaped'] for row in subset]):.3f} "
            f"mean_energy_drop={np.mean([row['energy_drop'] for row in subset]):.6f}"
        )


if __name__ == "__main__":
    main()
