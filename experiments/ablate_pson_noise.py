"""Ablate isotropic, orthogonal, and precision-orthogonal noise modes.

Usage:
    uv run python -m experiments.ablate_pson_noise --quick
    uv run python -m experiments.ablate_pson_noise --trials 20 --steps 80
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling, QuadraticCoupling
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision


@dataclass
class QuadraticWell(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Local quadratic energy with an exposed curvature."""

    target: float
    curvature_value: float

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        diff = float(eta) - self.target
        return 0.5 * self.curvature_value * diff * diff

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return self.curvature_value * (float(eta) - self.target)

    def curvature(self, eta: OrderParameter) -> float:
        return self.curvature_value


def delta_f90(energies: List[float]) -> int:
    assert len(energies) > 0, "energy trace required"
    if len(energies) < 2:
        return len(energies)
    total_drop = energies[0] - min(energies)
    if total_drop <= 0.0:
        return len(energies)
    threshold = energies[0] - 0.9 * total_drop
    for idx, energy in enumerate(energies):
        if energy <= threshold:
            return idx + 1
    return len(energies)


def build_quadratic_case(seed: int, size: int) -> Tuple[List[QuadraticWell], List[Tuple[int, int, Any]], Dict[str, Any], List[float]]:
    assert size >= 3, "size must be at least 3"
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0.15, 0.85, size=size)
    curvatures = np.geomspace(0.25, 8.0, num=size)
    modules = [QuadraticWell(float(targets[i]), float(curvatures[i])) for i in range(size)]
    couplings: List[Tuple[int, int, Any]] = []
    for idx in range(size - 1):
        couplings.append((idx, idx + 1, QuadraticCoupling(weight=0.15 + 0.03 * (idx % 3))))
    inputs = [float(x) for x in rng.uniform(0.0, 1.0, size=size)]
    return modules, couplings, {}, inputs


def build_mixed_gate_case(seed: int, size: int) -> Tuple[List[QuadraticWell], List[Tuple[int, int, Any]], Dict[str, Any], List[float]]:
    modules, couplings, constraints, inputs = build_quadratic_case(seed, size)
    constraints = dict(constraints)
    constraints["delta_benefit"] = 0.08
    for idx in range(0, size - 1, 2):
        couplings.append((idx, idx + 1, GateBenefitCoupling(weight=0.25, delta_key="delta_benefit")))
    return modules, couplings, constraints, inputs


def curvature_noise_cost(coord: EnergyCoordinator, etas: List[float], seed: int) -> float:
    coord._update_precision_cache(etas)  # type: ignore[attr-defined]
    grad = np.asarray(coord._grads(etas), dtype=float)  # type: ignore[attr-defined]
    raw = np.random.default_rng(seed).normal(0.0, 1.0, size=grad.shape)
    noise = coord._build_noise_vector(raw, grad, current_noise_mag=coord.noise_magnitude)  # type: ignore[attr-defined]
    diag = np.asarray(coord.get_precision_diagonal(), dtype=float)
    return float(np.sum(diag * noise * noise))


def run_one(mode: str, scenario: str, seed: int, steps: int, size: int, noise_magnitude: float) -> Dict[str, Any]:
    np.random.seed(seed)
    if scenario == "mixed":
        modules, couplings, constraints, inputs = build_mixed_gate_case(seed, size)
    else:
        modules, couplings, constraints, inputs = build_quadratic_case(seed, size)
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints=constraints,
        use_analytic=True,
        use_stiffness_updates=True,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        noise_mode=mode,
        noise_magnitude=noise_magnitude,
        precision_aware_noise_controller=(mode == "precision_orthogonal"),
        enable_orthogonal_noise=(mode != "isotropic"),
        assert_monotonic_energy=False,
    )
    etas = coord.compute_etas(inputs)
    initial_energy = float(coord._energy_value(etas))  # type: ignore[attr-defined]
    noise_cost = curvature_noise_cost(coord, list(etas), seed + 10_000)
    energies: List[float] = [initial_energy]
    coord.on_energy_updated.append(lambda energy: energies.append(float(energy)))
    start = time.perf_counter()
    out = coord.relax_etas(etas, steps=steps)
    wall_time = time.perf_counter() - start
    final_energy = float(coord._energy_value(out))  # type: ignore[attr-defined]
    rejected_steps = int(getattr(coord, "_rejected_steps", 0))
    accepted_steps = max(0, len(energies) - 1)
    attempted_steps = accepted_steps + rejected_steps
    acceptance_rate = accepted_steps / attempted_steps if attempted_steps > 0 else 1.0
    return {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "size": size,
        "noise_magnitude": noise_magnitude,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "acceptance_rate": acceptance_rate,
        "delta_f90_steps": delta_f90(energies),
        "energy_initial": initial_energy,
        "energy_final": final_energy,
        "energy_drop": initial_energy - final_energy,
        "noise_curvature_cost": noise_cost,
        "wall_time_sec": wall_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small validation sweep.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--noise-magnitude", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("logs/pson_noise_ablation.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = 3 if args.quick else int(args.trials)
    steps = 25 if args.quick else int(args.steps)
    modes = ["isotropic", "orthogonal", "precision_orthogonal"]
    scenarios = ["quadratic", "mixed"]
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for mode in modes:
            for seed in range(trials):
                rows.append(run_one(mode, scenario, seed, steps, int(args.size), float(args.noise_magnitude)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}")
    for scenario in scenarios:
        for mode in modes:
            subset = [row for row in rows if row["scenario"] == scenario and row["mode"] == mode]
            mean_drop = float(np.mean([row["energy_drop"] for row in subset]))
            mean_accept = float(np.mean([row["acceptance_rate"] for row in subset]))
            mean_noise_cost = float(np.mean([row["noise_curvature_cost"] for row in subset]))
            print(
                f"{scenario:9s} {mode:20s} "
                f"drop={mean_drop:.6f} accept={mean_accept:.3f} noise_curvature_cost={mean_noise_cost:.6e}"
            )


if __name__ == "__main__":
    main()
