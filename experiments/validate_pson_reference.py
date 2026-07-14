"""Compare PSON Monte Carlo curvature costs with closed-form references."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.coordinator import EnergyCoordinator
from experiments.ablate_pson_noise import QuadraticWell


def analytic_costs(curvatures: np.ndarray, magnitude: float, precision_epsilon: float) -> Dict[str, float]:
    """Return exact costs for an axis-aligned gradient in three dimensions."""
    assert curvatures.shape == (3,)
    isotropic = magnitude * magnitude * float(np.mean(curvatures))
    orthogonal = magnitude * magnitude * float(np.mean(curvatures[1:]))
    tangent_weights = 1.0 / (precision_epsilon + curvatures[1:])
    precision = magnitude * magnitude * float(
        np.dot(curvatures[1:], tangent_weights) / np.sum(tangent_weights)
    )
    return {
        "isotropic": isotropic,
        "orthogonal": orthogonal,
        "precision_orthogonal": precision,
    }


def run_reference(samples: int, seed: int, magnitude: float) -> List[Dict[str, Any]]:
    """Estimate fixed-norm noise costs for a closed-form diagonal case."""
    assert samples > 0
    curvatures = np.asarray([2.0, 4.0, 16.0], dtype=float)
    state = [0.5, 0.5, 0.5]
    target = [0.4, 0.5, 0.5]
    epsilon = 1e-8
    expected = analytic_costs(curvatures, magnitude, epsilon)
    rng = np.random.default_rng(seed)
    raw_draws = rng.normal(0.0, 1.0, size=(samples, 3))
    rows: List[Dict[str, Any]] = []
    for mode in ("isotropic", "orthogonal", "precision_orthogonal"):
        coord = EnergyCoordinator(
            modules=[QuadraticWell(target[i], float(curvatures[i])) for i in range(3)],
            couplings=[],
            constraints={},
            noise_mode=mode,
            noise_magnitude=magnitude,
            precision_aware_noise_controller=(mode == "precision_orthogonal"),
            precision_epsilon=epsilon,
            stability_guard=False,
        )
        snapshot = coord.inspect_state(state)
        gradient = np.asarray(snapshot.gradient, dtype=float)
        diagonal = np.asarray(snapshot.precision_diagonal, dtype=float)
        observed_costs = []
        for raw in raw_draws:
            noise = coord.build_noise_vector(raw, gradient)
            observed_costs.append(float(np.dot(diagonal, noise * noise)))
        observed = float(np.mean(observed_costs))
        reference = expected[mode]
        rows.append(
            {
                "mode": mode,
                "samples": samples,
                "seed": seed,
                "noise_magnitude": magnitude,
                "analytic_expected_cost": reference,
                "monte_carlo_mean_cost": observed,
                "relative_error": abs(observed - reference) / reference,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--noise-magnitude", type=float, default=0.02)
    parser.add_argument("--max-relative-error", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("logs/pson_analytic_reference.csv"))
    args = parser.parse_args()

    rows = run_reference(int(args.samples), int(args.seed), float(args.noise_magnitude))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} reference rows to {args.output}")
    for row in rows:
        print(
            f"{row['mode']:20s} expected={float(row['analytic_expected_cost']):.6e} "
            f"observed={float(row['monte_carlo_mean_cost']):.6e} "
            f"relative_error={100.0 * float(row['relative_error']):.3f}%"
        )
    if any(float(row["relative_error"]) > float(args.max_relative_error) for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
