"""Ablate isotropic, orthogonal, and precision-orthogonal noise modes.

Usage:
    uv run python -m experiments.ablate_pson_noise --quick
    uv run python -m experiments.ablate_pson_noise --trials 30 --steps 80
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from core.coordinator import EnergyCoordinator
from core.coordinator_noise import apply_box_feasible_noise
from core.couplings import (
    AsymmetricHingeCoupling,
    DirectedHingeCoupling,
    GateBenefitCoupling,
    QuadraticCoupling,
)
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision


NOISE_MODES = ("isotropic", "orthogonal", "precision_orthogonal")
SCENARIOS = (
    "quadratic_chain",
    "mixed_gate_chain",
    "quadratic_star",
    "quadratic_dense",
    "ill_conditioned_ring",
    "nonlinear_quartic",
    "active_hinges",
)


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


@dataclass
class QuarticWell(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Convex well with state-dependent curvature."""

    target: float
    quadratic_curvature: float
    quartic_strength: float

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        diff = float(eta) - self.target
        return 0.5 * self.quadratic_curvature * diff * diff + 0.25 * self.quartic_strength * diff**4

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        diff = float(eta) - self.target
        return self.quadratic_curvature * diff + self.quartic_strength * diff**3

    def curvature(self, eta: OrderParameter) -> float:
        diff = float(eta) - self.target
        return self.quadratic_curvature + 3.0 * self.quartic_strength * diff * diff


@dataclass
class SyntheticCase:
    """One generated problem instance and its audit metadata."""

    modules: List[EnergyModule]
    couplings: List[Tuple[int, int, Any]]
    constraints: Dict[str, Any]
    inputs: List[float]
    topology: str
    energy_family: str

    def metadata(self) -> Dict[str, Any]:
        curvatures = np.asarray(
            [
                float(module.curvature(value)) if isinstance(module, SupportsPrecision) else 0.0
                for module, value in zip(self.modules, self.inputs)
            ],
            dtype=float,
        )
        positive = curvatures[curvatures > 0.0]
        minimum = float(np.min(positive)) if positive.size else 0.0
        maximum = float(np.max(positive)) if positive.size else 0.0
        return {
            "topology": self.topology,
            "energy_family": self.energy_family,
            "size": len(self.modules),
            "coupling_count": len(self.couplings),
            "initial_curvature_min": minimum,
            "initial_curvature_max": maximum,
            "initial_curvature_ratio": maximum / minimum if minimum > 0.0 else float("inf"),
        }


@dataclass(frozen=True)
class NoiseCurvatureCosts:
    """Requested and initial-state box-feasible costs on shared noise draws."""

    diagonal_proxy_mean: float
    diagonal_proxy_draws: List[float]
    full_hessian_mean: float
    full_hessian_draws: List[float]
    realized_diagonal_proxy_mean: float
    realized_diagonal_proxy_draws: List[float]
    realized_full_hessian_mean: float
    realized_full_hessian_draws: List[float]
    box_scale_mean: float
    box_scaled_fraction: float


def delta_f90(energies: List[float]) -> int:
    assert energies, "energy trace required"
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


def _quadratic_modules(
    rng: np.random.Generator,
    size: int,
    curvature_min: float,
    curvature_max: float,
    *,
    shuffle_curvatures: bool,
) -> Tuple[List[QuadraticWell], List[float]]:
    targets = rng.uniform(0.15, 0.85, size=size)
    curvatures = np.geomspace(curvature_min, curvature_max, num=size)
    if shuffle_curvatures:
        curvatures = rng.permutation(curvatures)
    modules = [QuadraticWell(float(targets[i]), float(curvatures[i])) for i in range(size)]
    inputs = [float(value) for value in rng.uniform(0.0, 1.0, size=size)]
    return modules, inputs


def _chain_edges(size: int, *, base_weight: float = 0.15) -> List[Tuple[int, int, Any]]:
    return [
        (idx, idx + 1, QuadraticCoupling(weight=base_weight + 0.03 * (idx % 3)))
        for idx in range(size - 1)
    ]


def build_case(scenario: str, seed: int, base_size: int) -> SyntheticCase:
    """Build one deterministic synthetic case for a scenario and seed."""
    assert scenario in SCENARIOS, f"unknown scenario: {scenario}"
    assert base_size >= 6, "base size must be at least 6"
    rng = np.random.default_rng(seed)

    if scenario in {"quadratic_chain", "mixed_gate_chain"}:
        modules, inputs = _quadratic_modules(rng, base_size, 0.25, 8.0, shuffle_curvatures=False)
        couplings = _chain_edges(base_size)
        constraints: Dict[str, Any] = {}
        family = "quadratic"
        if scenario == "mixed_gate_chain":
            constraints["delta_benefit"] = 0.08
            for idx in range(0, base_size - 1, 2):
                couplings.append((idx, idx + 1, GateBenefitCoupling(weight=0.25, delta_key="delta_benefit")))
            family = "quadratic_plus_linear_gate"
        return SyntheticCase(modules, couplings, constraints, inputs, "chain", family)

    if scenario == "quadratic_star":
        modules, inputs = _quadratic_modules(rng, base_size, 0.2, 12.0, shuffle_curvatures=True)
        couplings = [
            (0, idx, QuadraticCoupling(weight=0.08 + 0.02 * (idx % 4)))
            for idx in range(1, base_size)
        ]
        return SyntheticCase(modules, couplings, {}, inputs, "star", "quadratic")

    if scenario == "quadratic_dense":
        size = max(6, base_size // 2)
        modules, inputs = _quadratic_modules(rng, size, 0.25, 8.0, shuffle_curvatures=True)
        edges = {(idx, idx + 1) for idx in range(size - 1)}
        for i in range(size):
            for j in range(i + 2, size):
                if rng.random() < 0.55:
                    edges.add((i, j))
        couplings = [
            (i, j, QuadraticCoupling(weight=float(rng.uniform(0.04, 0.14))))
            for i, j in sorted(edges)
        ]
        return SyntheticCase(modules, couplings, {}, inputs, "dense_random", "quadratic")

    if scenario == "ill_conditioned_ring":
        size = 2 * base_size
        modules, inputs = _quadratic_modules(rng, size, 0.05, 20.0, shuffle_curvatures=True)
        couplings = _chain_edges(size, base_weight=0.06)
        couplings.append((size - 1, 0, QuadraticCoupling(weight=0.08)))
        return SyntheticCase(modules, couplings, {}, inputs, "ring", "ill_conditioned_quadratic")

    if scenario == "nonlinear_quartic":
        targets = rng.uniform(0.15, 0.85, size=base_size)
        quadratic = rng.permutation(np.geomspace(0.2, 4.0, num=base_size))
        quartic = rng.uniform(0.5, 2.5, size=base_size)
        modules = [
            QuarticWell(float(targets[i]), float(quadratic[i]), float(quartic[i]))
            for i in range(base_size)
        ]
        inputs = [float(value) for value in rng.uniform(0.0, 1.0, size=base_size)]
        couplings = _chain_edges(base_size, base_weight=0.08)
        return SyntheticCase(modules, couplings, {}, inputs, "chain", "convex_quartic")

    modules, inputs = _quadratic_modules(rng, base_size, 0.25, 8.0, shuffle_curvatures=True)
    couplings = _chain_edges(base_size, base_weight=0.04)
    for idx in range(0, base_size - 1, 2):
        inputs[idx] = float(rng.uniform(0.05, 0.25))
        inputs[idx + 1] = float(rng.uniform(0.75, 0.95))
        if idx % 4 == 0:
            couplings.append((idx, idx + 1, DirectedHingeCoupling(weight=0.35)))
        else:
            couplings.append(
                (idx, idx + 1, AsymmetricHingeCoupling(weight=0.3, alpha_i=0.8, beta_j=1.2))
            )
    return SyntheticCase(modules, couplings, {}, inputs, "chain_with_pair_hinges", "piecewise_quadratic")


def synthetic_objective_hessian(
    case: SyntheticCase,
    etas: Sequence[float],
    *,
    term_weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return the exact Hessian of a generated objective at ``etas``.

    The seven synthetic families contain quadratic and quartic local wells,
    quadratic couplings, piecewise-quadratic hinges, and linear gate-benefit
    terms. Hinge curvature is selected from the active branch at the supplied
    state. The Hessian is undefined exactly at a hinge kink, so this helper
    rejects that state instead of silently choosing a one-sided value.
    """
    values = np.asarray(etas, dtype=float)
    if values.shape != (len(case.modules),):
        raise ValueError("etas must contain one scalar per synthetic module")
    if not np.all(np.isfinite(values)):
        raise ValueError("etas must be finite")

    if term_weights is None:
        configured_weights = case.constraints.get("term_weights", {})
        weights = configured_weights if isinstance(configured_weights, Mapping) else {}
    else:
        weights = term_weights

    def objective_weight(kind: str, term: Any) -> float:
        key = f"{kind}:{term.__class__.__name__}"
        return float(weights.get(key, 1.0))

    hessian = np.zeros((values.size, values.size), dtype=float)
    for index, module in enumerate(case.modules):
        local_weight = objective_weight("local", module)
        if isinstance(module, QuadraticWell):
            local_curvature = float(module.curvature_value)
        elif isinstance(module, QuarticWell):
            diff = float(values[index]) - float(module.target)
            local_curvature = float(
                module.quadratic_curvature + 3.0 * module.quartic_strength * diff * diff
            )
        else:
            raise TypeError(
                f"unsupported synthetic local-energy type: {module.__class__.__name__}"
            )
        hessian[index, index] += local_weight * local_curvature

    for i, j, coupling in case.couplings:
        coupling_weight = objective_weight("coup", coupling)
        if isinstance(coupling, GateBenefitCoupling):
            # With caller-supplied benefit frozen, this term is linear in eta_i.
            continue
        if isinstance(coupling, QuadraticCoupling):
            scale_i = 1.0
            scale_j = 1.0
        elif isinstance(coupling, DirectedHingeCoupling):
            gap = float(values[j]) - float(values[i])
            if gap == 0.0:
                raise ValueError("synthetic Hessian is undefined at a directed-hinge kink")
            if gap < 0.0:
                continue
            scale_i = 1.0
            scale_j = 1.0
        elif isinstance(coupling, AsymmetricHingeCoupling):
            scale_i = float(coupling.alpha_i)
            scale_j = float(coupling.beta_j)
            gap = scale_j * float(values[j]) - scale_i * float(values[i])
            if gap == 0.0:
                raise ValueError("synthetic Hessian is undefined at an asymmetric-hinge kink")
            if gap < 0.0:
                continue
        else:
            raise TypeError(
                f"unsupported synthetic coupling type: {coupling.__class__.__name__}"
            )

        curvature_scale = 2.0 * coupling_weight * float(coupling.weight)
        hessian[i, i] += curvature_scale * scale_i * scale_i
        hessian[j, j] += curvature_scale * scale_j * scale_j
        cross_curvature = -curvature_scale * scale_i * scale_j
        hessian[i, j] += cross_curvature
        hessian[j, i] += cross_curvature

    return hessian


def noise_curvature_costs(
    coord: EnergyCoordinator,
    etas: List[float],
    full_hessian: np.ndarray,
    seed: int,
    samples: int,
) -> NoiseCurvatureCosts:
    """Measure requested and box-feasible curvature costs on identical draws.

    ``build_noise_vector`` returns the requested-magnitude direction. The
    coordinator then applies one uniform scalar to keep the complete noise
    vector inside the box. Both quantities are retained so directional quality
    and the realized pipeline are not conflated.
    """
    assert samples > 0, "noise cost samples must be positive"
    snapshot = coord.inspect_state(etas)
    grad = np.asarray(snapshot.gradient, dtype=float)
    diag = np.asarray(snapshot.precision_diagonal, dtype=float)
    hessian = np.asarray(full_hessian, dtype=float)
    if hessian.shape != (grad.size, grad.size):
        raise ValueError("full_hessian shape must match the synthetic state dimension")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("full_hessian must be finite")
    rng = np.random.default_rng(seed)
    diagonal_proxy_draws: List[float] = []
    full_hessian_draws: List[float] = []
    realized_diagonal_proxy_draws: List[float] = []
    realized_full_hessian_draws: List[float] = []
    box_scales: List[float] = []
    state = np.asarray(etas, dtype=float)
    for _ in range(samples):
        raw = rng.normal(0.0, 1.0, size=grad.shape)
        noise = coord.build_noise_vector(raw, grad)
        diagonal_proxy_draws.append(float(np.sum(diag * noise * noise)))
        full_hessian_draws.append(float(noise @ hessian @ noise))
        proposal = apply_box_feasible_noise(state, noise)
        realized_noise = proposal - state
        requested_norm = float(np.linalg.norm(noise))
        realized_norm = float(np.linalg.norm(realized_noise))
        box_scale = realized_norm / requested_norm if requested_norm > 0.0 else 1.0
        box_scale = max(0.0, min(1.0, box_scale))
        box_scales.append(float(box_scale))
        realized_diagonal_proxy_draws.append(
            float(np.sum(diag * realized_noise * realized_noise))
        )
        realized_full_hessian_draws.append(
            float(realized_noise @ hessian @ realized_noise)
        )
    return NoiseCurvatureCosts(
        diagonal_proxy_mean=float(np.mean(diagonal_proxy_draws)),
        diagonal_proxy_draws=diagonal_proxy_draws,
        full_hessian_mean=float(np.mean(full_hessian_draws)),
        full_hessian_draws=full_hessian_draws,
        realized_diagonal_proxy_mean=float(np.mean(realized_diagonal_proxy_draws)),
        realized_diagonal_proxy_draws=realized_diagonal_proxy_draws,
        realized_full_hessian_mean=float(np.mean(realized_full_hessian_draws)),
        realized_full_hessian_draws=realized_full_hessian_draws,
        box_scale_mean=float(np.mean(box_scales)),
        box_scaled_fraction=float(np.mean(np.asarray(box_scales) < 1.0 - 1e-12)),
    )


def run_one(
    mode: str,
    scenario: str,
    seed: int,
    steps: int,
    size: int,
    noise_magnitude: float,
    noise_cost_samples: int,
) -> Dict[str, Any]:
    """Run one paired scenario/mode trial."""
    assert mode in NOISE_MODES, f"unknown noise mode: {mode}"
    np.random.seed(seed)
    case = build_case(scenario, seed, size)
    coord = EnergyCoordinator(
        modules=case.modules,
        couplings=case.couplings,
        constraints=case.constraints,
        use_analytic=True,
        use_stiffness_updates=True,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        noise_mode=mode,
        noise_magnitude=noise_magnitude,
        precision_aware_noise_controller=(mode == "precision_orthogonal"),
        enable_orthogonal_noise=(mode != "isotropic"),
        assert_monotonic_energy=False,
        continue_after_rejection=True,
    )
    etas = coord.compute_etas(case.inputs)
    initial_snapshot = coord.inspect_state(etas)
    initial_energy = initial_snapshot.energy
    initial_hessian = synthetic_objective_hessian(
        case,
        etas,
        term_weights=initial_snapshot.term_weights,
    )
    noise_costs = noise_curvature_costs(
        coord,
        list(etas),
        initial_hessian,
        seed + 10_000,
        noise_cost_samples,
    )
    start = time.perf_counter()
    out = coord.relax_etas(etas, steps=steps)
    wall_time = time.perf_counter() - start
    final_energy = coord.inspect_state(out).energy
    metrics = coord.last_relaxation_metrics()
    rejected_steps = int(metrics["rejected_steps"])
    accepted_steps = int(metrics["accepted_steps"])
    attempted_steps = int(metrics["attempted_steps"])
    attempt_energies = [initial_energy, *[float(value) for value in metrics["attempt_energy_trace"]]]
    acceptance_rate = accepted_steps / attempted_steps if attempted_steps > 0 else 1.0
    return {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "steps": steps,
        **case.metadata(),
        "noise_magnitude": noise_magnitude,
        "noise_cost_samples": noise_cost_samples,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "acceptance_rate": acceptance_rate,
        "delta_f90_steps": delta_f90(attempt_energies),
        "energy_initial": initial_energy,
        "energy_final": final_energy,
        "energy_drop": initial_energy - final_energy,
        # Compatibility fields: these are diagonal-only curvature proxies.
        "noise_curvature_cost": noise_costs.diagonal_proxy_mean,
        "noise_curvature_cost_draws": ";".join(
            f"{value:.17g}" for value in noise_costs.diagonal_proxy_draws
        ),
        "noise_diagonal_curvature_proxy": noise_costs.diagonal_proxy_mean,
        "noise_diagonal_curvature_proxy_draws": ";".join(
            f"{value:.17g}" for value in noise_costs.diagonal_proxy_draws
        ),
        "noise_full_hessian_cost": noise_costs.full_hessian_mean,
        "noise_full_hessian_cost_draws": ";".join(
            f"{value:.17g}" for value in noise_costs.full_hessian_draws
        ),
        "noise_realized_diagonal_curvature_proxy": noise_costs.realized_diagonal_proxy_mean,
        "noise_realized_diagonal_curvature_proxy_draws": ";".join(
            f"{value:.17g}" for value in noise_costs.realized_diagonal_proxy_draws
        ),
        "noise_realized_full_hessian_cost": noise_costs.realized_full_hessian_mean,
        "noise_realized_full_hessian_cost_draws": ";".join(
            f"{value:.17g}" for value in noise_costs.realized_full_hessian_draws
        ),
        "noise_box_scale_mean": noise_costs.box_scale_mean,
        "noise_box_scaled_fraction": noise_costs.box_scaled_fraction,
        "wall_time_sec": wall_time,
    }


def paired_bootstrap_summary(
    rows: Sequence[Mapping[str, Any]],
    scenario: str,
    baseline_mode: str,
    *,
    comparison_mode: str = "precision_orthogonal",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260714,
) -> Dict[str, Any]:
    """Summarize a paired noise-cost effect with seed and draw uncertainty."""
    assert bootstrap_samples > 0, "bootstrap_samples must be positive"

    def by_seed(mode: str) -> Dict[int, Mapping[str, Any]]:
        return {
            int(row["seed"]): row
            for row in rows
            if str(row["scenario"]) == scenario and str(row["mode"]) == mode
        }

    baseline = by_seed(baseline_mode)
    comparison = by_seed(comparison_mode)
    seeds = sorted(baseline)
    assert seeds, f"no paired rows for {scenario}/{baseline_mode}"
    assert seeds == sorted(comparison), "paired modes must contain identical seeds"

    paired_rows = [baseline[seed] for seed in seeds] + [comparison[seed] for seed in seeds]
    if all("noise_realized_full_hessian_cost" in row for row in paired_rows):
        cost_field = "noise_realized_full_hessian_cost"
        draws_field = "noise_realized_full_hessian_cost_draws"
        cost_metric = "initial_box_feasible_full_hessian"
    elif all("noise_full_hessian_cost" in row for row in paired_rows):
        cost_field = "noise_full_hessian_cost"
        draws_field = "noise_full_hessian_cost_draws"
        cost_metric = "legacy_initial_requested_full_hessian"
    elif all("noise_diagonal_curvature_proxy" in row for row in paired_rows):
        cost_field = "noise_diagonal_curvature_proxy"
        draws_field = "noise_diagonal_curvature_proxy_draws"
        cost_metric = "initial_diagonal_curvature_proxy"
    else:
        # Compatibility with result rows generated before the proxy was relabeled.
        cost_field = "noise_curvature_cost"
        draws_field = "noise_curvature_cost_draws"
        cost_metric = "legacy_initial_diagonal_curvature_proxy"

    baseline_cost = np.asarray(
        [float(baseline[seed][cost_field]) for seed in seeds],
        dtype=float,
    )
    comparison_cost = np.asarray(
        [float(comparison[seed][cost_field]) for seed in seeds],
        dtype=float,
    )
    assert np.all(baseline_cost > 0.0), "relative reduction requires positive baseline costs"
    absolute_reduction = baseline_cost - comparison_cost
    relative_reduction = 1.0 - comparison_cost / baseline_cost

    def parsed_draws(row: Mapping[str, Any]) -> np.ndarray:
        encoded = str(row.get(draws_field, "")).strip()
        if not encoded:
            return np.asarray([float(row[cost_field])], dtype=float)
        return np.asarray([float(value) for value in encoded.split(";")], dtype=float)

    baseline_draws = np.stack([parsed_draws(baseline[seed]) for seed in seeds])
    comparison_draws = np.stack([parsed_draws(comparison[seed]) for seed in seeds])
    assert baseline_draws.shape == comparison_draws.shape, "paired modes must contain the same draw count"

    rng = np.random.default_rng(bootstrap_seed)
    seed_indices = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    seed_absolute_means = np.mean(absolute_reduction[seed_indices], axis=1)
    seed_relative_means = np.mean(relative_reduction[seed_indices], axis=1)
    seed_absolute_ci = np.quantile(seed_absolute_means, [0.025, 0.975])
    seed_relative_ci = np.quantile(seed_relative_means, [0.025, 0.975])

    hierarchical_absolute = np.empty(bootstrap_samples, dtype=float)
    hierarchical_relative = np.empty(bootstrap_samples, dtype=float)
    draws_per_trial = baseline_draws.shape[1]
    batch_size = 256
    for start in range(0, bootstrap_samples, batch_size):
        stop = min(start + batch_size, bootstrap_samples)
        batch = stop - start
        sampled_seeds = rng.integers(0, len(seeds), size=(batch, len(seeds)))
        sampled_draws = rng.integers(0, draws_per_trial, size=(batch, len(seeds), draws_per_trial))
        selected_baseline = baseline_draws[sampled_seeds[:, :, None], sampled_draws]
        selected_comparison = comparison_draws[sampled_seeds[:, :, None], sampled_draws]
        baseline_means = np.mean(selected_baseline, axis=2)
        comparison_means = np.mean(selected_comparison, axis=2)
        hierarchical_absolute[start:stop] = np.mean(baseline_means - comparison_means, axis=1)
        hierarchical_relative[start:stop] = np.mean(1.0 - comparison_means / baseline_means, axis=1)
    absolute_ci = np.quantile(hierarchical_absolute, [0.025, 0.975])
    relative_ci = np.quantile(hierarchical_relative, [0.025, 0.975])

    first = baseline[seeds[0]]
    return {
        "scenario": scenario,
        "baseline_mode": baseline_mode,
        "comparison_mode": comparison_mode,
        "trials": len(seeds),
        "noise_cost_samples": int(first["noise_cost_samples"]),
        "draws_per_trial": draws_per_trial,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_method": "paired_hierarchical_seed_draw",
        "noise_cost_metric": cost_metric,
        "noise_cost_field": cost_field,
        "baseline_cost_mean": float(np.mean(baseline_cost)),
        "comparison_cost_mean": float(np.mean(comparison_cost)),
        "absolute_reduction_mean": float(np.mean(absolute_reduction)),
        "absolute_ci_low": float(absolute_ci[0]),
        "absolute_ci_high": float(absolute_ci[1]),
        "relative_reduction_mean": float(np.mean(relative_reduction)),
        "relative_ci_low": float(relative_ci[0]),
        "relative_ci_high": float(relative_ci[1]),
        "seed_absolute_ci_low": float(seed_absolute_ci[0]),
        "seed_absolute_ci_high": float(seed_absolute_ci[1]),
        "seed_relative_ci_low": float(seed_relative_ci[0]),
        "seed_relative_ci_high": float(seed_relative_ci[1]),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    assert rows, "rows required"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small validation sweep.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--size", type=int, default=12, help="Base problem size; selected families derive smaller or larger sizes.")
    parser.add_argument("--noise-magnitude", type=float, default=0.02)
    parser.add_argument("--noise-cost-samples", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--output", type=Path, default=Path("logs/pson_noise_ablation.csv"))
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = 3 if args.quick else int(args.trials)
    steps = 25 if args.quick else int(args.steps)
    scenarios = [str(value) for value in args.scenarios]
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for mode in NOISE_MODES:
            for seed in range(trials):
                rows.append(
                    run_one(
                        mode,
                        scenario,
                        seed,
                        steps,
                        int(args.size),
                        float(args.noise_magnitude),
                        int(args.noise_cost_samples),
                    )
                )

    _write_rows(args.output, rows)
    summary_output = args.summary_output or args.output.with_name(f"{args.output.stem}_summary.csv")
    summaries: List[Dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for baseline_index, baseline_mode in enumerate(("isotropic", "orthogonal")):
            summaries.append(
                paired_bootstrap_summary(
                    rows,
                    scenario,
                    baseline_mode,
                    bootstrap_samples=int(args.bootstrap_samples),
                    bootstrap_seed=int(args.bootstrap_seed) + 10 * scenario_index + baseline_index,
                )
            )
    _write_rows(summary_output, summaries)

    print(f"wrote {len(rows)} trial rows to {args.output}")
    print(f"wrote {len(summaries)} paired bootstrap summaries to {summary_output}")
    for scenario in scenarios:
        for mode in NOISE_MODES:
            subset = [row for row in rows if row["scenario"] == scenario and row["mode"] == mode]
            mean_drop = float(np.mean([row["energy_drop"] for row in subset]))
            mean_accept = float(np.mean([row["acceptance_rate"] for row in subset]))
            mean_diagonal_proxy = float(
                np.mean([row["noise_realized_diagonal_curvature_proxy"] for row in subset])
            )
            mean_full_hessian_cost = float(
                np.mean([row["noise_realized_full_hessian_cost"] for row in subset])
            )
            mean_box_scaled_fraction = float(
                np.mean([row["noise_box_scaled_fraction"] for row in subset])
            )
            print(
                f"{scenario:24s} {mode:20s} "
                f"drop={mean_drop:.6f} accept={mean_accept:.3f} "
                f"realized_diagonal_curvature_proxy={mean_diagonal_proxy:.6e} "
                f"realized_full_hessian_cost={mean_full_hessian_cost:.6e} "
                f"box_scaled_fraction={mean_box_scaled_fraction:.3f}"
            )
        for summary in (row for row in summaries if row["scenario"] == scenario):
            print(
                f"{scenario:24s} precision_orthogonal vs {summary['baseline_mode']:10s} "
                f"paired_{summary['noise_cost_metric']}_cost_reduction="
                f"{100.0 * float(summary['relative_reduction_mean']):.2f}% "
                f"95% bootstrap CI "
                f"[{100.0 * float(summary['relative_ci_low']):.2f}%, "
                f"{100.0 * float(summary['relative_ci_high']):.2f}%]"
            )


if __name__ == "__main__":
    main()
