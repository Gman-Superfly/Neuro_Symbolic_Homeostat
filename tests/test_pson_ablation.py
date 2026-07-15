from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from core.couplings import AsymmetricHingeCoupling, DirectedHingeCoupling
from core.energy import total_energy
from experiments.ablate_pson_noise import (
    NOISE_MODES,
    SCENARIOS,
    QuarticWell,
    build_case,
    noise_curvature_costs,
    paired_bootstrap_summary,
    run_one,
    synthetic_objective_hessian,
)


class _FixedNoiseCoordinator:
    def inspect_state(self, etas: list[float]) -> SimpleNamespace:
        del etas
        return SimpleNamespace(gradient=(1.0, 0.0), precision_diagonal=(1.0, 1.0))

    def build_noise_vector(self, raw: np.ndarray, grad: np.ndarray) -> np.ndarray:
        del raw, grad
        return np.asarray([0.2, 0.2], dtype=float)


def _finite_difference_hessian(scenario: str, seed: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    case = build_case(scenario, seed=seed, base_size=size)
    state = np.asarray(case.inputs, dtype=float)
    step = 1e-4

    def energy(values: np.ndarray) -> float:
        return total_energy(
            values.tolist(),
            case.modules,
            case.couplings,
            case.constraints,
        )

    hessian = np.zeros((state.size, state.size), dtype=float)
    base_energy = energy(state)
    for i in range(state.size):
        direction_i = np.zeros_like(state)
        direction_i[i] = step
        hessian[i, i] = (
            energy(state + direction_i) - 2.0 * base_energy + energy(state - direction_i)
        ) / (step * step)
        for j in range(i + 1, state.size):
            direction_j = np.zeros_like(state)
            direction_j[j] = step
            mixed = (
                energy(state + direction_i + direction_j)
                - energy(state + direction_i - direction_j)
                - energy(state - direction_i + direction_j)
                + energy(state - direction_i - direction_j)
            ) / (4.0 * step * step)
            hessian[i, j] = mixed
            hessian[j, i] = mixed
    return synthetic_objective_hessian(case, state), hessian


def test_synthetic_families_cover_distinct_structures() -> None:
    cases = {scenario: build_case(scenario, seed=3, base_size=12) for scenario in SCENARIOS}

    assert {case.topology for case in cases.values()} >= {
        "chain",
        "star",
        "dense_random",
        "ring",
        "chain_with_pair_hinges",
    }
    assert {len(case.modules) for case in cases.values()} == {6, 12, 24}
    assert cases["ill_conditioned_ring"].metadata()["initial_curvature_ratio"] >= 300.0
    assert any(isinstance(module, QuarticWell) for module in cases["nonlinear_quartic"].modules)
    assert any(
        isinstance(coupling, (DirectedHingeCoupling, AsymmetricHingeCoupling))
        for _, _, coupling in cases["active_hinges"].couplings
    )


def test_quartic_family_reports_state_dependent_curvature() -> None:
    case = build_case("nonlinear_quartic", seed=4, base_size=12)
    module = case.modules[0]
    assert isinstance(module, QuarticWell)

    at_target = module.curvature(module.target)
    away_from_target = module.curvature(module.target + 0.4)

    assert away_from_target > at_target


def test_exact_synthetic_hessian_matches_all_seven_generated_objectives() -> None:
    for scenario in SCENARIOS:
        exact, finite_difference = _finite_difference_hessian(scenario, seed=5, size=12)

        assert np.allclose(exact, exact.T, rtol=0.0, atol=1e-12), scenario
        assert np.allclose(exact, finite_difference, rtol=2e-5, atol=2e-5), scenario


def test_gate_benefit_is_linear_and_adds_no_hessian_curvature() -> None:
    quadratic = build_case("quadratic_chain", seed=7, base_size=12)
    mixed_gate = build_case("mixed_gate_chain", seed=7, base_size=12)

    quadratic_hessian = synthetic_objective_hessian(quadratic, quadratic.inputs)
    mixed_gate_hessian = synthetic_objective_hessian(mixed_gate, mixed_gate.inputs)

    assert np.array_equal(mixed_gate_hessian, quadratic_hessian)


def test_noise_costs_record_the_realized_uniform_box_scaling() -> None:
    costs = noise_curvature_costs(
        _FixedNoiseCoordinator(),  # type: ignore[arg-type]
        [0.95, 0.5],
        np.eye(2, dtype=float),
        seed=1,
        samples=1,
    )

    assert math.isclose(costs.full_hessian_mean, 0.08)
    assert math.isclose(costs.realized_full_hessian_mean, 0.005)
    assert math.isclose(costs.box_scale_mean, 0.25)
    assert costs.box_scaled_fraction == 1.0


def test_paired_bootstrap_preserves_seed_pairing() -> None:
    rows = []
    for seed, baseline_cost in enumerate((2.0, 4.0, 6.0, 8.0)):
        common = {"scenario": "known", "seed": seed, "noise_cost_samples": 32}
        baseline_draws = ";".join(str(baseline_cost * scale) for scale in (0.8, 1.0, 1.2))
        comparison_draws = ";".join(str(0.5 * baseline_cost * scale) for scale in (0.8, 1.0, 1.2))
        rows.append(
            {
                **common,
                "mode": "isotropic",
                "noise_curvature_cost": baseline_cost,
                "noise_curvature_cost_draws": baseline_draws,
            }
        )
        rows.append(
            {
                **common,
                "mode": "precision_orthogonal",
                "noise_curvature_cost": 0.5 * baseline_cost,
                "noise_curvature_cost_draws": comparison_draws,
            }
        )

    summary = paired_bootstrap_summary(
        rows,
        "known",
        "isotropic",
        bootstrap_samples=500,
        bootstrap_seed=11,
    )

    assert summary["trials"] == 4
    assert summary["draws_per_trial"] == 3
    assert summary["bootstrap_method"] == "paired_hierarchical_seed_draw"
    assert summary["noise_cost_metric"] == "legacy_initial_diagonal_curvature_proxy"
    assert math.isclose(float(summary["relative_reduction_mean"]), 0.5)
    assert math.isclose(float(summary["relative_ci_low"]), 0.5)
    assert math.isclose(float(summary["relative_ci_high"]), 0.5)
    assert math.isclose(float(summary["absolute_reduction_mean"]), 2.5)


def test_ablation_reports_exact_full_hessian_cost_for_all_families() -> None:
    for scenario in SCENARIOS:
        row = run_one(
            "orthogonal",
            scenario,
            seed=2,
            steps=1,
            size=12,
            noise_magnitude=0.02,
            noise_cost_samples=3,
        )

        assert math.isfinite(float(row["noise_full_hessian_cost"])), scenario
        assert float(row["noise_full_hessian_cost"]) >= 0.0, scenario
        assert row["noise_curvature_cost"] == row["noise_diagonal_curvature_proxy"]
        assert (
            row["noise_curvature_cost_draws"]
            == row["noise_diagonal_curvature_proxy_draws"]
        )
        assert len(str(row["noise_full_hessian_cost_draws"]).split(";")) == 3
        assert len(str(row["noise_realized_full_hessian_cost_draws"]).split(";")) == 3
        assert 0.0 <= float(row["noise_box_scale_mean"]) <= 1.0
        assert 0.0 <= float(row["noise_box_scaled_fraction"]) <= 1.0


def test_representative_ablation_runs_are_finite_and_paired() -> None:
    scenarios = ("quadratic_dense", "nonlinear_quartic", "active_hinges")
    for scenario in scenarios:
        rows = [
            run_one(mode, scenario, seed=2, steps=3, size=12, noise_magnitude=0.02, noise_cost_samples=3)
            for mode in NOISE_MODES
        ]
        assert {int(row["size"]) for row in rows} == {int(rows[0]["size"])}
        assert {int(row["coupling_count"]) for row in rows} == {int(rows[0]["coupling_count"])}
        for row in rows:
            assert math.isfinite(float(row["energy_final"]))
            assert math.isfinite(float(row["noise_curvature_cost"]))
            assert math.isfinite(float(row["noise_full_hessian_cost"]))
            assert int(row["accepted_steps"]) + int(row["rejected_steps"]) == 3
