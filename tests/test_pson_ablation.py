from __future__ import annotations

import math

from core.couplings import AsymmetricHingeCoupling, DirectedHingeCoupling
from experiments.ablate_pson_noise import (
    NOISE_MODES,
    SCENARIOS,
    QuarticWell,
    build_case,
    paired_bootstrap_summary,
    run_one,
)


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
    assert math.isclose(float(summary["relative_reduction_mean"]), 0.5)
    assert math.isclose(float(summary["relative_ci_low"]), 0.5)
    assert math.isclose(float(summary["relative_ci_high"]), 0.5)
    assert math.isclose(float(summary["absolute_reduction_mean"]), 2.5)


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
            assert int(row["accepted_steps"]) + int(row["rejected_steps"]) == 3
