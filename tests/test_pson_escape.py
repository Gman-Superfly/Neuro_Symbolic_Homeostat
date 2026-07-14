from __future__ import annotations

import math

from experiments.benchmark_pson_escape import paired_escape_summary, run_trial
from experiments.benchmark_pson_escape import AsymmetricDoubleWell


def test_double_well_derivatives_match_finite_differences() -> None:
    module = AsymmetricDoubleWell()
    step = 1e-5
    for eta in (0.2, 0.35, 0.5, 0.8):
        numeric_grad = (module.local_energy(eta + step, {}) - module.local_energy(eta - step, {})) / (2.0 * step)
        numeric_second = (
            module.d_local_energy_d_eta(eta + step, {}) - module.d_local_energy_d_eta(eta - step, {})
        ) / (2.0 * step)

        assert math.isclose(module.d_local_energy_d_eta(eta, {}), numeric_grad, rel_tol=0.0, abs_tol=1e-7)
        assert math.isclose(module.curvature(eta), abs(numeric_second), rel_tol=1e-7, abs_tol=1e-7)


def test_no_noise_remains_in_left_well() -> None:
    row = run_trial("none", seed=0, steps=10, dimension=8, noise_magnitude=0.55)

    assert row["escaped"] == 0
    assert row["final_escape_coordinate"] == 0.2
    assert row["energy_drop"] == 0.0


def test_precision_noise_can_escape_controlled_well() -> None:
    row = run_trial("precision_orthogonal", seed=3, steps=10, dimension=8, noise_magnitude=0.55)

    assert row["escaped"] == 1
    assert float(row["maximum_escape_coordinate"]) >= 0.6
    assert float(row["energy_drop"]) > 0.0


def test_escape_summary_preserves_pairing() -> None:
    rows = []
    for seed in range(4):
        rows.append({"mode": "none", "seed": seed, "escaped": 0})
        rows.append({"mode": "precision_orthogonal", "seed": seed, "escaped": 1})
    summary = paired_escape_summary(
        rows,
        "precision_orthogonal",
        "none",
        bootstrap_samples=100,
        bootstrap_seed=3,
    )

    assert summary["paired_escape_rate_difference"] == 1.0
    assert summary["difference_ci_low"] == 1.0
    assert summary["difference_ci_high"] == 1.0
