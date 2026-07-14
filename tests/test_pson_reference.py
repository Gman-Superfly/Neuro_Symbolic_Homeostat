from __future__ import annotations

import math

import numpy as np

from experiments.validate_pson_reference import analytic_costs, run_reference


def test_axis_aligned_reference_has_expected_arithmetic_and_harmonic_costs() -> None:
    costs = analytic_costs(np.asarray([2.0, 4.0, 16.0]), magnitude=1.0, precision_epsilon=0.0)

    assert math.isclose(costs["isotropic"], 22.0 / 3.0)
    assert math.isclose(costs["orthogonal"], 10.0)
    assert math.isclose(costs["precision_orthogonal"], 6.4)


def test_monte_carlo_noise_pipeline_matches_closed_form_reference() -> None:
    rows = run_reference(samples=20_000, seed=17, magnitude=0.02)

    assert {row["mode"] for row in rows} == {"isotropic", "orthogonal", "precision_orthogonal"}
    assert all(float(row["relative_error"]) < 0.03 for row in rows)
