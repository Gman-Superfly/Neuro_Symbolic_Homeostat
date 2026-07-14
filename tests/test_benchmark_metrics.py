from __future__ import annotations

from experiments.benchmark_delta_f90 import run_config


def test_benchmark_separates_reference_and_adaptive_objectives() -> None:
    result = run_config(
        name="smallgain_test",
        coord_kwargs={
            "use_analytic": True,
            "_adapter": "smallgain",
            "line_search": True,
            "normalize_grads": True,
            "noise_mode": "none",
        },
        steps=4,
        scenario="baseline",
        dense_size=8,
        log_budget=False,
        budget_name="unused",
        run_id="test",
    )

    assert result["energy_final"] == result["reference_energy_final"]
    assert result["accepted_steps"] >= 1
    assert result["objective_versions"] >= 1
    assert "adaptive_energy_final" in result
    assert "adaptive_delta_f90_steps" in result
