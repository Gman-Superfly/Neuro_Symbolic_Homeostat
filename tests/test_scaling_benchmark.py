from __future__ import annotations

import math

from experiments.benchmark_scaling import (
    build_scaling_case,
    fit_scaling_model,
    run_matrix,
    run_scaling_case,
)


def test_scaling_case_respects_edge_budget_and_connectivity_floor() -> None:
    modules, couplings, inputs = build_scaling_case(size=16, edge_factor=4, seed=3)

    assert len(modules) == len(inputs) == 16
    assert len(couplings) == 64
    assert {(i, j) for i, j, _ in couplings} >= {(index, index + 1) for index in range(15)}


def test_scaling_smoke_result_is_finite() -> None:
    row = run_scaling_case(size=8, edge_factor=2, repeats=1, steps=2, seed=4)

    assert row["edge_count"] == 16
    assert math.isfinite(float(row["inspect_mean_sec"]))
    assert math.isfinite(float(row["relax_mean_sec_per_step"]))
    assert float(row["inspect_q25_sec"]) <= float(row["inspect_median_sec"]) <= float(row["inspect_q75_sec"])
    assert int(row["peak_memory_bytes"]) > 0
    assert row["python_version"]


def test_scaling_subprocess_isolation_and_descriptive_model() -> None:
    rows = run_matrix(
        sizes=[4, 8],
        edge_factors=[1, 2],
        repeats=1,
        steps=1,
        seed=5,
        warmups=0,
        isolate_processes=True,
        environment_label="test",
    )

    assert len(rows) == 4
    assert all(row["environment_label"] == "test" for row in rows)
    model = fit_scaling_model(rows)
    assert model["status"] == "fitted"
    assert model["model_scope"] == "descriptive_only"
    assert math.isfinite(float(model["size_exponent"]))
    assert math.isfinite(float(model["edge_count_exponent"]))
