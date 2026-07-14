"""Measure coordinator runtime and memory as graph size increases.

The fitted exponents are descriptive summaries of this benchmark matrix. They
do not establish implementation-independent complexity bounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from experiments.ablate_pson_noise import QuadraticWell


def build_scaling_case(
    size: int,
    edge_factor: int,
    seed: int,
) -> Tuple[List[QuadraticWell], List[Tuple[int, int, Any]], List[float]]:
    """Build a connected quadratic graph with a controlled edge budget."""
    if size < 2:
        raise ValueError("size must be at least two")
    if edge_factor < 1:
        raise ValueError("edge_factor must be positive")
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0.1, 0.9, size=size)
    curvatures = rng.permutation(np.geomspace(0.25, 8.0, num=size))
    modules = [QuadraticWell(float(targets[i]), float(curvatures[i])) for i in range(size)]
    inputs = [float(value) for value in rng.uniform(0.0, 1.0, size=size)]

    target_edges = min(size * edge_factor, size * (size - 1) // 2)
    edges = {(index, index + 1) for index in range(size - 1)}
    while len(edges) < target_edges:
        i, j = sorted(rng.choice(size, size=2, replace=False).tolist())
        edges.add((int(i), int(j)))
    couplings = [
        (i, j, QuadraticCoupling(weight=float(rng.uniform(0.02, 0.12))))
        for i, j in sorted(edges)
    ]
    return modules, couplings, inputs


def _coordinator(modules: List[QuadraticWell], couplings: List[Tuple[int, int, Any]]) -> EnergyCoordinator:
    return EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={},
        use_analytic=True,
        use_vectorized_quadratic=True,
        use_precision_preconditioning=True,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        noise_mode="none",
        assert_monotonic_energy=True,
    )


def _statistics(values: Sequence[float], prefix: str) -> Dict[str, float]:
    samples = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean_sec": float(np.mean(samples)),
        f"{prefix}_std_sec": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        f"{prefix}_q25_sec": float(np.quantile(samples, 0.25)),
        f"{prefix}_median_sec": float(np.median(samples)),
        f"{prefix}_q75_sec": float(np.quantile(samples, 0.75)),
        f"{prefix}_p95_sec": float(np.quantile(samples, 0.95)),
    }


def _blas_name() -> str:
    config = getattr(np.__config__, "CONFIG", {})
    if not isinstance(config, dict):
        return "unknown"
    dependencies = config.get("Build Dependencies", {})
    blas = dependencies.get("blas", {}) if isinstance(dependencies, dict) else {}
    return str(blas.get("name", "unknown")) if isinstance(blas, dict) else "unknown"


def environment_metadata(label: str) -> Dict[str, str]:
    """Return enough runtime context to distinguish benchmark environments."""
    return {
        "environment_label": label,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "blas_backend": _blas_name(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unspecified"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "unspecified"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unspecified"),
    }


def run_scaling_case(
    size: int,
    edge_factor: int,
    repeats: int,
    steps: int,
    seed: int,
    warmups: int = 1,
    environment_label: str = "local",
) -> Dict[str, Any]:
    if repeats < 1 or steps < 1 or warmups < 0:
        raise ValueError("repeats and steps must be positive; warmups must be non-negative")
    modules, couplings, inputs = build_scaling_case(size, edge_factor, seed)

    for _ in range(warmups):
        warmup_coordinator = _coordinator(modules, couplings)
        warmup_etas = warmup_coordinator.compute_etas(inputs)
        warmup_coordinator.inspect_state(warmup_etas)
        warmup_coordinator.relax_etas(list(warmup_etas), steps=steps)

    construction_times: List[float] = []
    inspect_times: List[float] = []
    relax_times: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        coordinator = _coordinator(modules, couplings)
        construction_times.append(time.perf_counter() - start)
        etas = coordinator.compute_etas(inputs)

        start = time.perf_counter()
        snapshot = coordinator.inspect_state(etas)
        inspect_times.append(time.perf_counter() - start)
        if snapshot.lipschitz_bound <= 0.0:
            raise RuntimeError("scaling case produced a non-positive Lipschitz estimate")

        start = time.perf_counter()
        coordinator.relax_etas(list(etas), steps=steps)
        relax_times.append(time.perf_counter() - start)

    tracemalloc.start()
    memory_coordinator = _coordinator(modules, couplings)
    memory_etas = memory_coordinator.compute_etas(inputs)
    memory_coordinator.inspect_state(memory_etas)
    memory_coordinator.relax_etas(list(memory_etas), steps=steps)
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    row: Dict[str, Any] = {
        "size": size,
        "edge_factor": edge_factor,
        "edge_count": len(couplings),
        "repeats": repeats,
        "warmups": warmups,
        "relaxation_steps": steps,
        "peak_memory_bytes": int(peak_memory_bytes),
        "construction_samples_json": json.dumps(construction_times, separators=(",", ":")),
        "inspect_samples_json": json.dumps(inspect_times, separators=(",", ":")),
        "relax_samples_json": json.dumps(relax_times, separators=(",", ":")),
    }
    row.update(_statistics(construction_times, "construction"))
    row.update(_statistics(inspect_times, "inspect"))
    row.update(_statistics(relax_times, "relax"))
    row["relax_mean_sec_per_step"] = float(row["relax_mean_sec"]) / steps
    row["relax_median_sec_per_step"] = float(row["relax_median_sec"]) / steps
    row.update(environment_metadata(environment_label))
    return row


def _run_case_subprocess(
    size: int,
    edge_factor: int,
    repeats: int,
    steps: int,
    seed: int,
    warmups: int,
    environment_label: str,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "experiments.benchmark_scaling",
        "--worker-case",
        str(size),
        str(edge_factor),
        "--repeats",
        str(repeats),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--warmups",
        str(warmups),
        "--environment-label",
        environment_label,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return dict(json.loads(completed.stdout.strip().splitlines()[-1]))


def run_matrix(
    sizes: Sequence[int],
    edge_factors: Sequence[int],
    repeats: int,
    steps: int,
    seed: int,
    warmups: int = 1,
    isolate_processes: bool = False,
    environment_label: str = "local",
) -> List[Dict[str, Any]]:
    cases = [(size, edge_factor) for size in sizes for edge_factor in edge_factors]
    random.Random(seed).shuffle(cases)
    rows: List[Dict[str, Any]] = []
    for size, edge_factor in cases:
        case_seed = seed + 100 * size + edge_factor
        if isolate_processes:
            row = _run_case_subprocess(
                size, edge_factor, repeats, steps, case_seed, warmups, environment_label
            )
        else:
            row = run_scaling_case(
                size,
                edge_factor,
                repeats,
                steps,
                case_seed,
                warmups=warmups,
                environment_label=environment_label,
            )
        rows.append(row)
    return sorted(rows, key=lambda row: (int(row["size"]), int(row["edge_factor"])))


def fit_scaling_model(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit a descriptive log-linear timing model to the measured matrix."""
    if len(rows) < 3:
        return {"status": "insufficient_rows", "model_scope": "descriptive_only"}
    design = np.asarray(
        [[1.0, np.log(float(row["size"])), np.log(float(row["edge_count"]))] for row in rows],
        dtype=float,
    )
    response = np.log(
        np.asarray([float(row["relax_median_sec_per_step"]) for row in rows], dtype=float)
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coefficients
    residual_sum = float(np.sum((response - fitted) ** 2))
    total_sum = float(np.sum((response - np.mean(response)) ** 2))
    return {
        "status": "fitted",
        "model_scope": "descriptive_only",
        "response": "relax_median_sec_per_step",
        "row_count": len(rows),
        "intercept": float(coefficients[0]),
        "size_exponent": float(coefficients[1]),
        "edge_count_exponent": float(coefficients[2]),
        "r_squared": float(1.0 - residual_sum / total_sum) if total_sum > 0.0 else 1.0,
    }


def _write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--edge-factors", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--environment-label", default="local")
    parser.add_argument("--output", type=Path, default=Path("logs/scaling_benchmark.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("logs/scaling_model.json"))
    parser.add_argument(
        "--isolate-processes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each size and edge-factor case in a fresh Python process.",
    )
    parser.add_argument("--worker-case", type=int, nargs=2, metavar=("SIZE", "EDGE_FACTOR"))
    args = parser.parse_args()

    if args.worker_case is not None:
        size, edge_factor = args.worker_case
        row = run_scaling_case(
            size,
            edge_factor,
            int(args.repeats),
            int(args.steps),
            int(args.seed),
            warmups=int(args.warmups),
            environment_label=str(args.environment_label),
        )
        print(json.dumps(row, sort_keys=True))
        return

    rows = run_matrix(
        args.sizes,
        args.edge_factors,
        int(args.repeats),
        int(args.steps),
        int(args.seed),
        warmups=int(args.warmups),
        isolate_processes=bool(args.isolate_processes),
        environment_label=str(args.environment_label),
    )
    _write_rows(args.output, rows)
    model = fit_scaling_model(rows)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {len(rows)} scaling rows to {args.output}")
    print(f"wrote descriptive scaling model to {args.model_output}")
    for row in rows:
        print(
            f"n={row['size']:4d} edges={row['edge_count']:5d} "
            f"inspect_median={float(row['inspect_median_sec']) * 1000.0:8.3f}ms "
            f"relax_step_median={float(row['relax_median_sec_per_step']) * 1000.0:8.3f}ms "
            f"peak={int(row['peak_memory_bytes']) / (1024.0 * 1024.0):7.2f}MiB"
        )


if __name__ == "__main__":
    main()
