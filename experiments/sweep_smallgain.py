from __future__ import annotations

import argparse
import subprocess
from itertools import product
from typing import Iterable, List, Tuple

import polars as pl


def build_jobs(
    rhos: Iterable[float],
    dws: Iterable[float],
    scenarios: Iterable[str],
    steps: int,
    dense_size: int,
) -> List[Tuple[str, float, float, int, int]]:
    """
    Build sweep jobs as tuples: (scenario, rho, dw, steps, dense_size).
    """
    jobs: List[Tuple[str, float, float, int, int]] = []
    for scen, rho, dw in product(scenarios, rhos, dws):
        jobs.append((scen, float(rho), float(dw), int(steps), int(dense_size)))
    return jobs


def run_job(
    scenario: str,
    rho: float,
    dw: float,
    steps: int,
    dense_size: int,
    run_id_prefix: str,
    log_budget: bool,
) -> None:
    run_id = f"{run_id_prefix}_{scenario}_rho{rho}_dw{dw}"
    cmd = [
        "python",
        "-m",
        "experiments.benchmark_delta_f90",
        "--configs",
        "smallgain",
        "--scenario",
        scenario,
        "--steps",
        str(int(steps)),
        "--run_id",
        run_id,
        "--sg_rho",
        str(float(rho)),
        "--sg_dw",
        str(float(dw)),
    ]
    if scenario == "dense":
        cmd += ["--dense_size", str(int(dense_size))]
    if log_budget:
        cmd += ["--log_budget"]
    # Use uv when executed via `uv run`; here we assume the script is called with uv, so python resolves correctly.
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize(run_id_prefix: str, out_csv: str) -> str:
    """
    Summarize results across runs that start with run_id_prefix.
    Returns the path written to.
    """
    path = "logs/benchmark_delta_f90.csv"
    df = pl.read_csv(path)
    df = df.filter(pl.col("run_id").str.starts_with(run_id_prefix))
    # Keep key KPI columns
    cols = [
        "run_id",
        "config",
        "delta_f90_steps",
        "wall_time_sec",
        "energy_final",
        "total_backtracks",
        "sg_rho",
        "sg_dw",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df.select(cols)
    out.write_csv(f"plots/{out_csv}")
    return f"plots/{out_csv}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--dws", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--scenarios", type=str, nargs="+", choices=["baseline", "dense"], default=["baseline", "dense"])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--dense_size", type=int, default=16)
    parser.add_argument("--run_id_prefix", type=str, default="sgsweep")
    parser.add_argument("--log_budget", action="store_true")
    parser.add_argument("--summary_out", type=str, default="df90_smallgain_sweep_summary.csv")
    parser.add_argument("--quick", action="store_true", help="Quick mode: steps=40, dense_size=8 override for a faster sweep")
    args = parser.parse_args()

    steps = 40 if args.quick else int(args.steps)
    dense_size = 8 if args.quick else int(args.dense_size)

    jobs = build_jobs(args.rhos, args.dws, args.scenarios, steps, dense_size)
    for scenario, rho, dw, st, ds in jobs:
        run_job(scenario, rho, dw, st, ds, args.run_id_prefix, args.log_budget)
    path = summarize(args.run_id_prefix, args.summary_out)
    print(f"Wrote summary to {path}")


if __name__ == "__main__":
    main()


