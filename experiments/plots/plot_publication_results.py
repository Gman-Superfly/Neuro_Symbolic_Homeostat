"""Generate publication figures from recorded CSV artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "quadratic_chain": "Quadratic chain",
    "mixed_gate_chain": "Mixed gate chain",
    "quadratic_star": "Quadratic star",
    "quadratic_dense": "Quadratic dense",
    "ill_conditioned_ring": "Ill-conditioned ring",
    "nonlinear_quartic": "Nonlinear quartic",
    "active_hinges": "Active hinges",
}


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_cost_reductions(summary_path: Path, output_path: Path) -> None:
    rows = _read(summary_path)
    scenarios = list(LABELS)
    y = np.arange(len(scenarios), dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, baseline, color, label in (
        (-0.16, "isotropic", "#2f6f9f", "vs isotropic"),
        (0.16, "orthogonal", "#d57a2a", "vs orthogonal"),
    ):
        selected = {row["scenario"]: row for row in rows if row["baseline_mode"] == baseline}
        means = np.asarray([100.0 * float(selected[name]["relative_reduction_mean"]) for name in scenarios])
        lows = np.asarray([100.0 * float(selected[name]["relative_ci_low"]) for name in scenarios])
        highs = np.asarray([100.0 * float(selected[name]["relative_ci_high"]) for name in scenarios])
        ax.errorbar(
            means,
            y + offset,
            xerr=np.vstack((means - lows, highs - means)),
            fmt="o",
            capsize=3,
            color=color,
            label=label,
        )
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, [LABELS[name] for name in scenarios])
    ax.set_xlabel("Paired realized full-Hessian cost reduction (%)")
    ax.set_title("Precision-orthogonal noise across generated families")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_escape_rates(trials_path: Path, summary_path: Path, output_path: Path) -> None:
    trials = _read(trials_path)
    summary = _read(summary_path)
    modes = ["none", "isotropic", "orthogonal", "precision_orthogonal"]
    labels = ["No noise", "Isotropic", "Orthogonal", "Precision-orthogonal"]
    rates = [np.mean([float(row["escaped"]) for row in trials if row["mode"] == mode]) for mode in modes]
    precision = next(
        row for row in summary if row["mode"] == "precision_orthogonal" and row["baseline_mode"] == "none"
    )
    error_low = [0.0, 0.0, 0.0, rates[-1] - float(precision["difference_ci_low"])]
    error_high = [0.0, 0.0, 0.0, float(precision["difference_ci_high"]) - rates[-1]]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = ["#7a7a7a", "#2f6f9f", "#d57a2a", "#3c8c5a"]
    ax.bar(labels, np.asarray(rates) * 100.0, color=colors, width=0.68)
    ax.errorbar(
        np.arange(4),
        np.asarray(rates) * 100.0,
        yerr=np.asarray([error_low, error_high]) * 100.0,
        fmt="none",
        color="#222222",
        capsize=4,
    )
    ax.set_ylabel("Escape rate (%)")
    ax.set_title("Controlled anisotropic double-well escape")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scaling(scaling_path: Path, output_path: Path) -> None:
    rows = _read(scaling_path)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    colors = {16: "#2f6f9f", 64: "#d57a2a", 256: "#3c8c5a"}
    for size in sorted({int(row["size"]) for row in rows}):
        selected = sorted((row for row in rows if int(row["size"]) == size), key=lambda row: int(row["edge_count"]))
        x_values = [int(row["edge_count"]) for row in selected]
        medians = [1000.0 * float(row["relax_median_sec_per_step"]) for row in selected]
        lower = [
            median - 1000.0 * float(row["relax_q25_sec"]) / int(row["relaxation_steps"])
            for median, row in zip(medians, selected)
        ]
        upper = [
            1000.0 * float(row["relax_q75_sec"]) / int(row["relaxation_steps"]) - median
            for median, row in zip(medians, selected)
        ]
        ax.errorbar(
            x_values,
            medians,
            yerr=[lower, upper],
            marker="o",
            color=colors.get(size),
            label=f"n={size}",
            capsize=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Coupling edges")
    ax.set_ylabel("Median relaxation time per step (ms)")
    ax.set_title("Local runtime scaling on the recorded environment")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_cost_reductions(args.logs / "pson_noise_ablation_summary.csv", args.output_dir / "pson_cost_reduction.png")
    plot_escape_rates(
        args.logs / "pson_escape_trials.csv",
        args.logs / "pson_escape_summary.csv",
        args.output_dir / "pson_escape_rate.png",
    )
    plot_scaling(args.logs / "scaling_benchmark.csv", args.output_dir / "runtime_scaling.png")
    print(f"wrote publication figures to {args.output_dir}")


if __name__ == "__main__":
    main()
