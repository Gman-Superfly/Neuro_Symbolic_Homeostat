from __future__ import annotations

import argparse
from typing import List, Tuple

import polars as pl
import matplotlib.pyplot as plt


def _select_prefixed(df: pl.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def compute_gain_budget_timeseries(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    Returns a DataFrame with alloc:total, cost:total and passes through the original prefixed columns.
    """
    alloc_cols = _select_prefixed(df, "alloc:")
    cost_cols = _select_prefixed(df, "cost:")
    if len(alloc_cols) == 0 and len(cost_cols) == 0:
        return df, alloc_cols, cost_cols
    out = df
    if len(alloc_cols) > 0:
        out = out.with_columns(pl.sum_horizontal(pl.col(alloc_cols)).alias("alloc:total"))
    if len(cost_cols) > 0:
        out = out.with_columns(pl.sum_horizontal(pl.col(cost_cols)).alias("cost:total"))
    if len(alloc_cols) > 0 and len(cost_cols) > 0:
        # Avoid divide by zero
        out = out.with_columns(
            (pl.when(pl.col("cost:total") > 0.0)
               .then(pl.col("alloc:total") / pl.col("cost:total"))
               .otherwise(None)
             ).alias("alloc_to_cost_ratio")
        )
    return out, alloc_cols, cost_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="EnergyBudgetTracker CSV (e.g., logs\\benchmark_delta_f90_budget.csv)")
    parser.add_argument("--run_id", type=str, default=None, help="Filter by run_id value")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG). If not set, show() is called")
    parser.add_argument("--topk", type=int, default=5, help="Plot top-K allocation families by mean allocation")
    parser.add_argument("--include_cost_series", action="store_true", help="Overlay cost:* family series")
    args = parser.parse_args()

    df = pl.read_csv(args.input)
    if args.run_id is not None and "run_id" in df.columns:
        df = df.filter(pl.col("run_id") == args.run_id)

    df2, alloc_cols, cost_cols = compute_gain_budget_timeseries(df)

    if len(alloc_cols) == 0 and len(cost_cols) == 0:
        print("No alloc:* or cost:* columns present; nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    # Plot totals first
    if "alloc:total" in df2.columns:
        ax.plot(df2["alloc:total"], label="alloc:total", color="tab:blue")
    if "cost:total" in df2.columns:
        ax.plot(df2["cost:total"], label="cost:total", color="tab:red")
    if "alloc_to_cost_ratio" in df2.columns:
        ax2 = ax.twinx()
        ax2.plot(df2["alloc_to_cost_ratio"], label="alloc_to_cost_ratio", color="tab:green", linestyle="--", alpha=0.7)
        ax2.set_ylabel("alloc_to_cost_ratio")
        # Merge legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best", fontsize="small")
    else:
        ax.legend(loc="best", fontsize="small")

    # Top-K family allocations
    if len(alloc_cols) > 0:
        alloc_means = df2.select([pl.mean(c).alias(c) for c in alloc_cols]).to_dict(as_series=False)
        # Flatten dict of lists
        means = [(k, float(v[0]) if isinstance(v, list) and len(v) > 0 else 0.0) for k, v in alloc_means.items()]
        means = sorted(means, key=lambda kv: kv[1], reverse=True)[: max(0, int(args.topk))]
        for k, _ in means:
            ax.plot(df2[k], label=k, alpha=0.6)
        ax.legend(loc="upper left", fontsize="x-small")

    # Optional: overlay cost family series
    if args.include_cost_series and len(cost_cols) > 0:
        for c in sorted(cost_cols):
            ax.plot(df2[c], label=c, linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize="x-small")

    ax.set_xlabel("step")
    ax.set_ylabel("allocation / cost")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()


