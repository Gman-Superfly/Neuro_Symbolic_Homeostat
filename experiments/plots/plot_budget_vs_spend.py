from __future__ import annotations

import argparse
import polars as pl
import matplotlib.pyplot as plt


def compute_alloc_cost_totals(df: pl.DataFrame) -> tuple[list[float] | None, list[float] | None]:
    """Return per-step totals for all alloc:* and cost:* columns (row-wise sums).

    Returns:
        (alloc_total, cost_total) where each is a list of floats (length = rows) or None if
        no corresponding columns exist.
    """
    alloc_cols = [c for c in df.columns if c.startswith("alloc:")]
    cost_cols = [c for c in df.columns if c.startswith("cost:")]
    alloc_total = None
    cost_total = None
    if alloc_cols:
        alloc_df = df.select(alloc_total=pl.sum_horizontal(pl.col(alloc_cols)))
        alloc_total = alloc_df.get_column("alloc_total").to_list()
    if cost_cols:
        cost_df = df.select(cost_total=pl.sum_horizontal(pl.col(cost_cols)))
        cost_total = cost_df.get_column("cost_total").to_list()
    return alloc_total, cost_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="logs/energy_budget.csv")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = pl.read_csv(args.input)
    if args.run_id is not None:
        df = df.filter(pl.col("run_id") == args.run_id)

    # Aggregate allocation and cost columns row-wise (per step), if present
    alloc_total, cost_total = compute_alloc_cost_totals(df)

    # Plot per-step curves for global spend, aggregated alloc/cost, and contraction margin when available
    has_spent = "spent:global" in df.columns
    fig, ax1 = plt.subplots(figsize=(8, 4))
    if has_spent:
        ax1.plot(df["spent:global"], label="spent:global", color="tab:blue")
    if alloc_total is not None:
        ax1.plot(alloc_total, label="alloc:total", color="tab:green", alpha=0.7)
    if cost_total is not None:
        ax1.plot(cost_total, label="cost:total", color="tab:red", alpha=0.5)
    # Right axis for contraction margin and homotopy signals (if present)
    has_margin = "contraction_margin" in df.columns
    has_homo_scale = "homotopy_scale" in df.columns
    has_homo_backoffs = "homotopy_backoffs" in df.columns
    if has_margin or has_homo_scale or has_homo_backoffs:
        ax2 = ax1.twinx()
        lines2, labels2 = [], []
        if has_margin:
            l1 = ax2.plot(df["contraction_margin"], label="contraction_margin", color="tab:orange", alpha=0.6)
            lines2 += l1
            labels2 += ["contraction_margin"]
        if has_homo_scale:
            l2 = ax2.plot(df["homotopy_scale"], label="homotopy_scale", color="tab:purple", alpha=0.6, linestyle="--")
            lines2 += l2
            labels2 += ["homotopy_scale"]
        if has_homo_backoffs:
            # Plot backoffs as a step line (scaled if large)
            try:
                backoffs = df["homotopy_backoffs"]
                l3 = ax2.plot(backoffs, label="homotopy_backoffs", color="tab:gray", alpha=0.6, linestyle=":")
                lines2 += l3
                labels2 += ["homotopy_backoffs"]
            except Exception:
                pass
        ax2.set_ylabel("contraction_margin / homotopy")
    else:
        lines2, labels2 = [], []
    ax1.set_xlabel("step")
    ax1.set_ylabel("global spend")
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()


