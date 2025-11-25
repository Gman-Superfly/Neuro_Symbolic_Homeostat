"""Lightweight metrics logging using Polars."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import polars as pl

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def log_records(name: str, records: List[Dict[str, Any]]) -> Path:
    """Append records to a CSV file under logs/.

    Args:
        name: Base filename without extension.
        records: List of dict rows.
    Returns:
        Path to the written CSV file.
    """
    assert isinstance(name, str) and len(name) > 0, "Invalid log name"
    assert isinstance(records, list), "records must be a list"
    if not records:
        return LOG_DIR / f"{name}.csv"
    df = pl.DataFrame(records)
    out = LOG_DIR / f"{name}.csv"
    if out.exists():
        # read, vstack, and overwrite (align schemas if they differ)
        prev = pl.read_csv(out)
        try:
            df = pl.concat([prev, df], how="vertical_relaxed")
        except Exception:
            # Fallback: align by union of columns
            prev_cols = set(prev.columns)
            new_cols = set(df.columns)
            all_cols = sorted(prev_cols | new_cols)

            # Determine target dtypes per column
            target_dtypes: dict[str, pl.PolarsDataType] = {}
            for c in all_cols:
                dt_prev = prev.schema.get(c, None)
                dt_new = df.schema.get(c, None)
                if dt_prev is None and dt_new is not None:
                    target_dtypes[c] = dt_new
                elif dt_new is None and dt_prev is not None:
                    target_dtypes[c] = dt_prev
                elif dt_prev is None and dt_new is None:
                    target_dtypes[c] = pl.Utf8
                else:
                    # both present
                    if dt_prev == dt_new:
                        target_dtypes[c] = dt_prev  # type: ignore[assignment]
                    else:
                        # Prefer Utf8 if any is Utf8; else use Float64 for numeric mixes
                        if dt_prev == pl.Utf8 or dt_new == pl.Utf8:
                            target_dtypes[c] = pl.Utf8
                        else:
                            target_dtypes[c] = pl.Float64

            def _ensure_cols_and_types(frame: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
                outf = frame
                # Add missing columns with nulls cast to target dtype
                for c in cols:
                    if c not in outf.columns:
                        outf = outf.with_columns(pl.lit(None, dtype=target_dtypes[c]).alias(c))
                    else:
                        # Cast existing column to target dtype if needed
                        if outf.schema[c] != target_dtypes[c]:
                            outf = outf.with_columns(pl.col(c).cast(target_dtypes[c]))
                return outf.select(cols)

            prev_aligned = _ensure_cols_and_types(prev, all_cols)
            df_aligned = _ensure_cols_and_types(df, all_cols)
            df = pl.concat([prev_aligned, df_aligned], how="vertical")
    df.write_csv(out)
    return out


def log_record(name: str, record: Dict[str, Any]) -> Path:
    """Append a single record to a CSV file."""
    return log_records(name, [record])



