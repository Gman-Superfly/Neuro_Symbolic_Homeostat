"""Audit synthetic module and coupling curvature reports by finite differences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.coordinator import EnergyCoordinator
from core.curvature_audit import audit_curvature_contract
from experiments.ablate_pson_noise import SCENARIOS, build_case


def run_audit(samples: int, base_size: int, seed: int) -> List[Dict[str, Any]]:
    assert samples > 0
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        case = build_case(scenario, seed=seed, base_size=base_size)
        coordinator = EnergyCoordinator(case.modules, case.couplings, case.constraints, noise_mode="none")
        states = [case.inputs]
        states.extend(rng.uniform(0.02, 0.98, size=len(case.modules)).tolist() for _ in range(samples - 1))
        for record in audit_curvature_contract(coordinator, states):
            rows.append({"scenario": scenario, **record.as_dict()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32, help="States sampled per problem family.")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--strict", action="store_true", help="Fail if a reported bound is underestimated.")
    parser.add_argument("--output", type=Path, default=Path("logs/curvature_contract_audit.csv"))
    args = parser.parse_args()

    rows = run_audit(int(args.samples), int(args.size), int(args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {status: sum(row["status"] == status for row in rows) for status in ("covered", "unreported", "underreported")}
    print(f"wrote {len(rows)} audit rows to {args.output}")
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    if args.strict and counts["underreported"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
