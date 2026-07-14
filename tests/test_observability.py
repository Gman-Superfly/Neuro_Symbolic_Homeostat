from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

import cf_logging.metrics_log as metrics_log
from cf_logging.observability import EnergyBudgetTracker, RelaxationTracker
from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision


@dataclass(frozen=True)
class QuadraticModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    target: float

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        difference = float(eta) - self.target
        return 0.5 * difference * difference

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return float(eta) - self.target

    def curvature(self, eta: OrderParameter) -> float:
        return 1.0


def test_trackers_flush_accepted_relaxation_to_csv(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(metrics_log, "LOG_DIR", tmp_path)
    coordinator = EnergyCoordinator(
        modules=[QuadraticModule(target=0.25)],
        couplings=[],
        constraints={},
        noise_mode="none",
        enable_orthogonal_noise=False,
        step_size=0.1,
    )
    relaxation = RelaxationTracker(name="relaxation", run_id="test")
    budget = EnergyBudgetTracker(
        name="budget",
        run_id="test",
        log_free_energy_decomposition=True,
    )
    relaxation.attach(coordinator)
    budget.attach(coordinator)

    coordinator.relax_etas([0.9], steps=3)
    relaxation.flush()
    budget.flush()

    relaxation_frame = pl.read_csv(tmp_path / "relaxation.csv")
    budget_frame = pl.read_csv(tmp_path / "budget.csv")
    assert relaxation_frame.height == 3
    assert budget_frame.height == 3
    assert relaxation_frame.get_column("energy").is_sorted(descending=True)
    assert {"U_internal_energy", "S_entropy", "F_free_energy"}.issubset(
        budget_frame.columns
    )
