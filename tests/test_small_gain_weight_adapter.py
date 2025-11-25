from __future__ import annotations

from typing import Dict

from core.weight_adapters import SmallGainWeightAdapter
from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.sequence.monotonic_eta import SequenceConsistencyModule


def test_small_gain_greedy_allocation_prefers_high_value_low_cost() -> None:
    adapter = SmallGainWeightAdapter(
        budget_fraction=1.0,
        max_step_change=0.1,
        ema_alpha=0.0,  # no smoothing to see raw effect
        floor=0.1,
        ceiling=3.0,
    )
    # Inject costs and global margin snapshot
    adapter.edge_costs = {"coup:A": 1.0, "coup:B": 5.0}  # A cheaper per ΔL
    adapter.global_margin = 1.0
    # Term norms (value); A has larger value
    term_grad_norms: Dict[str, float] = {"coup:A": 10.0, "coup:B": 5.0}
    current = {"coup:A": 1.0, "coup:B": 1.0}
    updated = adapter.step(term_grad_norms, energy=0.0, current=current)
    assert updated["coup:A"] >= updated["coup:B"]
    assert updated["coup:A"] - 1.0 <= 0.1 + 1e-12  # per-step cap


def test_small_gain_respects_floor_and_ceiling() -> None:
    adapter = SmallGainWeightAdapter(
        budget_fraction=1.0,
        max_step_change=0.5,  # large cap to hit ceiling
        floor=0.5,
        ceiling=1.2,
        ema_alpha=0.0,
    )
    adapter.edge_costs = {"coup:A": 0.1}
    adapter.global_margin = 10.0
    term_grad_norms = {"coup:A": 100.0}
    current = {"coup:A": 1.2}
    updated = adapter.step(term_grad_norms, energy=0.0, current=current)
    # Already at ceiling
    assert updated["coup:A"] == 1.2


def test_small_gain_fallback_returns_identity_when_no_values() -> None:
    adapter = SmallGainWeightAdapter()
    term_grad_norms: Dict[str, float] = {"local:X": 3.0}  # no coup:* keys → no values
    current = {"local:X": 0.9}
    updated = adapter.step(term_grad_norms, energy=0.0, current=current)
    assert updated == current


def test_small_gain_keeps_monotone_energy_on_small_problem() -> None:
    # Two sequence modules coupled; adapter active
    mods = [SequenceConsistencyModule(), SequenceConsistencyModule()]
    coups = [(0, 1, QuadraticCoupling(weight=1.0))]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        assert_monotonic_energy=True,
        noise_magnitude=0.0,
        line_search=False,
        step_size=0.02,
        stability_guard=True,
    )
    adapter = SmallGainWeightAdapter(
        budget_fraction=0.5,  # conservative
        max_step_change=0.05,
        ema_alpha=0.3,
        floor=0.1,
        ceiling=3.0,
    )
    # Attach adapter
    coord.weight_adapter = adapter  # type: ignore[assignment]
    # Run a few steps; assertion in coordinator ensures monotone acceptance
    etas0 = [0.2, 0.8]
    _ = coord.relax_etas(etas0, steps=20)


