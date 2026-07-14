from __future__ import annotations

from typing import Dict

import numpy as np

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
    # The coordinator validates each proposal under a fixed objective version.
    etas0 = [0.2, 0.8]
    _ = coord.relax_etas(etas0, steps=20)
    transitions = coord.last_relaxation_metrics()["guard_transitions"]
    assert transitions
    for transition in transitions:
        assert transition["energy_after"] <= transition["energy_before"] + 1e-12


def test_gershgorin_step_cap_contracts_spd_quadratic_iteration() -> None:
    rng = np.random.default_rng(123)
    for size in (3, 6, 10):
        matrix = rng.normal(0.0, 0.2, size=(size, size))
        hessian = matrix.T @ matrix + np.eye(size) * 0.5
        lipschitz_bound = float(np.max(np.sum(np.abs(hessian), axis=1)))
        assert lipschitz_bound > 0.0

        alpha = 0.9 * (2.0 / lipschitz_bound)
        iteration_matrix = np.eye(size) - alpha * hessian
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(iteration_matrix))))

        assert spectral_radius < 1.0


def test_coordinator_injects_family_costs_and_positive_margin() -> None:
    adapter = SmallGainWeightAdapter(budget_fraction=0.5, max_step_change=0.05)
    coord = EnergyCoordinator(
        modules=[SequenceConsistencyModule(), SequenceConsistencyModule()],
        couplings=[(0, 1, QuadraticCoupling(weight=0.1))],
        constraints={},
        weight_adapter=adapter,
        expose_lipschitz_details=True,
        noise_mode="none",
        enable_orthogonal_noise=False,
        step_size=0.03,
        stability_guard=True,
        assert_monotonic_energy=False,
    )

    coord.relax_etas([0.2, 0.8], steps=2)

    family_key = "coup:QuadraticCoupling"
    assert adapter.edge_costs[family_key] > 0.0
    assert adapter.global_margin > 0.0
    assert adapter.last_allocations[family_key] > 0.0
    assert adapter.last_spent_global <= adapter.budget_fraction * adapter.global_margin
    details = getattr(coord, "_last_lipschitz_details")
    assert details["L_est"] > adapter.edge_costs[family_key]


