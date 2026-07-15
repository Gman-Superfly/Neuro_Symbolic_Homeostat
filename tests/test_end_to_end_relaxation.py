from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import pytest

from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling, QuadraticCoupling
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision
from core.weight_adapters import SmallGainWeightAdapter
from experiments.demo_constraint_correction import run_constraint_correction


@dataclass(frozen=True)
class QuadraticModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Small local quadratic used for end-to-end relaxation tests."""

    target: float
    curvature_value: float = 1.0

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        diff = float(eta) - self.target
        return 0.5 * self.curvature_value * diff * diff

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return self.curvature_value * (float(eta) - self.target)

    def curvature(self, eta: OrderParameter) -> float:
        return self.curvature_value


@dataclass(frozen=True)
class BoxOnlyQuadraticModule(EnergyModule):
    """Quadratic that rejects every energy evaluation outside the eta box."""

    target: float
    curvature_value: float = 1.0

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        del constraints
        value = float(eta)
        if value < 0.0 or value > 1.0:
            raise ValueError("eta outside [0, 1]")
        return 0.5 * self.curvature_value * (value - self.target) ** 2


@dataclass
class OneShotCouplingWeightAdapter:
    """Increase one coupling-family weight after the first accepted proposal."""

    updated: bool = False

    def step(
        self,
        term_grad_norms: Mapping[str, float],
        energy: float,
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        del term_grad_norms, energy
        weights = dict(current)
        if not self.updated:
            weights["coup:QuadraticCoupling"] = 4.0
            self.updated = True
        return weights


def test_safe_defaults_keep_accepted_energy_monotone() -> None:
    modules = [QuadraticModule(0.2), QuadraticModule(0.8)]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, QuadraticCoupling(weight=0.2))],
        constraints={},
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
        step_size=0.05,
        stability_guard=True,
    )
    etas0 = [0.9, 0.1]
    energies = [coord.energy(list(etas0))]
    coord.on_energy_updated.append(lambda energy: energies.append(float(energy)))

    coord.relax_etas(list(etas0), steps=25)

    accepted = energies[1:]
    assert len(accepted) > 2
    for before, after in zip(accepted, accepted[1:]):
        assert after <= before + 1e-12
    metrics = coord.last_relaxation_metrics()
    assert metrics["accepted_steps"] == len(accepted)
    assert metrics["energy_trace"] == accepted


def test_gradient_norm_decreases_on_quadratic_relaxation() -> None:
    modules = [QuadraticModule(0.1, 2.0), QuadraticModule(0.9, 3.0)]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, QuadraticCoupling(weight=0.1))],
        constraints={},
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
        use_stiffness_updates=True,
        step_size=0.2,
        stability_guard=False,
    )
    etas0 = [0.8, 0.2]
    grad0 = float(np.linalg.norm(np.asarray(coord._grads(etas0), dtype=float)))  # type: ignore[attr-defined]
    etas1 = coord.relax_etas(list(etas0), steps=20)
    grad1 = float(np.linalg.norm(np.asarray(coord._grads(etas1), dtype=float)))  # type: ignore[attr-defined]

    assert grad1 < grad0


@pytest.mark.parametrize("use_analytic", [False, True])
def test_finite_difference_fallback_is_box_aware_and_quadratic_exact_at_edges(
    use_analytic: bool,
) -> None:
    coord = EnergyCoordinator(
        modules=[BoxOnlyQuadraticModule(target=0.75, curvature_value=2.0)],
        couplings=[],
        constraints={},
        use_analytic=use_analytic,
        stability_guard=False,
        noise_mode="none",
    )

    lower = coord._grads([0.0])  # type: ignore[attr-defined]
    upper = coord._grads([1.0])  # type: ignore[attr-defined]

    assert lower == pytest.approx([-1.5], abs=1e-10)
    assert upper == pytest.approx([0.5], abs=1e-10)


def test_full_finite_difference_keeps_isolated_local_objectives_in_partial_graph() -> None:
    coord = EnergyCoordinator(
        modules=[
            QuadraticModule(0.2),
            QuadraticModule(0.8),
            QuadraticModule(0.9),
        ],
        couplings=[(0, 1, QuadraticCoupling(weight=0.1))],
        constraints={},
        use_analytic=False,
        neighbor_gradients_only=True,
        use_precision_preconditioning=False,
        stability_guard=False,
        noise_mode="none",
        step_size=0.1,
    )

    initial = [0.5, 0.5, 0.1]
    gradient = coord._grads(initial)  # type: ignore[attr-defined]
    result = coord.relax_etas(initial, steps=1)

    assert gradient[2] == pytest.approx(-0.8, abs=1e-10)
    assert result[2] > initial[2]


def test_pson_receives_isolated_local_gradient_in_partial_graph() -> None:
    coord = EnergyCoordinator(
        modules=[
            QuadraticModule(0.2),
            QuadraticModule(0.8),
            QuadraticModule(0.9),
        ],
        couplings=[(0, 1, QuadraticCoupling(weight=0.1))],
        constraints={},
        use_analytic=False,
        neighbor_gradients_only=True,
        use_precision_preconditioning=False,
        stability_guard=False,
        noise_mode="precision_orthogonal",
        noise_magnitude=0.1,
        step_size=0.1,
    )
    captured: list[np.ndarray] = []

    def capture_noise(raw: np.ndarray, gradient: np.ndarray, magnitude: float) -> np.ndarray:
        del raw, magnitude
        captured.append(np.asarray(gradient, dtype=float).copy())
        return np.zeros_like(gradient)

    coord._build_noise_vector = capture_noise  # type: ignore[method-assign]
    coord.relax_etas([0.5, 0.5, 0.1], steps=1)

    assert len(captured) == 1
    assert captured[0][2] == pytest.approx(-0.8, abs=1e-10)


def test_adaptive_guard_compares_each_proposal_under_one_objective_version() -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.2), QuadraticModule(0.8)],
        couplings=[(0, 1, QuadraticCoupling(weight=0.2))],
        constraints={},
        weight_adapter=OneShotCouplingWeightAdapter(),
        use_analytic=True,
        use_precision_preconditioning=False,
        enable_orthogonal_noise=False,
        noise_mode="none",
        step_size=0.02,
        stability_guard=True,
        assert_monotonic_energy=True,
    )

    coord.relax_etas([0.9, 0.1], steps=4)

    metrics = coord.last_relaxation_metrics()
    transitions = metrics["guard_transitions"]
    assert metrics["accepted_steps"] == 4
    assert metrics["objective_version"] == 1
    assert [transition["objective_version"] for transition in transitions] == [0, 1, 1, 1]
    for transition in transitions:
        assert transition["energy_after"] <= transition["energy_before"] + 1e-12


def test_free_energy_guard_accepts_equal_state_within_tolerance() -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5)],
        couplings=[],
        constraints={},
        use_free_energy_guard=True,
        free_energy_epsilon=1e-6,
        noise_mode="none",
        enable_orthogonal_noise=False,
    )

    result = coord.relax_etas([0.5], steps=1)

    assert result == [0.5]
    assert coord.last_relaxation_metrics()["accepted_steps"] == 1


def test_step_cap_slack_positive_across_coupling_sweep() -> None:
    for weight in (0.05, 0.2, 0.5):
        coord = EnergyCoordinator(
            modules=[QuadraticModule(0.3), QuadraticModule(0.7)],
            couplings=[(0, 1, QuadraticCoupling(weight=weight))],
            constraints={},
            use_analytic=True,
            enable_orthogonal_noise=False,
            noise_mode="none",
            step_size=0.05,
            stability_guard=True,
            log_step_cap_slack=True,
        )
        coord.relax_etas([0.9, 0.1], steps=3)
        slacks = [value for value in getattr(coord, "_step_cap_slack_history") if np.isfinite(value)]
        assert slacks, "expected step-cap slacks"
        assert min(slacks) > 0.0


def test_auto_noise_controller_receives_previous_proposal_backtracks(monkeypatch: Any) -> None:
    class RecordingController:
        def __init__(self) -> None:
            self.backtracks: list[int] = []
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

        def step(
            self,
            grad: np.ndarray,
            energy_drop_ratio: float,
            backtracks: int,
            iter_idx: int,
        ) -> float:
            del grad, energy_drop_ratio, iter_idx
            self.backtracks.append(int(backtracks))
            return 0.0

    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5)],
        couplings=[],
        constraints={},
        noise_mode="orthogonal",
        noise_magnitude=0.1,
        auto_noise_controller=True,
        line_search=True,
        stability_guard=False,
    )
    recorder = RecordingController()
    coord._noise_controller = recorder  # type: ignore[assignment]
    proposal_count = 0

    def fake_step_with_backtracking(
        etas: list[OrderParameter],
        objective_grads: list[float],
        direction: list[float],
        step_init: float,
    ) -> list[OrderParameter]:
        nonlocal proposal_count
        del objective_grads, direction, step_init
        coord._last_step_backtracks = 3 if proposal_count == 0 else 0
        coord._last_acceptance_reason = "armijo_accepted"
        proposal_count += 1
        return list(etas)

    monkeypatch.setattr(coord, "_step_with_backtracking", fake_step_with_backtracking)

    coord.relax_etas([0.5], steps=2)

    assert recorder.reset_calls == 1
    assert recorder.backtracks == [0, 3]


def test_constraint_correction_demo_reduces_all_reported_violations() -> None:
    result = run_constraint_correction()
    for key, before in result["raw_violations"].items():
        after = result["relaxed_violations"][key]
        assert after < before, f"{key} did not improve: {before} -> {after}"
    assert result["relaxed_energy"] < result["raw_energy"]


def test_wrong_sign_gate_benefit_pushes_gate_closed() -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5), QuadraticModule(0.0)],
        couplings=[(0, 1, GateBenefitCoupling(weight=1.0, delta_key="delta_benefit"))],
        constraints={"delta_benefit": -0.2},
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
        step_size=0.05,
        stability_guard=False,
        assert_monotonic_energy=False,
    )
    etas = coord.relax_etas([0.5, 0.0], steps=1)

    assert etas[0] < 0.5


def test_guarded_gate_benefit_uses_exact_zero_curvature_contract() -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5), QuadraticModule(0.0)],
        couplings=[(0, 1, GateBenefitCoupling(weight=1.0, delta_key="delta_benefit"))],
        constraints={"delta_benefit": 0.2},
        use_analytic=True,
        use_precision_preconditioning=False,
        noise_mode="none",
        step_size=0.05,
        stability_guard=True,
    )
    initial = [0.5, 0.0]

    result = coord.relax_etas(initial, steps=1)

    assert result[0] > initial[0]
    assert coord.energy(result) < coord.energy(initial)


def test_gate_benefit_is_frozen_within_relaxation_and_refreshed_next_run() -> None:
    constraints = {"delta_benefit": 0.2}
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5, 0.0), QuadraticModule(0.0, 0.0)],
        couplings=[(0, 1, GateBenefitCoupling(weight=1.0, delta_key="delta_benefit"))],
        constraints=constraints,
        use_analytic=True,
        use_precision_preconditioning=False,
        enable_orthogonal_noise=False,
        noise_mode="none",
        step_size=0.1,
        stability_guard=False,
    )
    accepted = 0

    def reverse_external_benefit(_: list[OrderParameter]) -> None:
        nonlocal accepted
        accepted += 1
        if accepted == 1:
            constraints["delta_benefit"] = -0.2

    coord.on_eta_updated.append(reverse_external_benefit)
    first = coord.relax_etas([0.5, 0.0], steps=2)

    assert first[0] == pytest.approx(0.54)
    assert constraints["delta_benefit"] == -0.2
    assert coord.constraints is constraints

    coord.on_eta_updated.remove(reverse_external_benefit)
    second = coord.relax_etas(first, steps=1)
    assert second[0] == pytest.approx(0.52)


def test_nonfinite_gate_benefit_is_rejected_at_run_boundary() -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5), QuadraticModule(0.5)],
        couplings=[(0, 1, GateBenefitCoupling(delta_key="delta_benefit"))],
        constraints={"delta_benefit": math.inf},
    )

    with pytest.raises(ValueError, match="gate-benefit constraint.*must be finite"):
        coord.relax_etas([0.5, 0.5], steps=1)


def test_mapping_term_weights_are_frozen_with_external_constraints() -> None:
    term_weights = UserDict({"coup:GateBenefitCoupling": 1.0})
    constraints = {
        "delta_benefit": 0.2,
        "term_weights": term_weights,
    }
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5, 0.0), QuadraticModule(0.0, 0.0)],
        couplings=[(0, 1, GateBenefitCoupling(delta_key="delta_benefit"))],
        constraints=constraints,
        use_precision_preconditioning=False,
        noise_mode="none",
        step_size=0.1,
        stability_guard=False,
    )

    coord.on_eta_updated.append(
        lambda _: term_weights.__setitem__("coup:GateBenefitCoupling", 2.0)
    )
    result = coord.relax_etas([0.5, 0.0], steps=2)

    assert result[0] == pytest.approx(0.54)
    assert term_weights["coup:GateBenefitCoupling"] == 2.0


def test_post_armijo_noise_rejection_overrides_acceptance_reason(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        np.random,
        "normal",
        lambda *args, **kwargs: np.ones(kwargs["size"], dtype=float),
    )
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5)],
        couplings=[],
        constraints={},
        noise_mode="isotropic",
        noise_magnitude=1.0,
        use_precision_preconditioning=False,
        line_search=True,
        stability_guard=False,
    )

    result = coord.relax_etas([0.5], steps=1)
    metrics = coord.last_relaxation_metrics()

    assert result == [0.5]
    assert metrics["accepted_steps"] == 0
    assert metrics["rejected_steps"] == 1
    assert coord._last_acceptance_reason == "non_monotonic_rejected"
    assert metrics["last_acceptance_reason"] == "non_monotonic_rejected"
    assert metrics["acceptance_reasons"] == ["non_monotonic_rejected"]


def test_too_large_isotropic_noise_rejects_and_restores_state() -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5)],
        couplings=[],
        constraints={},
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="isotropic",
        noise_magnitude=1.0,
        step_size=0.05,
        stability_guard=False,
        assert_monotonic_energy=False,
    )
    etas0 = [0.5]
    etas1 = coord.relax_etas(list(etas0), steps=1)

    assert getattr(coord, "_rejected_steps") == 1
    assert etas1 == etas0


def test_rejection_can_restore_and_continue_remaining_attempts(monkeypatch: Any) -> None:
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: np.ones(kwargs["size"], dtype=float))
    coord = EnergyCoordinator(
        modules=[QuadraticModule(0.5)],
        couplings=[],
        constraints={},
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="isotropic",
        noise_magnitude=1.0,
        step_size=0.05,
        stability_guard=False,
        assert_monotonic_energy=False,
        continue_after_rejection=True,
    )

    result = coord.relax_etas([0.5], steps=3)
    metrics = coord.last_relaxation_metrics()

    assert result == [0.5]
    assert metrics["accepted_steps"] == 0
    assert metrics["rejected_steps"] == 3
    assert metrics["attempted_steps"] == 3
    assert metrics["attempt_energy_trace"] == [0.0, 0.0, 0.0]
    assert metrics["acceptance_reasons"] == ["non_monotonic_rejected"] * 3


def test_sparse_smallgain_preserves_monotone_energy_but_is_not_speed_claim() -> None:
    modules = [QuadraticModule(0.2), QuadraticModule(0.8)]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, QuadraticCoupling(weight=0.1))],
        constraints={},
        weight_adapter=SmallGainWeightAdapter(budget_fraction=0.5, max_step_change=0.05),
        expose_lipschitz_details=True,
        use_analytic=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
        step_size=0.03,
        stability_guard=True,
        assert_monotonic_energy=False,
    )
    energies = [coord.energy([0.9, 0.1])]
    coord.on_energy_updated.append(lambda energy: energies.append(float(energy)))

    coord.relax_etas([0.9, 0.1], steps=10)

    accepted = energies[1:]
    assert len(accepted) > 1
    assert accepted[-1] <= accepted[0]
