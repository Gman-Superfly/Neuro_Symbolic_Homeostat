"""Tests for diagonal precision preconditioning and the Lipschitz curvature estimate.

These tests validate two claims from the paper's relaxation mechanism:

1. The stability guard's Lipschitz estimate must upper-bound the true curvature
   for the 0.9 * 2/L step cap to guarantee descent. At a box edge the
   finite-difference window is clipped on one side, so the estimate must
   normalize by the actual window width rather than the nominal 2*eps.

2. Dividing each coordinate's gradient by its diagonal curvature (the diagonal
   natural-gradient step) converges in fewer iterations than plain gradient
   descent when the curvature spread across coordinates is large.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping

import math

from core.coordinator import EnergyCoordinator
from core.couplings import DirectedHingeCoupling, QuadraticCoupling
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision
from experiments.demo_constraint_correction import ProductExclusionCoupling, SumToOneCoupling, TargetScoreModule


@dataclass(frozen=True)
class QuadraticModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Local quadratic energy 0.5 * stiffness * (eta - target)^2."""

    target: float
    stiffness: float

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x) if x is not None else 0.0

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        diff = float(eta) - self.target
        return 0.5 * self.stiffness * diff * diff

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return self.stiffness * (float(eta) - self.target)

    def curvature(self, eta: OrderParameter) -> float:  # type: ignore[override]
        return self.stiffness


def _iters_to_converge(
    targets: List[float],
    stiffness: List[float],
    init: List[float],
    *,
    precondition: bool,
    tol: float,
    max_iters: int,
) -> int:
    """Relax one step at a time and return the iteration count at convergence."""
    modules = [QuadraticModule(target=t, stiffness=s) for t, s in zip(targets, stiffness)]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=False,
        use_precision_preconditioning=precondition,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        stability_cap_fraction=0.9,
        noise_mode="none",
        enable_orthogonal_noise=False,
        assert_monotonic_energy=False,
        enable_early_stop=False,
    )
    etas = [float(e) for e in init]
    for iters in range(1, max_iters + 1):
        etas = [float(x) for x in coord.relax_etas(etas, steps=1)]
        err = max(abs(e - t) for e, t in zip(etas, targets))
        if err <= tol:
            return iters
    return max_iters + 1


def test_lipschitz_curvature_not_underestimated_at_box_edge() -> None:
    """The local curvature estimate at a box edge matches the interior value.

    For a quadratic the true curvature equals the stiffness everywhere. Before
    the fix the clipped finite-difference window halved the estimate at eta=0 and
    eta=1, which underbounds L and breaks the step-cap descent guarantee.
    """
    stiffness = 50.0
    coord = EnergyCoordinator(
        modules=[QuadraticModule(target=0.5, stiffness=stiffness)],
        couplings=[],
        constraints={},
        use_analytic=True,
        stability_guard=True,
        enable_orthogonal_noise=False,
        noise_mode="none",
    )
    l_interior = coord._estimate_lipschitz_bound([0.5])  # type: ignore[attr-defined]
    l_edge_low = coord._estimate_lipschitz_bound([0.0])  # type: ignore[attr-defined]
    l_edge_high = coord._estimate_lipschitz_bound([1.0])  # type: ignore[attr-defined]

    assert math.isclose(l_interior, stiffness, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(l_edge_low, stiffness, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(l_edge_high, stiffness, rel_tol=1e-6, abs_tol=1e-6)
    assert l_edge_low >= 0.99 * l_interior
    assert l_edge_high >= 0.99 * l_interior


def test_preconditioning_converges_faster_on_ill_conditioned_problem() -> None:
    """Diagonal preconditioning beats plain gradient descent under large curvature spread.

    The energy is uncoupled, so the curvature is diagonal and axis-aligned, which
    is the regime where diagonal preconditioning is expected to help. Both runs use
    the identical Lipschitz-derived step; only the per-coordinate scaling differs.
    """
    targets = [0.5, 0.5, 0.5]
    stiffness = [100.0, 1.0, 0.1]  # condition number 1000
    init = [0.8, 0.8, 0.8]
    tol = 1e-3
    max_iters = 20000

    plain_iters = _iters_to_converge(
        targets, stiffness, init, precondition=False, tol=tol, max_iters=max_iters
    )
    precond_iters = _iters_to_converge(
        targets, stiffness, init, precondition=True, tol=tol, max_iters=max_iters
    )

    assert precond_iters <= max_iters, "preconditioned run did not converge"
    assert plain_iters <= max_iters, "plain run did not converge"
    assert precond_iters < plain_iters
    # The spread is 1000x, so the conditioning benefit should be substantial.
    assert plain_iters >= 3 * precond_iters


def _coupled_pair(precondition: bool, guard: bool, step: float) -> EnergyCoordinator:
    """Build two quadratics joined by a strong quadratic coupling.

    The total Hessian is [[17, -16], [-16, 17]] with eigenvalues 33 (antisymmetric)
    and 1 (symmetric), so the safe step for plain gradient descent is 2/33 = 0.0606.
    """
    modules = [QuadraticModule(target=0.2, stiffness=1.0), QuadraticModule(target=0.8, stiffness=1.0)]
    return EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, QuadraticCoupling(weight=8.0))],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=False,
        use_precision_preconditioning=precondition,
        stability_guard=guard,
        auto_step_from_lipschitz=guard,
        stability_cap_fraction=0.9,
        step_size=step,
        noise_mode="none",
        enable_orthogonal_noise=False,
        assert_monotonic_energy=False,
    )


def test_curvature_awareness_converges_where_plain_gd_stalls_above_2_over_L() -> None:
    """Curvature handling keeps an aggressive step safe where plain GD stalls.

    The coupled Hessian has largest eigenvalue 33, so plain gradient descent
    requires a step below 2/33 = 0.0606. A requested step of 0.1 overshoots on the
    first move; the acceptance guard rejects it and the iteration makes no progress.
    Either curvature-aware mechanism, diagonal precision preconditioning or the
    Gershgorin step cap, scales the effective step into the safe region and
    converges to the coupled minimum near [0.5, 0.5] without manual step tuning.
    """
    step = 0.1  # above the plain-GD safe step 2/L = 0.0606
    init = [0.9, 0.1]

    plain = _coupled_pair(precondition=False, guard=False, step=step)
    out_plain = plain.relax_etas(list(init), steps=200)
    energy_plain = plain.energy(out_plain)
    assert getattr(plain, "_rejected_steps") >= 1, "plain GD step above 2/L should be rejected"
    assert math.isclose(out_plain[0], init[0], abs_tol=1e-9)
    assert math.isclose(out_plain[1], init[1], abs_tol=1e-9)

    precond = _coupled_pair(precondition=True, guard=False, step=step)
    out_precond = precond.relax_etas(list(init), steps=200)
    energy_precond = precond.energy(out_precond)

    guard = _coupled_pair(precondition=False, guard=True, step=step)
    out_guard = guard.relax_etas(list(init), steps=200)
    energy_guard = guard.energy(out_guard)

    assert energy_precond < 0.5 * energy_plain
    assert energy_guard < 0.5 * energy_plain
    for out in (out_precond, out_guard):
        assert abs((out[0] + out[1]) - 1.0) < 0.05
        assert abs(out[0] - out[1]) < 0.1


def test_gershgorin_cap_can_be_conservative_on_mixed_preconditioned_problem() -> None:
    """Mixed constraints can converge above the initial conservative cap.

    This test records an observed boundary for the current implementation:
    the initial Gershgorin cap can be lower than what still converges on a
    mixed hinge and product-coupling problem. This does not weaken safety.
    It documents that the bound can be conservative, so speed impact is
    problem dependent.
    """
    raw = [0.82, 0.71, 0.25, 0.68]
    requested_step = 0.1

    def _build(guard: bool) -> EnergyCoordinator:
        modules = [TargetScoreModule(target=value, stiffness=0.25) for value in raw]
        couplings = [
            (0, 1, SumToOneCoupling(weight=12.0)),
            (0, 1, ProductExclusionCoupling(weight=2.0)),
            (2, 3, DirectedHingeCoupling(weight=8.0)),
        ]
        return EnergyCoordinator(
            modules=modules,
            couplings=couplings,
            constraints={},
            use_analytic=True,
            use_stiffness_updates=False,
            use_precision_preconditioning=True,
            stability_guard=guard,
            auto_step_from_lipschitz=guard,
            noise_mode="none",
            enable_orthogonal_noise=False,
            step_size=requested_step,
            assert_monotonic_energy=False,
        )

    probe = _build(guard=True)
    l_init = float(probe._estimate_lipschitz_bound(list(raw)))  # type: ignore[attr-defined]
    assert l_init > 0.0
    safe_cap = 0.9 * 2.0 / l_init
    assert requested_step > safe_cap

    no_guard = _build(guard=False)
    guarded = _build(guard=True)
    energy0 = float(no_guard.energy(list(raw)))
    out_no_guard = [float(value) for value in no_guard.relax_etas(list(raw), steps=500)]
    out_guarded = [float(value) for value in guarded.relax_etas(list(raw), steps=500)]
    energy_no_guard = float(no_guard.energy(out_no_guard))
    energy_guarded = float(guarded.energy(out_guarded))

    assert energy_no_guard < energy0
    assert energy_guarded < energy0
    assert getattr(no_guard, "_rejected_steps") == 0
    assert getattr(guarded, "_rejected_steps") == 0
    assert energy_no_guard <= energy_guarded
