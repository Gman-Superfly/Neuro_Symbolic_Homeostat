"""Tests for diagonal precision preconditioning and the Lipschitz curvature estimate.

These tests validate two claims from the paper's relaxation mechanism:

1. The stability guard's Lipschitz estimate must upper-bound the true curvature
   for the 0.9 * 2/L step cap to guarantee descent. At a box edge the
   finite-difference window is clipped on one side, so the estimate must
   normalize by the actual window width rather than the nominal 2*eps.

2. Dividing each coordinate's gradient by a positive diagonal preconditioner
   converges in fewer iterations than plain gradient descent when the curvature
   spread across coordinates is large.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple

import math
import numpy as np
import pytest

import core.coordinator_stability as coordinator_stability
from core.coordinator import EnergyCoordinator
from core.couplings import DirectedHingeCoupling, QuadraticCoupling
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision


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


@dataclass(frozen=True)
class ScaledQuadraticCoupling:
    """Custom quadratic coupling used to validate the curvature protocol."""

    weight: float
    scale_i: float
    scale_j: float

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        del constraints
        residual = self.scale_i * float(eta_i) - self.scale_j * float(eta_j)
        return 0.5 * self.weight * residual * residual

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        del constraints
        residual = self.scale_i * float(eta_i) - self.scale_j * float(eta_j)
        return (
            self.weight * self.scale_i * residual,
            -self.weight * self.scale_j * residual,
        )

    def coupling_curvature_bounds(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float, float]:
        del eta_i, eta_j, constraints
        return (
            self.weight * self.scale_i * self.scale_i,
            self.weight * self.scale_j * self.scale_j,
            self.weight * abs(self.scale_i * self.scale_j),
        )


@dataclass(frozen=True)
class UnderreportedQuadraticCoupling:
    """Quadratic edge with a deliberately invalid curvature report."""

    weight: float
    report_fraction: float

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        del constraints
        diff = float(eta_i) - float(eta_j)
        return self.weight * diff * diff

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        del constraints
        grad = 2.0 * self.weight * (float(eta_i) - float(eta_j))
        return grad, -grad

    def coupling_curvature_bounds(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float, float]:
        del eta_i, eta_j, constraints
        reported = 2.0 * self.weight * self.report_fraction
        return reported, reported, reported


@dataclass(frozen=True)
class UnreportedQuadraticCoupling:
    """Custom edge that deliberately omits the curvature protocol."""

    weight: float

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        del constraints
        diff = float(eta_i) - float(eta_j)
        return self.weight * diff * diff

    def d_coupling_energy_d_etas(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> Tuple[float, float]:
        del constraints
        grad = 2.0 * self.weight * (float(eta_i) - float(eta_j))
        return grad, -grad


@pytest.mark.parametrize("invalid_weight", [-1.0, math.nan])
def test_invalid_custom_curvature_bounds_fail_closed(invalid_weight: float) -> None:
    coordinator = EnergyCoordinator(
        modules=[QuadraticModule(0.5, 1.0), QuadraticModule(0.5, 1.0)],
        couplings=[
            (
                0,
                1,
                ScaledQuadraticCoupling(
                    weight=invalid_weight,
                    scale_i=1.0,
                    scale_j=1.0,
                ),
            )
        ],
        constraints={},
    )

    with pytest.raises(ValueError, match="non-negative and not NaN"):
        coordinator.inspect_state([0.4, 0.6])


def test_infinite_custom_curvature_requires_line_search_for_guarded_update() -> None:
    coordinator = EnergyCoordinator(
        modules=[QuadraticModule(0.5, 1.0), QuadraticModule(0.5, 1.0)],
        couplings=[
            (
                0,
                1,
                UnderreportedQuadraticCoupling(
                    weight=1.0,
                    report_fraction=math.inf,
                ),
            )
        ],
        constraints={},
        use_precision_preconditioning=False,
        stability_guard=True,
        line_search=False,
    )

    with pytest.raises(ValueError, match="curvature bound is infinite"):
        coordinator.relax_etas([0.4, 0.6], steps=1)


def test_missing_custom_curvature_report_requires_line_search_for_guarded_update() -> None:
    fixed_step = EnergyCoordinator(
        modules=[QuadraticModule(0.5, 1.0), QuadraticModule(0.5, 1.0)],
        couplings=[(0, 1, UnreportedQuadraticCoupling(weight=1.0))],
        constraints={},
        use_precision_preconditioning=False,
        stability_guard=True,
        line_search=False,
    )

    assert math.isinf(fixed_step.inspect_state([0.4, 0.6]).update_lipschitz_bound)
    with pytest.raises(ValueError, match="curvature bound is infinite"):
        fixed_step.relax_etas([0.4, 0.6], steps=1)

    searched = EnergyCoordinator(
        modules=[QuadraticModule(0.5, 1.0), QuadraticModule(0.5, 1.0)],
        couplings=[(0, 1, UnreportedQuadraticCoupling(weight=1.0))],
        constraints={},
        use_precision_preconditioning=False,
        stability_guard=True,
        line_search=True,
    )
    result = searched.relax_etas([0.4, 0.6], steps=1)

    assert all(math.isfinite(value) for value in result)
    assert searched.last_relaxation_metrics()["accepted_steps"] == 1


def test_precision_cache_uses_nonnegative_curvature_magnitude_for_signed_term_weight() -> None:
    coordinator = EnergyCoordinator(
        modules=[QuadraticModule(0.5, 2.0)],
        couplings=[],
        constraints={"term_weights": {"local:QuadraticModule": -3.0}},
        use_precision_preconditioning=True,
    )

    snapshot = coordinator.inspect_state([0.4])

    assert snapshot.precision_diagonal == (6.0,)
    assert snapshot.preconditioner_diagonal == (6.0,)


@pytest.mark.parametrize("invalid_weight", [math.nan, math.inf, -math.inf])
def test_nonfinite_term_weight_fails_closed(invalid_weight: float) -> None:
    coordinator = EnergyCoordinator(
        modules=[QuadraticModule(0.5, 2.0)],
        couplings=[],
        constraints={"term_weights": {"local:QuadraticModule": invalid_weight}},
    )

    with pytest.raises(ValueError, match="term weight .* must be finite"):
        coordinator.inspect_state([0.4])


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


def test_composed_gershgorin_bound_covers_random_quadratic_graph_hessians() -> None:
    for seed in range(10):
        rng = np.random.default_rng(seed)
        size = int(rng.integers(3, 10))
        stiffness = rng.uniform(0.1, 3.0, size=size)
        modules = [QuadraticModule(target=float(rng.uniform()), stiffness=float(value)) for value in stiffness]
        couplings = []
        hessian = np.diag(stiffness)
        for i in range(size):
            for j in range(i + 1, size):
                if rng.uniform() > 0.45:
                    continue
                weight = float(rng.uniform(0.05, 1.5))
                couplings.append((i, j, QuadraticCoupling(weight=weight)))
                hessian[i, i] += 2.0 * weight
                hessian[j, j] += 2.0 * weight
                hessian[i, j] -= 2.0 * weight
                hessian[j, i] -= 2.0 * weight

        coord = EnergyCoordinator(
            modules=modules,
            couplings=couplings,
            constraints={},
            use_analytic=True,
            noise_mode="none",
            stability_guard=True,
            use_stiffness_updates=False,
            use_precision_preconditioning=False,
        )
        state = [float(value) for value in rng.uniform(0.0, 1.0, size=size)]
        estimate = float(coord._estimate_lipschitz_bound(state))  # type: ignore[attr-defined]
        lambda_max = float(np.max(np.linalg.eigvalsh(hessian)))

        assert estimate + 1e-8 >= lambda_max
        alpha = 0.9 * 2.0 / estimate
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.eye(size) - alpha * hessian))))
        assert spectral_radius < 1.0


def test_preconditioned_gershgorin_bound_controls_implemented_iteration() -> None:
    """The guarded matrix must be the matrix used by the preconditioned update."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        size = int(rng.integers(3, 10))
        stiffness = rng.uniform(0.1, 3.0, size=size)
        modules = [QuadraticModule(target=float(rng.uniform()), stiffness=float(value)) for value in stiffness]
        couplings = []
        hessian = np.diag(stiffness)
        for i in range(size):
            for j in range(i + 1, size):
                if rng.uniform() > 0.45:
                    continue
                weight = float(rng.uniform(0.05, 1.5))
                couplings.append((i, j, QuadraticCoupling(weight=weight)))
                hessian[i, i] += 2.0 * weight
                hessian[j, j] += 2.0 * weight
                hessian[i, j] -= 2.0 * weight
                hessian[j, i] -= 2.0 * weight

        coord = EnergyCoordinator(
            modules=modules,
            couplings=couplings,
            constraints={},
            use_analytic=True,
            noise_mode="none",
            stability_guard=True,
            use_precision_preconditioning=True,
            precision_epsilon=1e-12,
        )
        state = [float(value) for value in rng.uniform(0.0, 1.0, size=size)]
        precision = rng.uniform(0.05, 5.0, size=size)
        estimate = float(coordinator_stability.estimate_preconditioned_lipschitz_bound(coord, state, precision))
        normalized_hessian = hessian / np.sqrt(np.outer(precision, precision))
        lambda_max = float(np.max(np.linalg.eigvalsh(normalized_hessian)))

        assert estimate + 1e-8 >= lambda_max
        alpha = 0.9 * 2.0 / estimate
        iteration = np.eye(size) - alpha * (hessian / precision[:, None])
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(iteration))))
        assert spectral_radius < 1.0


@pytest.mark.parametrize("use_stiffness_updates", [False, True])
def test_small_curvature_preconditioned_counterexample_contracts_without_rejection(
    use_stiffness_updates: bool,
) -> None:
    """Regression for H=P=0.1, where a raw-Hessian 2/L cap overshoots."""
    coord = EnergyCoordinator(
        modules=[QuadraticModule(target=0.5, stiffness=0.1)],
        couplings=[],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=use_stiffness_updates,
        use_precision_preconditioning=not use_stiffness_updates,
        stiffness_epsilon=1e-12,
        precision_epsilon=1e-12,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        stability_cap_fraction=0.9,
        noise_mode="none",
        enable_orthogonal_noise=False,
    )
    initial = [0.6]
    initial_energy = float(coord.energy(initial))

    output = coord.relax_etas(initial, steps=1)

    assert getattr(coord, "_rejected_steps") == 0
    assert math.isclose(output[0], 0.42, rel_tol=0.0, abs_tol=1e-10)
    assert coord.energy(output) < initial_energy


@pytest.mark.parametrize("use_stiffness_updates", [False, True])
def test_guard_covers_step_that_crosses_inactive_hinge_boundary(
    use_stiffness_updates: bool,
) -> None:
    """The bound includes possible hinge curvature beyond the starting active set."""
    coord = EnergyCoordinator(
        modules=[
            QuadraticModule(target=0.0, stiffness=0.1),
            QuadraticModule(target=1.0, stiffness=0.1),
        ],
        couplings=[(0, 1, DirectedHingeCoupling(weight=8.0))],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=use_stiffness_updates,
        use_precision_preconditioning=not use_stiffness_updates,
        stiffness_epsilon=1e-12,
        precision_epsilon=1e-12,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        stability_cap_fraction=0.9,
        noise_mode="none",
        enable_orthogonal_noise=False,
    )
    initial = [0.5, 0.5]
    initial_energy = float(coord.energy(initial))

    output = coord.relax_etas(initial, steps=1)

    assert output[0] < initial[0]
    assert output[1] > initial[1]
    assert coord.energy(output) < initial_energy
    assert getattr(coord, "_rejected_steps") == 0


@pytest.mark.parametrize("use_stiffness_updates", [False, True])
def test_projected_armijo_uses_raw_gradient_with_preconditioned_direction(
    use_stiffness_updates: bool,
) -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(target=0.5, stiffness=0.1)],
        couplings=[],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=use_stiffness_updates,
        use_precision_preconditioning=not use_stiffness_updates,
        stiffness_epsilon=1e-12,
        precision_epsilon=1e-12,
        stability_guard=False,
        line_search=True,
        armijo_c=0.5,
        step_size=0.5,
        noise_mode="none",
        enable_orthogonal_noise=False,
    )

    output = coord.relax_etas([0.6], steps=1)

    assert math.isclose(output[0], 0.55, rel_tol=0.0, abs_tol=1e-12)
    assert getattr(coord, "_last_acceptance_reason") == "armijo_accepted"
    assert getattr(coord, "_last_step_backtracks") == 0


@pytest.mark.parametrize("use_stiffness_updates", [False, True])
def test_projected_armijo_accepts_constrained_stationary_noop(
    use_stiffness_updates: bool,
) -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(target=-0.2, stiffness=0.1)],
        couplings=[],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=use_stiffness_updates,
        use_precision_preconditioning=not use_stiffness_updates,
        stiffness_epsilon=1e-12,
        precision_epsilon=1e-12,
        stability_guard=False,
        line_search=True,
        step_size=0.5,
        noise_mode="none",
        enable_orthogonal_noise=False,
    )

    output = coord.relax_etas([0.0], steps=1)

    assert output == [0.0]
    assert getattr(coord, "_last_acceptance_reason") == "armijo_accepted"
    assert getattr(coord, "_last_step_backtracks") == 0
    assert coord.last_relaxation_metrics()["accepted_steps"] == 1


@pytest.mark.parametrize("use_stiffness_updates", [False, True])
def test_projected_armijo_exhaustion_is_a_rejected_no_step(
    use_stiffness_updates: bool,
) -> None:
    coord = EnergyCoordinator(
        modules=[QuadraticModule(target=0.5, stiffness=0.1)],
        couplings=[],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=use_stiffness_updates,
        use_precision_preconditioning=not use_stiffness_updates,
        stiffness_epsilon=1e-12,
        precision_epsilon=1e-12,
        stability_guard=False,
        line_search=True,
        armijo_c=0.9,
        max_backtrack=0,
        step_size=0.5,
        noise_mode="none",
        enable_orthogonal_noise=False,
    )

    output = coord.relax_etas([0.6], steps=1)
    metrics = coord.last_relaxation_metrics()

    assert output == [0.6]
    assert getattr(coord, "_last_acceptance_reason") == "armijo_failed_no_step"
    assert getattr(coord, "_last_step_backtracks") == 0
    assert metrics["accepted_steps"] == 0
    assert metrics["rejected_steps"] == 1
    assert metrics["last_acceptance_reason"] == "armijo_failed_no_step"
    assert metrics["acceptance_reasons"] == ["armijo_failed_no_step"]


@pytest.mark.parametrize("use_stiffness_updates", [False, True])
@pytest.mark.parametrize("curvature,epsilon,scale", [(0.1, 1e-12, 100.0), (0.1, 0.2, 10.0)])
def test_preconditioned_guard_is_invariant_to_joint_curvature_scaling(
    use_stiffness_updates: bool,
    curvature: float,
    epsilon: float,
    scale: float,
) -> None:
    """Scaling H and P together must not change the dimensionless trajectory."""

    def one_step(local_curvature: float, floor: float) -> tuple[float, int, float, float]:
        coord = EnergyCoordinator(
            modules=[QuadraticModule(target=0.5, stiffness=local_curvature)],
            couplings=[],
            constraints={},
            use_analytic=True,
            use_stiffness_updates=use_stiffness_updates,
            use_precision_preconditioning=not use_stiffness_updates,
            stiffness_epsilon=floor,
            precision_epsilon=floor,
            stability_guard=True,
            auto_step_from_lipschitz=True,
            stability_cap_fraction=0.9,
            noise_mode="none",
            enable_orthogonal_noise=False,
        )
        snapshot = coord.inspect_state([0.6])
        output = coord.relax_etas([0.6], steps=1)
        return (
            float(output[0]),
            int(getattr(coord, "_rejected_steps")),
            float(snapshot.update_lipschitz_bound),
            float(snapshot.preconditioner_diagonal[0]),
        )

    base = one_step(curvature, epsilon)
    scaled = one_step(scale * curvature, scale * epsilon)

    assert base[1] == 0
    assert scaled[1] == 0
    assert math.isclose(base[0], scaled[0], rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(base[2], scaled[2], rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(scaled[3], scale * base[3], rel_tol=0.0, abs_tol=1e-10)


def test_custom_coupling_curvature_protocol_composes_end_to_end() -> None:
    custom = ScaledQuadraticCoupling(weight=1.7, scale_i=1.3, scale_j=-0.6)
    modules = [QuadraticModule(target=0.2, stiffness=0.8), QuadraticModule(target=0.7, stiffness=1.1)]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, custom)],
        constraints={},
        use_analytic=True,
        noise_mode="none",
        stability_guard=True,
    )
    state = [0.9, 0.1]
    hessian = np.array(
        [
            [0.8 + custom.weight * custom.scale_i**2, -custom.weight * custom.scale_i * custom.scale_j],
            [-custom.weight * custom.scale_i * custom.scale_j, 1.1 + custom.weight * custom.scale_j**2],
        ],
        dtype=float,
    )

    estimate = float(coord._estimate_lipschitz_bound(state))  # type: ignore[attr-defined]
    coord._update_precision_cache(state)  # type: ignore[attr-defined]
    precision = np.asarray(coord.get_precision_diagonal(), dtype=float)
    independent_precision = np.asarray([0.7, 3.2], dtype=float)
    normalized_estimate = float(
        coordinator_stability.estimate_preconditioned_lipschitz_bound(
            coord,
            state,
            independent_precision,
        )
    )
    normalized_hessian = hessian / np.sqrt(np.outer(independent_precision, independent_precision))

    assert estimate + 1e-8 >= float(np.max(np.linalg.eigvalsh(hessian)))
    assert normalized_estimate + 1e-8 >= float(np.max(np.linalg.eigvalsh(normalized_hessian)))
    assert np.allclose(precision, np.diag(hessian), rtol=0.0, atol=1e-10)
    result = coord.relax_etas(state, steps=20)
    assert coord.energy(result) <= coord.energy(state)


def test_preconditioning_converges_faster_on_ill_conditioned_problem() -> None:
    """Diagonal preconditioning beats plain gradient descent under large curvature spread.

    The energy is uncoupled, so the curvature is diagonal and axis-aligned, which
    is the regime where diagonal preconditioning is expected to help. Each run uses
    the Lipschitz bound for its own implemented iteration matrix.
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


@pytest.mark.parametrize("use_stiffness_updates", [False, True])
def test_guarded_coupled_step_matches_preconditioned_matrix_and_contracts_p_norm(
    use_stiffness_updates: bool,
) -> None:
    modules = [QuadraticModule(target=0.2, stiffness=1.0), QuadraticModule(target=0.8, stiffness=1.0)]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, QuadraticCoupling(weight=8.0))],
        constraints={},
        use_analytic=True,
        use_stiffness_updates=use_stiffness_updates,
        use_precision_preconditioning=not use_stiffness_updates,
        stiffness_epsilon=1e-12,
        precision_epsilon=1e-12,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        stability_cap_fraction=0.9,
        noise_mode="none",
        enable_orthogonal_noise=False,
    )
    initial = np.asarray([0.55, 0.45], dtype=float)
    hessian = np.asarray([[17.0, -16.0], [-16.0, 17.0]], dtype=float)
    linear = np.asarray([0.2, 0.8], dtype=float)
    optimum = np.linalg.solve(hessian, linear)
    snapshot = coord.inspect_state(initial.tolist())
    precision = np.asarray(snapshot.preconditioner_diagonal, dtype=float)
    alpha = 0.9 * 2.0 / snapshot.update_lipschitz_bound
    expected = np.clip(initial - alpha * ((hessian @ initial - linear) / precision), 0.0, 1.0)

    output = np.asarray(coord.relax_etas(initial.tolist(), steps=1), dtype=float)
    error_before = float(np.sqrt(np.sum(precision * (initial - optimum) ** 2)))
    error_after = float(np.sqrt(np.sum(precision * (output - optimum) ** 2)))

    assert np.allclose(output, expected, rtol=0.0, atol=1e-12)
    assert error_after < error_before
    assert getattr(coord, "_rejected_steps") == 0


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


def test_preconditioned_gershgorin_cap_is_a_conservative_certificate() -> None:
    """A stable step can exist above the sufficient normalized row-sum cap."""
    modules = [QuadraticModule(target=0.5, stiffness=1.0) for _ in range(3)]
    couplings = [
        (0, 1, QuadraticCoupling(weight=1.0)),
        (1, 2, QuadraticCoupling(weight=1.0)),
    ]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={},
        use_analytic=True,
        use_stiffness_updates=False,
        use_precision_preconditioning=True,
        precision_epsilon=1e-12,
        stability_guard=True,
        noise_mode="none",
        enable_orthogonal_noise=False,
    )
    state = [0.2, 0.5, 0.8]
    hessian = np.array(
        [
            [3.0, -2.0, 0.0],
            [-2.0, 5.0, -2.0],
            [0.0, -2.0, 3.0],
        ],
        dtype=float,
    )
    coord._update_precision_cache(state)  # type: ignore[attr-defined]
    precision = np.asarray(coord.get_precision_diagonal(), dtype=float)
    estimate = float(
        coordinator_stability.estimate_preconditioned_lipschitz_bound(coord, state, precision)
    )
    normalized_hessian = hessian / np.sqrt(np.outer(precision, precision))
    exact_largest = float(np.max(np.linalg.eigvalsh(normalized_hessian)))
    alpha = 0.5 * ((2.0 / estimate) + (2.0 / exact_largest))
    iteration = np.eye(3) - alpha * (hessian / precision[:, None])

    assert estimate > exact_largest
    assert alpha > 2.0 / estimate
    assert alpha < 2.0 / exact_largest
    assert float(np.max(np.abs(np.linalg.eigvals(iteration)))) < 1.0


def test_underreported_custom_curvature_is_caught_by_monotone_restoration() -> None:
    """An invalid curvature report voids the cap but not state restoration."""
    modules = [
        QuadraticModule(target=0.5, stiffness=1.0),
        QuadraticModule(target=0.5, stiffness=1.0),
    ]
    coupling = UnderreportedQuadraticCoupling(weight=8.0, report_fraction=0.1)
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[(0, 1, coupling)],
        constraints={},
        use_analytic=True,
        use_precision_preconditioning=False,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        stability_cap_fraction=0.9,
        noise_mode="none",
        assert_monotonic_energy=True,
        continue_after_rejection=True,
    )
    initial = [0.9, 0.1]
    estimated_l = float(coord._estimate_lipschitz_bound(initial))  # type: ignore[attr-defined]
    true_l = 33.0

    assert estimated_l < true_l
    assert 0.9 * 2.0 / estimated_l > 2.0 / true_l

    output = coord.relax_etas(list(initial), steps=5)
    metrics = coord.last_relaxation_metrics()

    assert np.allclose(output, initial, rtol=0.0, atol=1e-12)
    assert metrics["accepted_steps"] == 0
    assert metrics["rejected_steps"] == 5
