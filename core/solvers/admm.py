"""Experimental ADMM-like relaxation with residual telemetry."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ..couplings import (
    AsymmetricHingeCoupling,
    DampedGateBenefitCoupling,
    DirectedHingeCoupling,
    GateBenefitCoupling,
    QuadraticCoupling,
)
from ..interfaces import OrderParameter, SupportsCouplingGrads, SupportsLocalEnergyGrad
from ..finite_difference import box_derivative
from ..prox_utils import prox_linear_gate
from ..solver_config import ADMMSolverConfig


def solve_admm(
    coordinator: Any,
    etas0: List[OrderParameter],
    config: ADMMSolverConfig,
) -> List[OrderParameter]:
    """Run the repository's ADMM-like splitting path with restoration."""
    rho = config.rho
    step_size = config.step_size
    etas = [float(value) for value in etas0]
    weights = coordinator._combined_term_weights()  # noqa: SLF001
    quadratic_edges: list[tuple[int, int, float]] = []
    hinge_edges: list[tuple[int, int, float, float, float]] = []
    for i, j, coupling in coordinator.couplings:
        weight = float(weights.get(f"coup:{coupling.__class__.__name__}", 1.0))
        if isinstance(coupling, QuadraticCoupling):
            quadratic_edges.append((i, j, float(coupling.weight) * weight))
        elif isinstance(coupling, DirectedHingeCoupling):
            hinge_edges.append((i, j, float(coupling.weight) * weight, 1.0, 1.0))
        elif isinstance(coupling, AsymmetricHingeCoupling):
            hinge_edges.append(
                (i, j, float(coupling.weight) * weight, float(coupling.alpha_i), float(coupling.beta_j))
            )

    s = np.zeros(len(quadratic_edges), dtype=float)
    u = np.zeros(len(quadratic_edges), dtype=float)
    hinge_s = np.zeros(len(hinge_edges), dtype=float)
    hinge_u = np.zeros(len(hinge_edges), dtype=float)
    previous_energy = coordinator._energy_value(etas)  # noqa: SLF001
    coordinator._emit_eta(etas)  # noqa: SLF001
    coordinator._emit_energy(previous_energy)  # noqa: SLF001
    primal_residuals: List[float] = []
    dual_residuals: List[float] = []
    accepted = 0
    rejected = 0

    for _ in range(config.steps):
        previous_state = list(etas)
        previous_s = s.copy()
        previous_hinge_s = hinge_s.copy()

        for index, (i, j, weight) in enumerate(quadratic_edges):
            difference = etas[i] - etas[j]
            s[index] = rho * (difference - u[index]) / (rho + 2.0 * weight)
        for index, (i, j, weight, alpha, beta) in enumerate(hinge_edges):
            gap = beta * etas[j] - alpha * etas[i]
            hinge_s[index] = max(0.0, rho * (gap - hinge_u[index]) / (rho + 2.0 * weight))

        gradients = np.zeros(len(etas), dtype=float)
        for index, (module, eta) in enumerate(zip(coordinator.modules, etas)):
            weight = float(weights.get(f"local:{module.__class__.__name__}", 1.0))
            if isinstance(module, SupportsLocalEnergyGrad):
                gradients[index] += weight * float(module.d_local_energy_d_eta(eta, coordinator.constraints))
            else:
                gradients[index] += weight * box_derivative(
                    lambda value, local_module=module: float(
                        local_module.local_energy(value, coordinator.constraints)
                    ),
                    eta,
                    coordinator.grad_eps,
                )

        for index, (i, j, _) in enumerate(quadratic_edges):
            residual = s[index] - (etas[i] - etas[j]) + u[index]
            gradients[i] -= rho * residual
            gradients[j] += rho * residual
        for index, (i, j, _, alpha, beta) in enumerate(hinge_edges):
            residual = hinge_s[index] - (beta * etas[j] - alpha * etas[i]) + hinge_u[index]
            gradients[i] += rho * residual * alpha
            gradients[j] -= rho * residual * beta

        for i, j, coupling in coordinator.couplings:
            if isinstance(coupling, (QuadraticCoupling, DirectedHingeCoupling, AsymmetricHingeCoupling)):
                continue
            weight = float(weights.get(f"coup:{coupling.__class__.__name__}", 1.0))
            if config.gate_prox and isinstance(coupling, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                continue
            if isinstance(coupling, SupportsCouplingGrads):
                grad_i, grad_j = coupling.d_coupling_energy_d_etas(etas[i], etas[j], coordinator.constraints)
            else:
                grad_i = box_derivative(
                    lambda value, edge=coupling, other=etas[j]: float(
                        edge.coupling_energy(value, other, coordinator.constraints)
                    ),
                    etas[i],
                    coordinator.grad_eps,
                )
                grad_j = box_derivative(
                    lambda value, edge=coupling, other=etas[i]: float(
                        edge.coupling_energy(other, value, coordinator.constraints)
                    ),
                    etas[j],
                    coordinator.grad_eps,
                )
            gradients[i] += weight * float(grad_i)
            gradients[j] += weight * float(grad_j)

        for index in range(len(etas)):
            etas[index] = float(np.clip(etas[index] - step_size * gradients[index], 0.0, 1.0))

        if config.gate_prox and config.gate_damping > 0.0:
            for i, j, coupling in coordinator.couplings:
                if not isinstance(coupling, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                    continue
                weight = float(weights.get(f"coup:{coupling.__class__.__name__}", 1.0))
                grad_i, _ = coupling.d_coupling_energy_d_etas(etas[i], etas[j], coordinator.constraints)
                proximal = prox_linear_gate(etas[i], -float(grad_i) * weight, step_size)
                etas[i] = float(
                    np.clip(
                        (1.0 - config.gate_damping) * etas[i] + config.gate_damping * proximal,
                        0.0,
                        1.0,
                    )
                )

        primal_terms: List[float] = []
        for index, (i, j, _) in enumerate(quadratic_edges):
            residual = s[index] - (etas[i] - etas[j])
            primal_terms.append(float(residual))
            u[index] += residual
        for index, (i, j, _, alpha, beta) in enumerate(hinge_edges):
            residual = hinge_s[index] - (beta * etas[j] - alpha * etas[i])
            primal_terms.append(float(residual))
            hinge_u[index] += residual
        dual_terms = [*(rho * (s - previous_s)), *(rho * (hinge_s - previous_hinge_s))]
        primal_residuals.append(float(np.linalg.norm(primal_terms)))
        dual_residuals.append(float(np.linalg.norm(dual_terms)))

        if coordinator.enforce_invariants:
            coordinator._check_invariants(etas)  # noqa: SLF001
        energy = coordinator._energy_value(etas)  # noqa: SLF001
        if energy > previous_energy + coordinator.monotonic_energy_tol:
            etas = previous_state
            rejected += 1
            break
        accepted += 1
        coordinator._emit_eta(etas)  # noqa: SLF001
        coordinator._emit_energy(energy)  # noqa: SLF001
        previous_energy = energy

    coordinator._last_solver_metrics = {  # noqa: SLF001
        "solver_mode": "admm",
        "attempted_steps": accepted + rejected,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "primal_residuals": primal_residuals,
        "dual_residuals": dual_residuals,
    }
    return etas
