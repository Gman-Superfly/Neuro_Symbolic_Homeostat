"""Experimental proximal relaxation with explicit rejected-state restoration."""

from __future__ import annotations

from typing import Any, Dict, List

from ..couplings import (
    AsymmetricHingeCoupling,
    DampedGateBenefitCoupling,
    DirectedHingeCoupling,
    GateBenefitCoupling,
    QuadraticCoupling,
)
from ..interfaces import OrderParameter, SupportsLocalEnergyGrad
from ..prox_utils import prox_asym_hinge_pair, prox_linear_gate, prox_quadratic_pair
from ..solver_config import ProximalSolverConfig


def _local_step(coordinator: Any, index: int, eta: float, tau: float, weights: Dict[str, float]) -> float:
    module = coordinator.modules[index]
    if isinstance(module, SupportsLocalEnergyGrad):
        gradient = float(module.d_local_energy_d_eta(eta, coordinator.constraints))
    else:
        base = float(module.local_energy(eta, coordinator.constraints))
        bumped = float(module.local_energy(min(1.0, eta + coordinator.grad_eps), coordinator.constraints))
        gradient = (bumped - base) / coordinator.grad_eps
    weight = float(weights.get(f"local:{module.__class__.__name__}", 1.0))
    return float(max(0.0, min(1.0, eta - tau * weight * gradient)))


def _apply_coupling(
    coordinator: Any,
    eta_i: float,
    eta_j: float,
    coupling: Any,
    tau: float,
    weights: Dict[str, float],
) -> tuple[float, float]:
    weight = float(weights.get(f"coup:{coupling.__class__.__name__}", 1.0))
    if isinstance(coupling, QuadraticCoupling):
        return prox_quadratic_pair(eta_i, eta_j, coupling.weight * weight, tau)
    if isinstance(coupling, DirectedHingeCoupling):
        return prox_asym_hinge_pair(
            eta_i, eta_j, coupling.weight * weight, alpha=1.0, beta=1.0, tau=tau
        )
    if isinstance(coupling, AsymmetricHingeCoupling):
        return prox_asym_hinge_pair(
            eta_i,
            eta_j,
            coupling.weight * weight,
            alpha=coupling.alpha_i,
            beta=coupling.beta_j,
            tau=tau,
        )
    if isinstance(coupling, (GateBenefitCoupling, DampedGateBenefitCoupling)):
        gradient, _ = coupling.d_coupling_energy_d_etas(eta_i, eta_j, coordinator.constraints)
        eta_i = prox_linear_gate(eta_i, -float(gradient) * weight, tau)
    return eta_i, eta_j


def _apply_pairwise_couplings(coordinator: Any, etas: List[float], tau: float, weights: Dict[str, float]) -> None:
    for i, j, coupling in coordinator.couplings:
        etas[i], etas[j] = _apply_coupling(coordinator, etas[i], etas[j], coupling, tau, weights)


def _star_sweep(coordinator: Any, etas: List[float], tau: float, weights: Dict[str, float]) -> None:
    coordinator._ensure_adjacency(len(etas))  # noqa: SLF001
    assert coordinator._adjacency is not None  # noqa: SLF001
    for center in range(len(coordinator.modules)):
        block = sorted({center, *[neighbor for neighbor, _ in coordinator._adjacency[center]]})  # noqa: SLF001
        updated = {index: _local_step(coordinator, index, etas[index], tau, weights) for index in block}
        for i, j, coupling in coordinator.couplings:
            if i not in updated or j not in updated:
                continue
            updated[i], updated[j] = _apply_coupling(
                coordinator,
                updated[i],
                updated[j],
                coupling,
                tau,
                weights,
            )
        for index, value in updated.items():
            etas[index] = float(max(0.0, min(1.0, value)))


def solve_proximal(
    coordinator: Any,
    etas0: List[OrderParameter],
    config: ProximalSolverConfig,
) -> List[OrderParameter]:
    """Run proximal relaxation and restore any rejected candidate."""
    etas = [float(value) for value in etas0]
    previous_energy = coordinator._energy_value(etas)  # noqa: SLF001
    coordinator._emit_eta(etas)  # noqa: SLF001
    coordinator._emit_energy(previous_energy)  # noqa: SLF001
    accepted = 0
    rejected = 0
    for _ in range(config.steps):
        previous_state = list(etas)
        weights = coordinator._combined_term_weights()  # noqa: SLF001
        if config.block_mode == "star":
            _star_sweep(coordinator, etas, config.tau, weights)
        else:
            for index in range(len(coordinator.modules)):
                etas[index] = _local_step(coordinator, index, etas[index], config.tau, weights)
            _apply_pairwise_couplings(coordinator, etas, config.tau, weights)
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
        "solver_mode": "proximal",
        "attempted_steps": accepted + rejected,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
    }
    return etas
