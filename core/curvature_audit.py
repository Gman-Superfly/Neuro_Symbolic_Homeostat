"""Finite-difference auditing for module and coupling curvature contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Mapping, Sequence, Tuple

import math

from .couplings import (
    AsymmetricHingeCoupling,
    DirectedHingeCoupling,
    GateBenefitCoupling,
    QuadraticCoupling,
)
from .interfaces import SupportsCouplingCurvature, SupportsCouplingGrads, SupportsLocalEnergyGrad, SupportsPrecision


@dataclass(frozen=True)
class CurvatureAuditRecord:
    """Observed and reported curvature for one component at one state."""

    component_type: str
    component_index: int
    component_name: str
    eta_i: float
    eta_j: float | None
    observed_diag_i: float
    observed_diag_j: float | None
    observed_offdiag: float | None
    reported_diag_i: float | None
    reported_diag_j: float | None
    reported_offdiag: float | None
    status: str
    maximum_underestimate_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _window(value: float, step: float) -> Tuple[float, float]:
    lo = max(0.0, min(1.0, value - step))
    hi = max(0.0, min(1.0, value + step))
    assert hi > lo, "finite-difference window collapsed"
    return lo, hi


def _module_gradient(module: Any, eta: float, constraints: Mapping[str, Any], step: float) -> float:
    if isinstance(module, SupportsLocalEnergyGrad):
        return float(module.d_local_energy_d_eta(eta, constraints))
    lo, hi = _window(eta, step)
    return float(module.local_energy(hi, constraints) - module.local_energy(lo, constraints)) / (hi - lo)


def _coupling_gradients(
    coupling: Any,
    eta_i: float,
    eta_j: float,
    constraints: Mapping[str, Any],
    step: float,
) -> Tuple[float, float]:
    if isinstance(coupling, SupportsCouplingGrads):
        gi, gj = coupling.d_coupling_energy_d_etas(eta_i, eta_j, constraints)
        return float(gi), float(gj)
    i_lo, i_hi = _window(eta_i, step)
    j_lo, j_hi = _window(eta_j, step)
    gi = (
        coupling.coupling_energy(i_hi, eta_j, constraints)
        - coupling.coupling_energy(i_lo, eta_j, constraints)
    ) / (i_hi - i_lo)
    gj = (
        coupling.coupling_energy(eta_i, j_hi, constraints)
        - coupling.coupling_energy(eta_i, j_lo, constraints)
    ) / (j_hi - j_lo)
    return float(gi), float(gj)


def _reported_coupling_bounds(
    coupling: Any,
    eta_i: float,
    eta_j: float,
    constraints: Mapping[str, Any],
) -> Tuple[float, float, float] | None:
    if isinstance(coupling, QuadraticCoupling):
        value = 2.0 * float(coupling.weight)
        return value, value, value
    if isinstance(coupling, DirectedHingeCoupling):
        value = 2.0 * float(coupling.weight) if eta_j > eta_i else 0.0
        return value, value, value
    if isinstance(coupling, AsymmetricHingeCoupling):
        gap = float(coupling.beta_j) * eta_j - float(coupling.alpha_i) * eta_i
        if gap <= 0.0:
            return 0.0, 0.0, 0.0
        weight = 2.0 * float(coupling.weight)
        return (
            weight * float(coupling.alpha_i) ** 2,
            weight * float(coupling.beta_j) ** 2,
            weight * abs(float(coupling.alpha_i) * float(coupling.beta_j)),
        )
    if isinstance(coupling, GateBenefitCoupling):
        return 0.0, 0.0, 0.0
    if isinstance(coupling, SupportsCouplingCurvature):
        values = coupling.coupling_curvature_bounds(eta_i, eta_j, constraints)
        return tuple(float(value) for value in values)
    return None


def _status_and_ratio(observed: Sequence[float], reported: Sequence[float] | None, tolerance: float) -> Tuple[str, float]:
    if reported is None:
        return "unreported", float("nan")
    ratios = [obs / max(rep, tolerance) for obs, rep in zip(observed, reported) if obs > tolerance]
    maximum_ratio = max(ratios, default=0.0)
    underreported = any(obs > rep + tolerance * max(1.0, obs) for obs, rep in zip(observed, reported))
    return ("underreported" if underreported else "covered"), float(maximum_ratio)


def audit_curvature_contract(
    coordinator: Any,
    states: Sequence[Sequence[float]],
    *,
    finite_difference_step: float = 1e-5,
    tolerance: float = 1e-4,
) -> List[CurvatureAuditRecord]:
    """Audit reported curvature against sampled finite-difference Hessians."""
    assert finite_difference_step > 0.0
    assert tolerance > 0.0
    records: List[CurvatureAuditRecord] = []
    for state in states:
        values = [float(value) for value in state]
        assert len(values) == len(coordinator.modules), "state dimension must match modules"
        for index, (module, eta) in enumerate(zip(coordinator.modules, values)):
            lo, hi = _window(eta, finite_difference_step)
            observed = abs(
                (_module_gradient(module, hi, coordinator.constraints, finite_difference_step)
                 - _module_gradient(module, lo, coordinator.constraints, finite_difference_step))
                / (hi - lo)
            )
            reported_value = abs(float(module.curvature(eta))) if isinstance(module, SupportsPrecision) else None
            reported = None if reported_value is None else [reported_value]
            status, ratio = _status_and_ratio([observed], reported, tolerance)
            records.append(
                CurvatureAuditRecord(
                    "module",
                    index,
                    module.__class__.__name__,
                    eta,
                    None,
                    float(observed),
                    None,
                    None,
                    reported_value,
                    None,
                    None,
                    status,
                    ratio,
                )
            )

        for edge_index, (i, j, coupling) in enumerate(coordinator.couplings):
            eta_i, eta_j = values[i], values[j]
            i_lo, i_hi = _window(eta_i, finite_difference_step)
            j_lo, j_hi = _window(eta_j, finite_difference_step)
            gi_lo, _ = _coupling_gradients(coupling, i_lo, eta_j, coordinator.constraints, finite_difference_step)
            gi_hi, _ = _coupling_gradients(coupling, i_hi, eta_j, coordinator.constraints, finite_difference_step)
            _, gj_lo = _coupling_gradients(coupling, eta_i, j_lo, coordinator.constraints, finite_difference_step)
            _, gj_hi = _coupling_gradients(coupling, eta_i, j_hi, coordinator.constraints, finite_difference_step)
            gi_j_lo, _ = _coupling_gradients(coupling, eta_i, j_lo, coordinator.constraints, finite_difference_step)
            gi_j_hi, _ = _coupling_gradients(coupling, eta_i, j_hi, coordinator.constraints, finite_difference_step)
            _, gj_i_lo = _coupling_gradients(coupling, i_lo, eta_j, coordinator.constraints, finite_difference_step)
            _, gj_i_hi = _coupling_gradients(coupling, i_hi, eta_j, coordinator.constraints, finite_difference_step)
            observed = [
                abs((gi_hi - gi_lo) / (i_hi - i_lo)),
                abs((gj_hi - gj_lo) / (j_hi - j_lo)),
                0.5 * abs(
                    (gi_j_hi - gi_j_lo) / (j_hi - j_lo)
                    + (gj_i_hi - gj_i_lo) / (i_hi - i_lo)
                ),
            ]
            reported_bounds = _reported_coupling_bounds(coupling, eta_i, eta_j, coordinator.constraints)
            reported = None if reported_bounds is None else [abs(value) for value in reported_bounds]
            status, ratio = _status_and_ratio(observed, reported, tolerance)
            records.append(
                CurvatureAuditRecord(
                    "coupling",
                    edge_index,
                    coupling.__class__.__name__,
                    eta_i,
                    eta_j,
                    observed[0],
                    observed[1],
                    observed[2],
                    None if reported is None else reported[0],
                    None if reported is None else reported[1],
                    None if reported is None else reported[2],
                    status,
                    ratio,
                )
            )
    assert all(math.isfinite(record.observed_diag_i) for record in records)
    return records
