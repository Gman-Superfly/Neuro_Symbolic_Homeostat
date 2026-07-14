"""Typed, mutually exclusive solver configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class SolverMode(str, Enum):
    GRADIENT = "gradient"
    PROXIMAL = "proximal"
    ADMM = "admm"


@dataclass(frozen=True)
class GradientSolverConfig:
    """Select the primary guarded gradient/stiffness relaxation path."""


@dataclass(frozen=True)
class ProximalSolverConfig:
    """Configuration for the experimental proximal solver."""

    steps: int = 50
    tau: float = 0.05
    block_mode: Literal["pairwise", "star"] = "pairwise"

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("proximal steps must be positive")
        if self.tau <= 0.0:
            raise ValueError("proximal tau must be positive")
        if self.block_mode not in {"pairwise", "star"}:
            raise ValueError("proximal block_mode must be 'pairwise' or 'star'")


@dataclass(frozen=True)
class ADMMSolverConfig:
    """Configuration for the experimental ADMM-like solver."""

    steps: int = 50
    rho: float = 1.0
    step_size: float = 0.05
    gate_prox: bool = True
    gate_damping: float = 0.5

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("ADMM steps must be positive")
        if self.rho <= 0.0:
            raise ValueError("ADMM rho must be positive")
        if self.step_size <= 0.0:
            raise ValueError("ADMM step size must be positive")
        if not 0.0 <= self.gate_damping <= 1.0:
            raise ValueError("ADMM gate damping must be in [0, 1]")


@dataclass(frozen=True)
class SolverConfig:
    """Mutually exclusive solver selection and mode-specific settings."""

    mode: SolverMode = SolverMode.GRADIENT
    gradient: GradientSolverConfig = field(default_factory=GradientSolverConfig)
    proximal: ProximalSolverConfig = field(default_factory=ProximalSolverConfig)
    admm: ADMMSolverConfig = field(default_factory=ADMMSolverConfig)

    @classmethod
    def gradient_solver(cls) -> "SolverConfig":
        return cls(mode=SolverMode.GRADIENT)

    @classmethod
    def proximal_solver(
        cls,
        *,
        steps: int = 50,
        tau: float = 0.05,
        block_mode: Literal["pairwise", "star"] = "pairwise",
    ) -> "SolverConfig":
        return cls(
            mode=SolverMode.PROXIMAL,
            proximal=ProximalSolverConfig(steps=steps, tau=tau, block_mode=block_mode),
        )

    @classmethod
    def admm_solver(
        cls,
        *,
        steps: int = 50,
        rho: float = 1.0,
        step_size: float = 0.05,
        gate_prox: bool = True,
        gate_damping: float = 0.5,
    ) -> "SolverConfig":
        return cls(
            mode=SolverMode.ADMM,
            admm=ADMMSolverConfig(
                steps=steps,
                rho=rho,
                step_size=step_size,
                gate_prox=gate_prox,
                gate_damping=gate_damping,
            ),
        )
