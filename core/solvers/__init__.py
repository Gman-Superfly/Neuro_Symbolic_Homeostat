"""Solver implementations dispatched by ``EnergyCoordinator``."""

from .admm import solve_admm
from .proximal import solve_proximal

__all__ = ["solve_admm", "solve_proximal"]
