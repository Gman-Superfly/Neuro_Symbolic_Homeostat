"""Demonstrate counterfactual gate-benefit coupling.

This script shows how GateBenefitCoupling creates a gradient force
on a closed gate (η=0) from a caller-supplied benefit estimate.
The benefit value is hard-coded here to isolate the gradient property.
"""

from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling, QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


def demo_wormhole_effect() -> None:
    """Show how closed gates (η=0) receive gradients from supplied benefit."""
    
    print("=" * 70)
    print("CGBC DEMONSTRATION (WORMHOLE NICKNAME)")
    print("=" * 70)
    print("This demo hard-codes delta_benefit to isolate the gradient property.")
    print("It does not show the system discovering the benefit estimate.")
    print()
    
    # Create two modules: a domain module and a gate module
    domain_mod = EnergyGatingModule(
        gain_fn=lambda _: 0.0,  # simple quadratic
        a=0.3,
        b=0.2,
    )
    
    gate_mod = EnergyGatingModule(
        gain_fn=lambda _: 0.1,  # has a gain term (cost of activation)
        cost=0.05,
        a=0.2,
        b=0.1,
    )
    
    # Scenario 1: WITHOUT Wormhole (standard quadratic coupling only)
    print("SCENARIO 1: Standard Quadratic Coupling (No Wormhole)")
    print("-" * 70)
    
    coord_standard = EnergyCoordinator(
        modules=[gate_mod, domain_mod],
        couplings=[
            (0, 1, QuadraticCoupling(weight=1.0)),  # standard spring
        ],
        constraints={},
        step_size=0.05,
        line_search=False,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        enable_orthogonal_noise=True,
        auto_noise_controller=True,
        precision_aware_noise_controller=True,
        noise_magnitude=1e-2,
        noise_schedule_decay=0.99,
    )
    
    # Start with gate CLOSED
    etas_standard = [0.0, 0.5]  # gate=0 (closed), domain=0.5
    print(f"Initial: eta_gate={etas_standard[0]:.3f}, eta_domain={etas_standard[1]:.3f}")
    
    E0_standard = coord_standard.energy(etas_standard)
    print(f"Initial Energy: {E0_standard:.6f}")
    
    # Compute gradient on closed gate (finite difference manually)
    eps = 1e-5
    etas_perturb = [etas_standard[0] + eps, etas_standard[1]]
    E_perturb = coord_standard.energy(etas_perturb)
    grad_standard_0 = (E_perturb - E0_standard) / eps
    
    print(f"Gradient on gate (eta=0): {grad_standard_0:.6f}")
    print("  -> Force is small and local (only sees current mismatch)")
    print()
    
    # Relax
    etas_standard_final = coord_standard.relax_etas(etas_standard, steps=30)
    E_final_standard = coord_standard.energy(etas_standard_final)
    
    print(f"After 30 steps: eta_gate={etas_standard_final[0]:.3f}, eta_domain={etas_standard_final[1]:.3f}")
    print(f"Final Energy: {E_final_standard:.6f}")
    print(f"Energy Drop: {E0_standard - E_final_standard:.6f}")
    print("  -> Gate opens slowly, with no supplied downstream benefit signal")
    print()
    print()
    
    # Scenario 2: WITH CGBC (GateBenefitCoupling)
    print("SCENARIO 2: GateBenefitCoupling (CGBC, wormhole nickname)")
    print("-" * 70)
    
    # Caller-supplied benefit estimate for this demo. A real system would compute
    # this value from an estimator, rollout, downstream loss difference, or supervision.
    potential_benefit = 0.3  # opening gate would improve domain by 0.3
    
    coord_wormhole = EnergyCoordinator(
        modules=[gate_mod, domain_mod],
        couplings=[
            (0, 1, QuadraticCoupling(weight=0.5)),  # weaker standard coupling
            (0, 1, GateBenefitCoupling(weight=2.0, delta_key="delta_benefit")),  # CGBC
        ],
        constraints={"delta_benefit": potential_benefit},
        step_size=0.05,
        line_search=False,
        stability_guard=True,
        auto_step_from_lipschitz=True,
        enable_orthogonal_noise=True,
        auto_noise_controller=True,
        precision_aware_noise_controller=True,
        noise_magnitude=1e-2,
        noise_schedule_decay=0.99,
    )
    
    # Start with SAME initial state: gate CLOSED
    etas_wormhole = [0.0, 0.5]  # gate=0 (closed), domain=0.5
    print(f"Initial: eta_gate={etas_wormhole[0]:.3f}, eta_domain={etas_wormhole[1]:.3f}")
    print(f"Caller-supplied delta_benefit: {potential_benefit:.3f}")
    
    E0_wormhole = coord_wormhole.energy(etas_wormhole)
    print(f"Initial Energy: {E0_wormhole:.6f}")
    
    # Compute gradient on closed gate (finite difference manually)
    etas_perturb_w = [etas_wormhole[0] + eps, etas_wormhole[1]]
    E_perturb_w = coord_wormhole.energy(etas_perturb_w)
    grad_wormhole_0 = (E_perturb_w - E0_wormhole) / eps
    
    print(f"Gradient on gate (eta=0): {grad_wormhole_0:.6f}")
    print("  -> Force is non-local because it comes from the supplied benefit estimate")
    print(f"  -> Gradient magnitude ratio: {abs(grad_wormhole_0 / max(abs(grad_standard_0), 1e-9)):.1f}x stronger")
    print()
    
    # Relax
    etas_wormhole_final = coord_wormhole.relax_etas(etas_wormhole, steps=30)
    E_final_wormhole = coord_wormhole.energy(etas_wormhole_final)
    
    print(f"After 30 steps: eta_gate={etas_wormhole_final[0]:.3f}, eta_domain={etas_wormhole_final[1]:.3f}")
    print(f"Final Energy: {E_final_wormhole:.6f}")
    print(f"Energy Drop: {E0_wormhole - E_final_wormhole:.6f}")
    print("  -> Gate opens faster because the supplied benefit estimate is positive")
    print()
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY: CGBC gate-benefit effect")
    print("=" * 70)
    print(f"Without CGBC: Final eta_gate = {etas_standard_final[0]:.3f} (local)")
    print(f"With CGBC:    Final eta_gate = {etas_wormhole_final[0]:.3f} (uses supplied benefit)")
    print()
    print("KEY INSIGHT:")
    print("  Standard coupling: gradient depends on current state")
    print("  CGBC coupling: gradient depends on caller-supplied delta_benefit")
    print()
    print("  The mechanism provides a gate gradient even when eta_gate = 0.")
    print("  Benefit estimation remains the caller's responsibility.")
    print("=" * 70)


if __name__ == "__main__":
    demo_wormhole_effect()

