"""Demonstrate the Wormhole Effect (Non-Local Gradient Teleportation).

This script shows how GateBenefitCoupling creates a gradient force
on a closed gate (η=0) based on POTENTIAL benefit, enabling escape
from local minima that would trap standard physics-based approaches.
"""

from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling, QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


def demo_wormhole_effect() -> None:
    """Show how closed gates (η=0) still feel gradient from potential benefit."""
    
    print("=" * 70)
    print("WORMHOLE EFFECT DEMONSTRATION")
    print("=" * 70)
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
    print(f"  -> Force is SMALL and LOCAL (only sees current mismatch)")
    print()
    
    # Relax
    etas_standard_final = coord_standard.relax_etas(etas_standard, steps=30)
    E_final_standard = coord_standard.energy(etas_standard_final)
    
    print(f"After 30 steps: eta_gate={etas_standard_final[0]:.3f}, eta_domain={etas_standard_final[1]:.3f}")
    print(f"Final Energy: {E_final_standard:.6f}")
    print(f"Energy Drop: {E0_standard - E_final_standard:.6f}")
    print(f"  -> Gate opens SLOWLY, no knowledge of potential benefit")
    print()
    print()
    
    # Scenario 2: WITH Wormhole (GateBenefitCoupling)
    print("SCENARIO 2: GateBenefitCoupling (WITH Wormhole)")
    print("-" * 70)
    
    # Simulate a potential benefit: if gate opens, domain improves by Delta_eta
    potential_benefit = 0.3  # opening gate would improve domain by 0.3
    
    coord_wormhole = EnergyCoordinator(
        modules=[gate_mod, domain_mod],
        couplings=[
            (0, 1, QuadraticCoupling(weight=0.5)),  # weaker standard coupling
            (0, 1, GateBenefitCoupling(weight=2.0, delta_key="delta_benefit")),  # WORMHOLE!
        ],
        constraints={"delta_benefit": potential_benefit},
        step_size=0.05,
        line_search=False,
    )
    
    # Start with SAME initial state: gate CLOSED
    etas_wormhole = [0.0, 0.5]  # gate=0 (closed), domain=0.5
    print(f"Initial: eta_gate={etas_wormhole[0]:.3f}, eta_domain={etas_wormhole[1]:.3f}")
    print(f"Potential Benefit (Delta_eta_domain if gate opens): {potential_benefit:.3f}")
    
    E0_wormhole = coord_wormhole.energy(etas_wormhole)
    print(f"Initial Energy: {E0_wormhole:.6f}")
    
    # Compute gradient on closed gate (finite difference manually)
    etas_perturb_w = [etas_wormhole[0] + eps, etas_wormhole[1]]
    E_perturb_w = coord_wormhole.energy(etas_perturb_w)
    grad_wormhole_0 = (E_perturb_w - E0_wormhole) / eps
    
    print(f"Gradient on gate (eta=0): {grad_wormhole_0:.6f}")
    print(f"  -> Force is STRONG and NON-LOCAL (feels future benefit!)")
    print(f"  -> Gradient magnitude ratio: {abs(grad_wormhole_0 / max(abs(grad_standard_0), 1e-9)):.1f}x stronger")
    print()
    
    # Relax
    etas_wormhole_final = coord_wormhole.relax_etas(etas_wormhole, steps=30)
    E_final_wormhole = coord_wormhole.energy(etas_wormhole_final)
    
    print(f"After 30 steps: eta_gate={etas_wormhole_final[0]:.3f}, eta_domain={etas_wormhole_final[1]:.3f}")
    print(f"Final Energy: {E_final_wormhole:.6f}")
    print(f"Energy Drop: {E0_wormhole - E_final_wormhole:.6f}")
    print(f"  -> Gate opens FAST, pulled by potential benefit (wormhole!)")
    print()
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY: The Wormhole Effect")
    print("=" * 70)
    print(f"Without Wormhole: Final eta_gate = {etas_standard_final[0]:.3f} (slow, local)")
    print(f"With Wormhole:    Final eta_gate = {etas_wormhole_final[0]:.3f} (fast, non-local)")
    print()
    print("KEY INSIGHT:")
    print("  Standard coupling: gradient depends on CURRENT state (local)")
    print("  Wormhole coupling: gradient depends on POTENTIAL benefit (non-local)")
    print()
    print("  The 'future' reaches back through the closed gate and pulls it open!")
    print("  This is how the system escapes local minima without random noise.")
    print("=" * 70)


if __name__ == "__main__":
    demo_wormhole_effect()

