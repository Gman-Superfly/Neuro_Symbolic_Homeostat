# Free-Energy Stability Guard (F = U - TS)

Status: Optional Feature (Phase 2)
Scope: Optional acceptance proxy based on (U-TS).

---

## 1. Concept

Standard stability guards enforce monotonicity in **Internal Energy** (\(U\)):
\[ U_{t+1} \le U_t \]

The **Free-Energy Guard** enforces monotonicity in **Helmholtz Free Energy** (\(F\)):
\[ F = U - T \cdot S \]
\[ F_{t+1} \le F_t \]

Where:
- \(U\): Total energy (locals + couplings).
- \(T\): Temperature parameter.
- \(S\): Entropy of the order parameter distribution.

---

## 2. Why Use It?

- **Entropy-aware acceptance**: At positive \(T\), the guard can accept an increase in \(U\) when the configured Bernoulli entropy increase lowers \(U-TS\).
- **Boundary preference**: The entropy term makes midpoint states cheaper than corner states in the acceptance proxy, subject to the competing internal-energy terms.
- **Mechanism boundary**: The proposal direction still comes from the configured internal-energy gradient and noise path. The implementation does not differentiate \(U-TS\), simulate physical finite-temperature dynamics, or establish thermodynamic consistency.

---

## 3. Entropy Definition

We approximate Shannon entropy for order parameters \(\eta \in [0,1]\) using a Bernoulli-like form:

\[
S(\eta) = - \left( \eta \ln(\eta) + (1-\eta) \ln(1-\eta) \right)
\]

Total entropy is the sum over all modules: \(S_{total} = \sum_i S(\eta_i)\).

*Note: Values are clamped slightly away from 0 and 1 to avoid \(\ln(0)\).*

---

## 4. Usage

Enable in `EnergyCoordinator`:

```python
coord = EnergyCoordinator(
    ...,
    use_free_energy_guard=True,
    free_energy_temperature=1.0,  # Higher T = more exploration
    free_energy_epsilon=1e-6,     # Tolerance
)
```

When enabled, `relax_etas` will:
1.  Compute \(F_{prev} = U_{prev} - T \cdot S_{prev}\).
2.  Take a step (gradient + noise).
3.  Compute \(F_{curr} = U_{curr} - T \cdot S_{curr}\).
4.  **Accept** only if \(F_{curr} \le F_{prev} + \varepsilon\).

---

## 5. Implementation Details

- **Location**: `core/coordinator.py` (`_compute_entropy`, `_compute_free_energy`).
- **Observability**: `EnergyBudgetTracker(log_free_energy_decomposition=True)` logs `U_internal_energy`, `S_entropy`, and `F_free_energy`. `RelaxationTracker` records the accepted total-energy trace.
- **Interaction**: Replaces the standard internal-energy acceptance check. The separate deterministic monotonicity assertion is bypassed on this branch.

---

## 6. Tuning Guidance

- **T = 0**: The acceptance proxy reduces to internal energy \(U\).
- **Small positive T**: Entropy receives a small weight in the acceptance decision. This does not change the proposal gradient.
- **Larger T**: The acceptance proxy increasingly favors states near \(\eta=0.5\). The useful scale depends on the magnitude of \(U\), so it requires an explicit sweep rather than a universal threshold.
