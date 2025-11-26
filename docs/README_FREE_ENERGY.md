# Free-Energy Stability Guard (F = U - TS)

Status: Optional Feature (Phase 2)  
Scope: Acceptance criteria based on thermodynamic free energy.

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

- **Allow Entropic Exploration**: At high \(T\), the system tolerates small increases in \(U\) if they lead to significantly higher entropy \(S\) (exploration of wider basins).
- **Avoid Premature Collapse**: Discourages the system from collapsing to low-entropy corner solutions (\(\eta \approx 0\) or \(1\)) too early if the energy gain isn't sufficient.
- **Thermodynamic Consistency**: Aligns the optimization process with true physical relaxation at finite temperature.

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
- **Observability**: When enabled, `RelaxationTracker` and `EnergyBudgetTracker` logs will include `U_internal_energy`, `S_entropy`, and `F_free_energy`.
- **Interaction**: Replaces the standard `assert_monotonic_energy` check.

---

## 6. Tuning Guidance

- **T = 0**: Equivalent to standard energy minimization (greedy).
- **Small T (0.01 - 0.1)**: Slight regularization against boundary collapse; cleaner gradients near 0/1.
- **Large T (1.0+)**: Strong preference for uncertainty (\(\eta \approx 0.5\)); system will only commit to 0 or 1 if constraints (\(U\)) are very strong. Useful for annealing.

