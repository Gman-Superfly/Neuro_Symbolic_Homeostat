# Observability: trackers and logging

Status: available in this repository
Scope: How to attach telemetry trackers to the EnergyCoordinator to monitor relaxation dynamics, stability, and budgets.

---

## 1. RelaxationTracker

**Purpose**: Logs the per‑step trajectory of energy and order parameters (\(\eta\)).

### Usage

```python
from core.coordinator import EnergyCoordinator
from cf_logging.observability import RelaxationTracker

# 1. Create coordinator
coord = EnergyCoordinator(...)

# 2. Create and attach tracker
tracker = RelaxationTracker(
    name="my_experiment_trace",  # Log filename/tag
    run_id="run_001",            # Identifier for this run
    log_per_eta=True             # Optional: log every eta_i value per step
)
tracker.attach(coord)

# 3. Run relaxation
coord.relax_etas(etas0, steps=50)

# 4. Flush logs (writes to CSV/output)
tracker.flush()
```

### Output columns
- `step`, `energy`, `delta_energy`
- `min_eta`, `max_eta`, `mean_eta`
- `eta:{i}` (if `log_per_eta=True`)
- `compute_cost` (seconds per step)
- `redemption_gain` (energy drop per second)

---

## 2. EnergyBudgetTracker

**Purpose**: Detailed breakdown of energy terms, gradient norms, stability margins, and entropy.

### Usage

```python
from cf_logging.observability import EnergyBudgetTracker

tracker = EnergyBudgetTracker(
    name="budget_log",
    run_id="run_001",
    warn_on_margin_shrink=True,
    log_free_energy_decomposition=True  # Log U, S, F (F=U-TS)
)
tracker.attach(coord)

# Run...
coord.relax_etas(etas0, steps=50)
tracker.flush()
```

### Output columns
- **Terms**: `energy:local:MyModule`, `energy:coup:MyCoupling`, `grad_norm:local:...`
- **Stability**: `contraction_margin`, `margin:global`, `margin:row:{i}` (if Small-Gain active)
- **Thermodynamics**: `U_internal_energy`, `S_entropy`, `F_free_energy`, `T_temperature`
- **Precision**: `precision:min`, `precision:mean`, `precision:max`
- **Events**: `monotonicity_violation`, `monotonicity_violation_count`, and `acceptance_reason`

---

## 3. Best practices

1.  **Attach once**: Attach trackers immediately after creating the coordinator.
2.  **Flush often**: Call `flush()` after each major relaxation loop or experiment block to ensure data is written.
3.  **Use run IDs**: Use unique `run_id`s to distinguish experiments in the aggregated log files.
4.  **Performance**: `log_per_eta` and detailed budget logging have overhead; disable for massive high-speed loops, enable for debugging/tuning.

---

## 4. Public state diagnostics

Experiments that need a point-in-time diagnostic should use the public snapshot API rather than coordinator cache methods:

```python
snapshot = coord.inspect_state(etas)

snapshot.energy
snapshot.gradient
snapshot.precision_diagonal
snapshot.lipschitz_bound
snapshot.term_weights
snapshot.term_gradient_norms
```

`coord.build_noise_vector(raw_noise, snapshot.gradient)` exposes the configured noise transformation for controlled ablations. Both APIs preserve the coordinator's internal cache ownership.
