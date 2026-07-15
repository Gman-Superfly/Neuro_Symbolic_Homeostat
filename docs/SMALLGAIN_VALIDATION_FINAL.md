# Small-Gain validation snapshot

Status: synthetic validation rerun after the coordinator publication cleanup.

## Question

The validation asks whether the current family-level allocator executes its
stated budget contract and how its energy trace compares with the analytic and
GradNorm presets in the included benchmark scenarios.

It does not test a general nonlinear small-gain theorem. The formal contraction
claim remains limited to quadratic/SPD energy with a valid Gershgorin upper
bound in the geometry of the implemented update. For a diagonal
preconditioner \(P\), that bound applies to
\(P^{-1/2}HP^{-1/2}\), not to the raw Hessian alone.

## Implementation checks

The focused tests verify that:

- family costs use the same `coup:<ClassName>` keys as gradient norms;
- the coordinator reports positive step-cap slack when the requested step
  remains below the normalized curvature cap;
- accepted allocations remain within the global spend budget;
- weight floors, ceilings, and per-step limits are respected;
- missing coupling values return the input weights unchanged;
- the quadratic/SPD iteration contracts under the tested Gershgorin cap.

Row margins remain telemetry. Edge-to-row spend booking is future work.

## Benchmark protocol

The benchmark command uses deterministic relaxation with zero noise magnitude.
The baseline scenario requests 60 steps. The dense scenario uses 16 modules and
requests 40 steps. Each configuration starts from the same scenario state.

The benchmark records two objectives. Fixed-reference energy uses the original
term weights and supports comparisons across configurations. Adaptive energy
uses the current adapter-maintained weights and describes the objective that
the corresponding solver is optimizing. Guard decisions compare baseline and
candidate states under one objective version.

ΔF90 includes the initial energy. It is the first trace index at which a run has
completed 90 percent of its own eventual energy drop. This definition measures
within-run progress and must be read alongside final energy.

## Results

| Scenario | Configuration | Reference ΔF90 | Reference final | Adaptive final | Backtracks |
|---|---:|---:|---:|---:|---:|
| Baseline | Analytic | 23 | 0.002730 | 0.002730 | 0 |
| Baseline | GradNorm | 17 | -0.002508 | -0.005015 | 10 |
| Baseline | Small-Gain | 20 | 0.005019 | -0.020208 | 9 |
| Dense | Analytic | 29 | 0.132823 | 0.132823 | 0 |
| Dense | GradNorm | 33 | 0.130522 | 0.261044 | 0 |
| Dense | Small-Gain | 31 | 0.180603 | -0.031118 | 0 |

Small-Gain reached the lowest adaptive energy in both runs because it changed
coupling-family weights. It did not reach the lowest fixed-reference energy in
either scenario. GradNorm reached the lowest fixed-reference energy in these
two runs, although the differences from the analytic baseline were small. The
result demonstrates why adaptive energies cannot rank solution quality across
different weighting policies.

## Reproduction

```powershell
uv run -m pytest tests/test_small_gain_weight_adapter.py -v
uv run python -m experiments.benchmark_delta_f90 `
  --configs analytic gradnorm smallgain --scenario baseline --steps 60
uv run python -m experiments.benchmark_delta_f90 `
  --configs analytic gradnorm smallgain --scenario dense --dense_size 16 --steps 40
```

The benchmark command appends rows to `logs/benchmark_delta_f90.csv`. Use a
distinct `--run_id` when comparing repeated runs.

## Limitations

- The scenarios are synthetic and cover a small set of module and coupling
  families.
- The value proxy is squared gradient norm, not observed counterfactual energy
  improvement.
- The allocator enforces a global spend limit and does not yet enforce row
  incidence budgets.
- Wall-clock timing is environment-dependent and is not used for the claims in
  this snapshot.
- No task-level accuracy, calibration, or noisy-benefit benchmark is included.

The evidence supports the current implementation contract and these measured
repository runs. Broader performance remains an empirical question.
