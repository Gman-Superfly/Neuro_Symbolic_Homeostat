# Local reproducibility workflow

Status: available in this repository
Scope: local verification and regeneration of the recorded synthetic evidence

## Environment

Install the locked core, development, and plotting dependencies:

```powershell
uv sync --extra plots
```

The repository does not use GitHub Actions. Verification is an explicit local CLI workflow.

## Verification

Run tests, lockfile checks, curvature auditing, analytic PSON references, figure generation, path sanitization, manifest generation, and diff checks:

```powershell
.\scripts\verify_publication.ps1
```

This command checks that the longer recorded experiment artifacts already exist. To regenerate them first:

```powershell
.\scripts\verify_publication.ps1 -Regenerate
```

## Recorded artifacts

- `logs/pson_noise_ablation.csv`: 630 problem-family and noise-mode trials, including the 32 individual curvature-cost draws for each trial.
- `logs/pson_noise_ablation_summary.csv`: paired hierarchical bootstrap summaries over seeds and draw indices.
- `logs/pson_escape_trials.csv`: 800 controlled nonconvex escape trials.
- `logs/pson_escape_summary.csv`: paired escape-rate differences and intervals.
- `logs/pson_analytic_reference.csv`: closed-form and Monte Carlo noise-cost comparison.
- `logs/curvature_contract_audit.csv`: sampled finite-difference curvature audit.
- `logs/scaling_benchmark.csv`: raw timing samples, warmup counts, median/IQR/p95 summaries, peak traced Python memory, and runtime metadata for each size and edge-count case.
- `logs/scaling_model.json`: descriptive log-linear size and edge-count fit for the recorded environment. The fitted exponents are empirical summaries, not complexity bounds.
- `logs/reproducibility_manifest.json`: commands, protocol parameters, dependency versions, Git state, row counts, sizes, and SHA-256 checksums.

Figures under `docs/figures/` are generated directly from these CSV files by `experiments.plots.plot_publication_results`.

The scaling command runs each size and edge-factor case in a fresh Python process by default. Use `--environment-label` to identify another machine or software environment and write it to a separate output file. A cross-environment comparison requires rerunning the same matrix in that environment; the repository currently records one local environment.

## Interpretation boundary

The workflow reproduces the repository's synthetic mechanism evidence. Real-model evaluation, an archival release and DOI, and independent external reproduction are deferred until manuscript submission and external review are possible.
