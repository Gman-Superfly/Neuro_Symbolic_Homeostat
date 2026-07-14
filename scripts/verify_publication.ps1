param(
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$gitSafeDirectory = $repoRoot.Replace("\", "/")
$gitPrefix = @("-c", "safe.directory=$gitSafeDirectory")
Push-Location $repoRoot
try {
    if (Test-Path ".github\workflows") {
        $workflowFiles = Get-ChildItem ".github\workflows" -File -Recurse
        if ($workflowFiles) {
            throw "GitHub workflow files are prohibited for this project."
        }
    }

    uv lock --check
    uv run python -m pytest -q
    uv run python -m experiments.audit_curvature_contract --samples 32 --strict
    uv run python -m experiments.validate_pson_reference --samples 100000

    if ($Regenerate) {
        uv run python -m experiments.ablate_pson_noise --trials 30 --steps 80 --noise-cost-samples 32 --bootstrap-samples 10000
        uv run python -m experiments.benchmark_pson_escape --trials 200 --steps 40 --bootstrap-samples 10000
        uv run python -m experiments.benchmark_scaling --sizes 16 64 256 --edge-factors 1 4 16 --repeats 7 --warmups 2 --steps 20 --environment-label windows_python_3_12_local
    }

    $requiredArtifacts = @(
        "logs\pson_noise_ablation.csv",
        "logs\pson_noise_ablation_summary.csv",
        "logs\pson_escape_trials.csv",
        "logs\pson_escape_summary.csv",
        "logs\scaling_benchmark.csv"
        "logs\scaling_model.json"
    )
    foreach ($artifact in $requiredArtifacts) {
        if (-not (Test-Path $artifact -PathType Leaf)) {
            throw "Missing publication artifact: $artifact. Run with -Regenerate."
        }
    }

    uv run --extra plots python -m experiments.plots.plot_publication_results
    uv run python -m experiments.create_reproducibility_manifest

    $candidateFiles = & git @gitPrefix ls-files --cached --others --exclude-standard |
        Where-Object { Test-Path $_ -PathType Leaf }
    $privateValues = @(
        $env:USERPROFILE,
        $env:USERPROFILE.Replace("\", "/"),
        "/home/$($env:USERNAME.ToLowerInvariant())"
    ) | Where-Object { $_ }
    foreach ($privateValue in $privateValues) {
        $matches = if ($candidateFiles) {
            Select-String -Path $candidateFiles -SimpleMatch $privateValue -ErrorAction SilentlyContinue
        }
        if ($matches) {
            $matches | Format-Table Path, LineNumber, Line
            throw "Private local path found in repository content."
        }
    }

    & git @gitPrefix diff --check
    Write-Host "Publication verification passed."
}
finally {
    Pop-Location
}
