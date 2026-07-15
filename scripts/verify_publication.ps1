param(
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$gitSafeDirectory = $repoRoot.Replace("\", "/")
$gitPrefix = @("-c", "safe.directory=$gitSafeDirectory")

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)] [string] $Executable,
        [Parameter(Mandatory = $true)] [string[]] $Arguments,
        [Parameter(Mandatory = $true)] [string] $Description
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

Push-Location $repoRoot
try {
    if (Test-Path ".github\workflows") {
        $workflowFiles = Get-ChildItem ".github\workflows" -File -Recurse
        if ($workflowFiles) {
            throw "GitHub workflow files are prohibited for this project."
        }
    }

    $requiredArtifacts = @(
        "logs\pson_noise_ablation.csv",
        "logs\pson_noise_ablation_summary.csv",
        "logs\pson_escape_trials.csv",
        "logs\pson_escape_summary.csv",
        "logs\pson_analytic_reference.csv",
        "logs\curvature_contract_audit.csv",
        "logs\scaling_benchmark.csv",
        "logs\scaling_model.json"
    )
    $requiredFigures = @(
        "docs\figures\pson_cost_reduction.png",
        "docs\figures\pson_escape_rate.png",
        "docs\figures\runtime_scaling.png"
    )
    $manifestPath = "logs\reproducibility_manifest.json"
    $recreatedDuringVerification = @(
        "logs\pson_analytic_reference.csv",
        "logs\curvature_contract_audit.csv"
    ) + $requiredFigures
    $recordedHashes = @{}
    if (-not $Regenerate) {
        foreach ($path in ($requiredArtifacts + $requiredFigures + @($manifestPath))) {
            if (-not (Test-Path $path -PathType Leaf)) {
                throw "Missing recorded publication artifact: $path. Run with -Regenerate."
            }
        }
        foreach ($path in $recreatedDuringVerification) {
            $recordedHashes[$path] = (Get-FileHash $path -Algorithm SHA256).Hash
        }
    }

    Invoke-CheckedNative "uv" @("lock", "--check") "Lockfile check"
    Invoke-CheckedNative "uv" @("run", "python", "-m", "pytest", "-q") "Test suite"
    Invoke-CheckedNative "uv" @("run", "python", "-m", "experiments.audit_curvature_contract", "--samples", "32", "--strict") "Curvature-contract audit"
    Invoke-CheckedNative "uv" @("run", "python", "-m", "experiments.validate_pson_reference", "--samples", "100000") "PSON analytic reference"

    if ($Regenerate) {
        Invoke-CheckedNative "uv" @("run", "python", "-m", "experiments.ablate_pson_noise", "--trials", "30", "--steps", "80", "--noise-cost-samples", "32", "--bootstrap-samples", "10000") "PSON ablation regeneration"
        Invoke-CheckedNative "uv" @("run", "python", "-m", "experiments.benchmark_pson_escape", "--trials", "200", "--steps", "40", "--bootstrap-samples", "10000") "PSON escape regeneration"
        Invoke-CheckedNative "uv" @("run", "python", "-m", "experiments.benchmark_scaling", "--sizes", "16", "64", "256", "--edge-factors", "1", "4", "16", "--repeats", "7", "--warmups", "2", "--steps", "20", "--environment-label", "windows_python_3_12_local") "Scaling benchmark regeneration"
    }

    foreach ($artifact in $requiredArtifacts) {
        if (-not (Test-Path $artifact -PathType Leaf)) {
            throw "Missing publication artifact: $artifact. Run with -Regenerate."
        }
    }

    Invoke-CheckedNative "uv" @("run", "--extra", "plots", "python", "-m", "experiments.plots.plot_publication_results") "Publication figure generation"

    foreach ($figure in $requiredFigures) {
        if (-not (Test-Path $figure -PathType Leaf)) {
            throw "Missing publication figure: $figure."
        }
    }

    if (-not $Regenerate) {
        foreach ($path in $recreatedDuringVerification) {
            $actualHash = (Get-FileHash $path -Algorithm SHA256).Hash
            if ($actualHash -ne $recordedHashes[$path]) {
                throw "Regeneration changed recorded publication artifact: $path. Run with -Regenerate and review the evidence diff."
            }
        }
    }

    if ($Regenerate) {
        Invoke-CheckedNative "uv" @("run", "python", "-m", "experiments.create_reproducibility_manifest") "Reproducibility manifest generation"
    }
    if (-not (Test-Path $manifestPath -PathType Leaf)) {
        throw "Missing reproducibility manifest."
    }
    Invoke-CheckedNative "uv" @("run", "python", "-m", "experiments.create_reproducibility_manifest", "--check") "Reproducibility manifest integrity check"

    $candidateFiles = & git @gitPrefix ls-files --cached --others --exclude-standard |
        Where-Object { Test-Path $_ -PathType Leaf }
    if ($LASTEXITCODE -ne 0) {
        throw "Repository file enumeration failed with exit code $LASTEXITCODE."
    }
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

    Invoke-CheckedNative "git" ($gitPrefix + @("diff", "--check")) "Whitespace validation"
    if ($Regenerate) {
        Write-Host "Publication artifacts regenerated and verification passed. Review the resulting evidence diff before publication."
    }
    else {
        Write-Host "Publication verification passed without artifact drift."
    }
}
finally {
    Pop-Location
}
