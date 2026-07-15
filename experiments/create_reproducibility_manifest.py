"""Create a machine-readable manifest for recorded publication artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence


ARTIFACTS = (
    "logs/pson_noise_ablation.csv",
    "logs/pson_noise_ablation_summary.csv",
    "logs/pson_escape_trials.csv",
    "logs/pson_escape_summary.csv",
    "logs/pson_analytic_reference.csv",
    "logs/curvature_contract_audit.csv",
    "logs/scaling_benchmark.csv",
    "logs/scaling_model.json",
    "docs/figures/pson_cost_reduction.png",
    "docs/figures/pson_escape_rate.png",
    "docs/figures/runtime_scaling.png",
)

COMMANDS = (
    "python -m pytest -q",
    "python -m experiments.audit_curvature_contract --samples 32 --strict",
    "python -m experiments.validate_pson_reference --samples 100000",
    "python -m experiments.ablate_pson_noise --trials 30 --steps 80 --noise-cost-samples 32 --bootstrap-samples 10000",
    "python -m experiments.benchmark_pson_escape --trials 200 --steps 40 --bootstrap-samples 10000",
    "python -m experiments.benchmark_scaling --sizes 16 64 256 --edge-factors 1 4 16 --repeats 7 --warmups 2 --steps 20 --environment-label windows_python_3_12_local",
    "python -m experiments.plots.plot_publication_results",
)


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _artifact(root: Path, relative_path: str) -> Dict[str, Any]:
    path = root / relative_path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record: Dict[str, Any] = {
        "path": relative_path.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            record["data_rows"] = sum(1 for _ in csv.reader(handle)) - 1
    return record


def build_manifest(root: Path, artifact_paths: Sequence[str] = ARTIFACTS) -> Dict[str, Any]:
    existing = [path for path in artifact_paths if (root / path).is_file()]
    status = _git(root, "status", "--porcelain")
    dependencies = {}
    for package in ("numpy", "polars", "matplotlib", "pytest"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "worktree_dirty": bool(status),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "dependencies": dependencies,
        },
        "protocol": {
            "pson_problem_seeds": 30,
            "noise_draws_per_seed": 32,
            "hierarchical_bootstrap_resamples": 10_000,
            "escape_trials": 200,
            "escape_steps": 40,
            "scaling_sizes": [16, 64, 256],
            "scaling_edge_factors": [1, 4, 16],
            "scaling_repeats": 7,
            "scaling_warmups": 2,
            "scaling_process_isolation": "one fresh Python process per size and edge-factor case",
        },
        "commands": list(COMMANDS),
        "artifacts": [_artifact(root, path) for path in existing],
        "deferred": [
            "real_model_evaluation",
            "archival_release_and_doi",
            "independent_external_reproduction",
        ],
    }


def verify_manifest(
    root: Path,
    manifest_path: Path,
    artifact_paths: Sequence[str] = ARTIFACTS,
) -> list[str]:
    """Return integrity errors for the recorded publication artifacts."""
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read manifest: {exc}"]
    recorded = {
        str(item.get("path", "")): item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    errors: list[str] = []
    for relative_path in artifact_paths:
        normalized = relative_path.replace("\\", "/")
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing artifact: {normalized}")
            continue
        expected = recorded.get(normalized)
        if expected is None:
            errors.append(f"artifact absent from manifest: {normalized}")
            continue
        actual = _artifact(root, relative_path)
        for field in ("sha256", "bytes", "data_rows"):
            if field in actual and actual[field] != expected.get(field):
                errors.append(
                    f"artifact {normalized} has {field}={actual[field]!r}, "
                    f"expected {expected.get(field)!r}"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/reproducibility_manifest.json"))
    parser.add_argument("--check", action="store_true", help="Verify recorded artifact hashes without rewriting the manifest.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    if args.check:
        errors = verify_manifest(root, output)
        if errors:
            raise SystemExit("reproducibility manifest check failed:\n" + "\n".join(errors))
        print(f"verified reproducibility manifest {output.relative_to(root)}")
        return
    manifest = build_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote reproducibility manifest to {output.relative_to(root)}")


if __name__ == "__main__":
    main()
