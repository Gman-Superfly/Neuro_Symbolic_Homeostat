from __future__ import annotations

import json
from pathlib import Path

from experiments.create_reproducibility_manifest import build_manifest, verify_manifest


def test_manifest_uses_relative_artifact_paths_and_checksums() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = build_manifest(root, artifact_paths=("pyproject.toml",))
    artifact = manifest["artifacts"][0]

    assert artifact["path"] == "pyproject.toml"
    assert len(artifact["sha256"]) == 64
    assert artifact["bytes"] > 0
    assert "real_model_evaluation" in manifest["deferred"]


def test_manifest_check_detects_artifact_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("recorded\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = build_manifest(tmp_path, artifact_paths=("artifact.txt",))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_manifest(tmp_path, manifest_path, artifact_paths=("artifact.txt",)) == []

    artifact.write_text("changed\n", encoding="utf-8")
    errors = verify_manifest(tmp_path, manifest_path, artifact_paths=("artifact.txt",))

    assert any("sha256" in error for error in errors)
