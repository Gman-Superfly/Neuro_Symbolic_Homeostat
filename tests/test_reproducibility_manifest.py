from __future__ import annotations

from pathlib import Path

from experiments.create_reproducibility_manifest import build_manifest


def test_manifest_uses_relative_artifact_paths_and_checksums() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = build_manifest(root, artifact_paths=("pyproject.toml",))
    artifact = manifest["artifacts"][0]

    assert artifact["path"] == "pyproject.toml"
    assert len(artifact["sha256"]) == 64
    assert artifact["bytes"] > 0
    assert "real_model_evaluation" in manifest["deferred"]
