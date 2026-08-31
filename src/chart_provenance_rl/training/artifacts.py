from __future__ import annotations

from pathlib import Path
from typing import Any

from chart_provenance_rl.serialization import (
    load_json,
    sha256_file,
    sha256_json,
    write_json,
)

ADAPTER_ARTIFACT_MANIFEST = "adapter_artifact_manifest.json"


def write_adapter_artifact_manifest(output_dir: Path) -> str:
    adapter_files = _adapter_files(output_dir)
    entries = [
        {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in adapter_files
    ]
    artifact_sha256 = sha256_json(entries)
    write_json(
        output_dir / ADAPTER_ARTIFACT_MANIFEST,
        {
            "schema_version": "1.0",
            "artifact_sha256": artifact_sha256,
            "files": entries,
        },
    )
    return artifact_sha256


def validate_adapter_artifact(output_dir: Path) -> str:
    manifest = load_json(output_dir / ADAPTER_ARTIFACT_MANIFEST)
    if not isinstance(manifest, dict):
        raise TypeError("Adapter artifact manifest must be an object")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Adapter artifact manifest contains no files")
    expected_names: set[str] = set()
    normalized_entries: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            raise TypeError("Adapter artifact file entries must be objects")
        name = item.get("path")
        size = item.get("size")
        expected_hash = item.get("sha256")
        if (
            not isinstance(name, str)
            or "/" in name
            or name in {"", ".", ".."}
            or not isinstance(size, int)
            or not isinstance(expected_hash, str)
        ):
            raise ValueError("Invalid adapter artifact file entry")
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Adapter artifact file is missing or unsafe: {name}")
        if path.stat().st_size != size or sha256_file(path) != expected_hash:
            raise ValueError(f"Adapter artifact file hash mismatch: {name}")
        expected_names.add(name)
        normalized_entries.append({"path": name, "size": size, "sha256": expected_hash})
    actual_names = {path.name for path in _adapter_files(output_dir)}
    if actual_names != expected_names:
        raise ValueError("Adapter artifact file set differs from its manifest")
    artifact_sha256 = sha256_json(normalized_entries)
    if manifest.get("artifact_sha256") != artifact_sha256:
        raise ValueError("Adapter artifact identity hash mismatch")
    return artifact_sha256


def _adapter_files(output_dir: Path) -> list[Path]:
    config = output_dir / "adapter_config.json"
    weights = sorted(output_dir.glob("adapter_model.*"))
    if not config.is_file() or not weights:
        raise FileNotFoundError(
            "Adapter output requires adapter_config.json and adapter_model weights"
        )
    files = [config, *weights]
    if any(path.is_symlink() for path in files):
        raise ValueError("Adapter artifact files must not be symlinks")
    return files
