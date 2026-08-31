from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any

from chart_provenance_rl.audits import audit_dataset
from chart_provenance_rl.constants import MODEL_ID, MODEL_LICENSE, MODEL_REVISION
from chart_provenance_rl.generation.config import default_config, smoke_config
from chart_provenance_rl.generation.generator import generate_dataset
from chart_provenance_rl.serialization import sha256_file, write_json

MODEL_SOURCE_MANIFEST = "chart_provenance_model_source.json"


def main() -> None:
    args = _parser().parse_args()
    prepare_assets(
        dataset_output=args.dataset_output,
        model_output=args.model_output,
        profile=args.profile,
    )


def prepare_assets(
    *,
    dataset_output: Path,
    model_output: Path,
    profile: str,
) -> None:
    from huggingface_hub import HfApi, snapshot_download

    dataset_root = dataset_output / "benchmark"
    config = smoke_config() if profile == "smoke" else default_config()
    generate_dataset(dataset_root, config, overwrite=dataset_root.exists())
    audit = audit_dataset(dataset_root)
    if audit["status"] != "pass":
        raise RuntimeError(f"Prepared dataset failed audit: {audit['errors']}")
    write_json(dataset_root / "audit_report.json", audit)
    if model_output.is_symlink() or (
        model_output.exists() and any(path.is_symlink() for path in model_output.rglob("*"))
    ):
        raise ValueError("Model snapshot output must not contain symlinks")
    if (model_output / MODEL_SOURCE_MANIFEST).is_file():
        validate_model_snapshot(model_output, verify_hashes=True)
        return
    model_output.mkdir(parents=True, exist_ok=True)
    model_info = HfApi().model_info(
        MODEL_ID,
        revision=MODEL_REVISION,
        files_metadata=True,
    )
    if model_info.sha != MODEL_REVISION or model_info.siblings is None:
        raise RuntimeError("Hugging Face returned an unexpected model revision")
    expected_paths = {
        sibling.rfilename for sibling in model_info.siblings if isinstance(sibling.rfilename, str)
    }
    if not expected_paths:
        raise RuntimeError("Hugging Face model revision contains no files")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=model_output,
    )
    if any(path.is_symlink() for path in model_output.rglob("*")):
        raise RuntimeError("Downloaded model snapshot contains symlinks")
    actual_paths = {
        path.relative_to(model_output).as_posix()
        for path in model_output.rglob("*")
        if path.is_file() and path.name != MODEL_SOURCE_MANIFEST and ".cache" not in path.parts
    }
    if actual_paths != expected_paths:
        raise RuntimeError("Downloaded model snapshot differs from Hub revision file list")
    files = [
        {
            "path": relative_path,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for relative_path in sorted(expected_paths)
        if (path := model_output / relative_path).is_file()
    ]
    write_json(
        model_output / MODEL_SOURCE_MANIFEST,
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "files": files,
        },
    )


def validate_model_snapshot(
    model_path: Path,
    *,
    verify_hashes: bool,
) -> dict[str, Any]:
    from chart_provenance_rl.serialization import load_json

    if model_path.is_symlink() or (model_path / MODEL_SOURCE_MANIFEST).is_symlink():
        raise ValueError("Model snapshot and manifest must not be symlinks")
    manifest = load_json(model_path / MODEL_SOURCE_MANIFEST)
    if not isinstance(manifest, dict):
        raise TypeError("Model source manifest must be an object")
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Model snapshot does not match the preregistered model revision")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Model source manifest has no files")
    manifest_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise TypeError("Model source file entries must be objects")
        relative_path = item.get("path")
        expected_size = item.get("size")
        expected_hash = item.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not isinstance(expected_size, int)
            or not isinstance(expected_hash, str)
        ):
            raise TypeError("Invalid model source file entry")
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe model snapshot path: {relative_path}")
        path = model_path / relative_path
        if path.is_symlink():
            raise ValueError(f"Model snapshot files must not be symlinks: {relative_path}")
        manifest_paths.add(relative_path)
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"Model snapshot file mismatch: {relative_path}")
        if verify_hashes and sha256_file(path) != expected_hash:
            raise ValueError(f"Model snapshot hash mismatch: {relative_path}")
    actual_paths = {
        path.relative_to(model_path).as_posix()
        for path in model_path.rglob("*")
        if path.is_file() and path.name != MODEL_SOURCE_MANIFEST and ".cache" not in path.parts
    }
    if actual_paths != manifest_paths:
        raise ValueError("Model snapshot contains missing or unmanifested files")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare immutable benchmark and model inputs.")
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    return parser


if __name__ == "__main__":
    main()
