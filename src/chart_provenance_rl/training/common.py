from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from chart_provenance_rl.constants import MODEL_ID, MODEL_REVISION
from chart_provenance_rl.serialization import (
    load_json,
    sha256_file,
    sha256_json,
    write_json,
)

GPU_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface-hub",
    "mlflow",
    "peft",
    "torch",
    "torchvision",
    "transformers",
    "trl",
)
RUN_MANIFEST = "run_manifest.json"


def configure_transformers_mlflow() -> None:
    if "MLFLOW_TRACKING_URI" not in os.environ:
        return
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "TRUE"
    os.environ["MLFLOW_MAX_LOG_PARAMS"] = "100"


def resolve_resume_checkpoint(output_dir: Path, value: str) -> str | None:
    if value == "none":
        if _checkpoints(output_dir):
            raise RuntimeError(
                "Output directory already contains checkpoints; use auto or a new path"
            )
        return None
    if value != "auto":
        checkpoint = Path(value)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if checkpoint.is_symlink():
            raise ValueError("Explicit resume checkpoint must not be a symlink")
        if checkpoint.resolve().parent != output_dir.resolve():
            raise ValueError("Explicit resume checkpoint must belong to the output directory")
        return str(checkpoint)
    checkpoints = _checkpoints(output_dir)
    if not checkpoints:
        return None
    return str(
        max(
            checkpoints,
            key=lambda path: int(path.name.removeprefix("checkpoint-")),
        )
    )


def build_immutable_training_manifest(
    *,
    stage: str,
    seed: int,
    max_steps: int,
    training_config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in GPU_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Missing required GPU package: {package}") from error
    immutable_config = {
        key: value
        for key, value in training_config.items()
        if key not in {"output_dir", "run_name"}
    }
    manifest: dict[str, Any] = {
        "stage": stage,
        "seed": seed,
        "max_steps": max_steps,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_sha256": source_tree_sha256(Path.cwd()),
        "packages": packages,
        "training_config": immutable_config,
    }
    if extra:
        manifest.update(extra)
    return manifest


def source_tree_sha256(root: Path) -> str:
    required_files = (root / "pyproject.toml", root / "uv.lock")
    if not all(path.is_file() for path in required_files):
        raise FileNotFoundError(
            "Training must run from a source tree containing pyproject.toml and uv.lock"
        )
    source_files = [
        *required_files,
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "src").rglob("py.typed")),
    ]
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in source_files
    ]
    return sha256_json(entries)


def validate_resume_manifest(
    output_dir: Path,
    immutable: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = output_dir / RUN_MANIFEST
    checkpoints = _checkpoints(output_dir)
    if not manifest_path.is_file():
        if checkpoints:
            raise RuntimeError("Checkpoints exist without an immutable run manifest")
        return None
    existing = load_json(manifest_path)
    if not isinstance(existing, dict):
        raise TypeError("Run manifest must be a JSON object")
    if existing.get("immutable") != immutable:
        raise RuntimeError("Run manifest does not match immutable training configuration")
    return existing


def finalize_training_manifest(
    output_dir: Path,
    *,
    immutable: dict[str, Any],
    trainable_parameter_names: tuple[str, ...],
    trainable_parameter_count: int,
) -> None:
    expected = {
        "immutable": immutable,
        "trainable_parameters": {
            "count": trainable_parameter_count,
            "names": list(trainable_parameter_names),
        },
    }
    existing = validate_resume_manifest(output_dir, immutable)
    if existing is not None:
        if existing != expected:
            raise RuntimeError("Trainable parameter set changed across resume")
        return
    write_json(output_dir / RUN_MANIFEST, expected)


def _checkpoints(output_dir: Path) -> list[Path]:
    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    unsafe = [
        path
        for path in checkpoints
        if path.is_symlink() or path.resolve().parent != output_dir.resolve()
    ]
    if unsafe:
        raise RuntimeError(
            "Checkpoint discovery found symlinked or escaped paths: "
            + ", ".join(str(path) for path in unsafe)
        )
    return checkpoints
