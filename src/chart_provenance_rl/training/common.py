from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from chart_provenance_rl.constants import MODEL_ID, MODEL_REVISION
from chart_provenance_rl.serialization import write_json

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


def resolve_resume_checkpoint(output_dir: Path, value: str) -> str | None:
    if value == "none":
        return None
    if value != "auto":
        checkpoint = Path(value)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return str(checkpoint)
    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    if not checkpoints:
        return None
    return str(
        max(
            checkpoints,
            key=lambda path: int(path.name.removeprefix("checkpoint-")),
        )
    )


def write_training_manifest(
    output_dir: Path,
    *,
    stage: str,
    seed: int,
    max_steps: int,
    extra: dict[str, Any] | None = None,
) -> None:
    packages: dict[str, str] = {}
    for package in GPU_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Missing required GPU package: {package}") from error
    manifest: dict[str, Any] = {
        "stage": stage,
        "seed": seed,
        "max_steps": max_steps,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "packages": packages,
    }
    if extra:
        manifest.update(extra)
    write_json(output_dir / "run_manifest.json", manifest)
