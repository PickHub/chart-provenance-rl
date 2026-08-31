from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path
from typing import Any

from PIL import Image

from chart_provenance_rl.audits import load_dataset_records
from chart_provenance_rl.constants import (
    DEFAULT_IMAGE_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    SYSTEM_PROMPT,
)
from chart_provenance_rl.evaluation.leakage import image_shuffle_map
from chart_provenance_rl.serialization import (
    load_json,
    sha256_file,
    write_jsonl,
)
from chart_provenance_rl.training.artifacts import (
    validate_adapter_artifact,
)
from chart_provenance_rl.training.common import RUN_MANIFEST
from chart_provenance_rl.training.model import (
    load_model_stack,
    validate_image_grid,
)


class ImageMode(StrEnum):
    NORMAL = "normal"
    BLANK = "blank"
    SHUFFLED = "shuffled"


def main() -> None:
    args = _parser().parse_args()
    predict(
        dataset_root=args.dataset_root,
        model_source=args.model_path if args.model_path else MODEL_ID,
        split=args.split,
        output=args.output,
        adapter_path=args.adapter_path,
        image_mode=ImageMode(args.image_mode),
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )


def predict(
    *,
    dataset_root: Path,
    model_source: str | Path,
    split: str,
    output: Path,
    adapter_path: Path | None,
    image_mode: ImageMode,
    max_new_tokens: int,
    limit: int | None,
) -> None:
    import torch

    records_by_split = load_dataset_records(dataset_root)
    try:
        records = records_by_split[split]
    except KeyError as error:
        raise KeyError(f"Unknown dataset split: {split}") from error
    shuffled_paths = image_shuffle_map(records) if image_mode is ImageMode.SHUFFLED else {}
    if limit is not None:
        records = records[:limit]
    dataset_manifest_sha256 = sha256_file(dataset_root / "dataset.json")
    adapter_provenance = _adapter_provenance(adapter_path)
    stack = load_model_stack(
        model_source=model_source,
        adapter_path=adapter_path,
        trainable_adapter=False,
    )
    stack.model.eval()
    predictions: list[dict[str, Any]] = []
    for record in records:
        rgb_image, image_source = _inference_image(
            dataset_root,
            record.record_id,
            record.image_path,
            image_mode,
            shuffled_paths,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": record.question},
                ],
            },
        ]
        inputs = stack.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        image_grid_pixels = validate_image_grid(stack.processor, inputs)
        inputs = inputs.to(stack.model.device)
        with torch.inference_mode():
            generated = stack.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = generated[:, prompt_length:]
        prediction = stack.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        predictions.append(
            {
                "record_id": record.record_id,
                "prediction": prediction,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_source": str(model_source),
                "adapter_path": str(adapter_path) if adapter_path else None,
                **adapter_provenance,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "decoding": "greedy",
                "image_mode": image_mode.value,
                "image_source": image_source,
                "processed_image_grid_pixels": image_grid_pixels,
            }
        )
    write_jsonl(output, predictions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic Qwen3-VL benchmark inference.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument(
        "--image-mode",
        choices=tuple(mode.value for mode in ImageMode),
        default=ImageMode.NORMAL.value,
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    return parser


def _inference_image(
    dataset_root: Path,
    record_id: str,
    image_path: str,
    image_mode: ImageMode,
    shuffled_paths: dict[str, str],
) -> tuple[Image.Image, str]:
    if image_mode is ImageMode.BLANK:
        return Image.new("RGB", DEFAULT_IMAGE_SIZE, "white"), "neutral_blank"
    source_path = shuffled_paths[record_id] if image_mode is ImageMode.SHUFFLED else image_path
    with Image.open(dataset_root / source_path) as image:
        return image.convert("RGB"), source_path


def _adapter_provenance(adapter_path: Path | None) -> dict[str, Any]:
    if adapter_path is None:
        return {
            "adapter_artifact_sha256": None,
            "run_manifest_sha256": None,
            "run_max_steps": None,
            "adapter_stage": "prompted_frozen",
            "adapter_seed": None,
            "adapter_reward_arm": None,
            "sft_adapter_sha256": None,
        }
    manifest_path = adapter_path / RUN_MANIFEST
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError("Adapter run manifest must be an object")
    immutable = manifest.get("immutable")
    if not isinstance(immutable, dict):
        raise TypeError("Adapter run manifest is missing immutable metadata")
    stage = immutable.get("stage")
    seed = immutable.get("seed")
    reward_arm = immutable.get("reward_arm")
    sft_adapter_sha256 = immutable.get("sft_adapter_sha256")
    max_steps = immutable.get("max_steps")
    if stage not in {"sft", "grpo"}:
        raise ValueError(f"Unsupported adapter stage: {stage!r}")
    if not isinstance(seed, int):
        raise TypeError("Adapter seed must be an integer")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise TypeError("Adapter max_steps must be a positive integer")
    if stage == "grpo" and not isinstance(reward_arm, str):
        raise TypeError("GRPO adapter manifest requires a reward arm")
    if stage == "sft" and reward_arm is not None:
        raise ValueError("SFT adapter manifest must not contain a reward arm")
    return {
        "adapter_artifact_sha256": validate_adapter_artifact(adapter_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "run_max_steps": max_steps,
        "adapter_stage": stage,
        "adapter_seed": seed,
        "adapter_reward_arm": reward_arm,
        "sft_adapter_sha256": sft_adapter_sha256,
    }


if __name__ == "__main__":
    main()
