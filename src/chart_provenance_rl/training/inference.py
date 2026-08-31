from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from chart_provenance_rl.audits import load_dataset_records
from chart_provenance_rl.constants import (
    MODEL_ID,
    MODEL_REVISION,
    SYSTEM_PROMPT,
)
from chart_provenance_rl.serialization import write_jsonl
from chart_provenance_rl.training.model import load_model_stack


def main() -> None:
    args = _parser().parse_args()
    predict(
        dataset_root=args.dataset_root,
        split=args.split,
        output=args.output,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )


def predict(
    *,
    dataset_root: Path,
    split: str,
    output: Path,
    adapter_path: Path | None,
    max_new_tokens: int,
    limit: int | None,
) -> None:
    import torch

    records_by_split = load_dataset_records(dataset_root)
    try:
        records = records_by_split[split]
    except KeyError as error:
        raise KeyError(f"Unknown dataset split: {split}") from error
    if limit is not None:
        records = records[:limit]
    stack = load_model_stack(
        adapter_path=adapter_path,
        trainable_adapter=False,
    )
    stack.model.eval()
    predictions: list[dict[str, Any]] = []
    for record in records:
        with Image.open(dataset_root / record.image_path) as image:
            rgb_image = image.convert("RGB")
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
                "adapter_path": str(adapter_path) if adapter_path else None,
                "decoding": "greedy",
            }
        )
    write_jsonl(output, predictions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic Qwen3-VL benchmark inference.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    return parser


if __name__ == "__main__":
    main()
