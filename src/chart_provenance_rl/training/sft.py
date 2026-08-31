from __future__ import annotations

import argparse
from pathlib import Path

from chart_provenance_rl.constants import MODEL_ID
from chart_provenance_rl.training.common import (
    resolve_resume_checkpoint,
    write_training_manifest,
)
from chart_provenance_rl.training.configuration import sft_config_kwargs
from chart_provenance_rl.training.data import load_sft_dataset
from chart_provenance_rl.training.model import (
    enforce_trainable_parameter_dtype,
    load_model_stack,
)


def main() -> None:
    args = _parser().parse_args()
    train_sft(
        dataset_root=args.dataset_root,
        model_source=args.model_path if args.model_path else MODEL_ID,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        resume=args.resume,
    )


def train_sft(
    *,
    dataset_root: Path,
    model_source: str | Path,
    output_dir: Path,
    seed: int,
    max_steps: int,
    resume: str,
) -> None:
    import torch
    from trl import SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    stack = load_model_stack(model_source=model_source)
    dataset = load_sft_dataset(dataset_root)
    config = SFTConfig(**sft_config_kwargs(output_dir, seed=seed, max_steps=max_steps))
    write_training_manifest(
        output_dir,
        stage="sft",
        seed=seed,
        max_steps=max_steps,
        extra={
            "quantization": "4bit-nf4-double-quant",
            "compute_dtype": "float16",
            "vision_frozen": True,
            "lora_rank": 8,
            "lora_target_modules": ["q_proj", "v_proj"],
            "trainable_parameter_dtype": "float16",
            "model_source": str(model_source),
        },
    )
    trainer = SFTTrainer(
        model=stack.model,
        args=config,
        train_dataset=dataset,
        processing_class=stack.processor,
        peft_config=stack.peft_config,
    )
    enforce_trainable_parameter_dtype(trainer.model, torch.float16)
    trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(output_dir, resume))
    trainer.save_model(str(output_dir))
    stack.processor.save_pretrained(str(output_dir))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL QLoRA SFT.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--resume", default="auto")
    return parser


if __name__ == "__main__":
    main()
