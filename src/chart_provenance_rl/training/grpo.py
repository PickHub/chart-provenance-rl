from __future__ import annotations

import argparse
from pathlib import Path

from chart_provenance_rl.constants import MODEL_ID
from chart_provenance_rl.evaluation.rewards import RewardArm
from chart_provenance_rl.training.common import (
    resolve_resume_checkpoint,
    write_training_manifest,
)
from chart_provenance_rl.training.configuration import grpo_config_kwargs
from chart_provenance_rl.training.data import load_grpo_dataset
from chart_provenance_rl.training.model import (
    enforce_trainable_parameter_dtype,
    load_model_stack,
)
from chart_provenance_rl.training.reward import TrainingReward


def main() -> None:
    args = _parser().parse_args()
    train_grpo(
        dataset_root=args.dataset_root,
        model_source=args.model_path if args.model_path else MODEL_ID,
        sft_adapter=args.sft_adapter,
        output_dir=args.output_dir,
        reward_arm=RewardArm(args.reward_arm),
        seed=args.seed,
        max_steps=args.max_steps,
        resume=args.resume,
    )


def train_grpo(
    *,
    dataset_root: Path,
    model_source: str | Path,
    sft_adapter: Path,
    output_dir: Path,
    reward_arm: RewardArm,
    seed: int,
    max_steps: int,
    resume: str,
) -> None:
    import torch
    from trl import GRPOConfig, GRPOTrainer

    if not (sft_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"GRPO arms must start from the same SFT adapter: {sft_adapter}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stack = load_model_stack(
        model_source=model_source,
        adapter_path=sft_adapter,
        trainable_adapter=True,
    )
    dataset = load_grpo_dataset(dataset_root)
    config_kwargs = grpo_config_kwargs(
        output_dir,
        seed=seed,
        max_steps=max_steps,
    )
    config_kwargs["run_name"] = f"chart-provenance-grpo-{reward_arm.value}-seed-{seed}"
    config = GRPOConfig(**config_kwargs)
    write_training_manifest(
        output_dir,
        stage="grpo",
        seed=seed,
        max_steps=max_steps,
        extra={
            "reward_arm": reward_arm.value,
            "sft_adapter": str(sft_adapter),
            "quantization": "4bit-nf4-double-quant",
            "compute_dtype": "float16",
            "vision_frozen": True,
            "num_generations": 2,
            "temperature": 1.0,
            "use_vllm": False,
            "beta": 0.0,
            "trainable_parameter_dtype": "float16",
            "model_source": str(model_source),
        },
    )
    reward = TrainingReward(
        reward_arm,
        rollout_log=output_dir / "rollout_diagnostics.jsonl",
    )
    trainer = GRPOTrainer(
        model=stack.model,
        args=config,
        reward_funcs=[reward],
        train_dataset=dataset,
        processing_class=stack.processor,
    )
    enforce_trainable_parameter_dtype(trainer.model, torch.float16)
    trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(output_dir, resume))
    trainer.save_model(str(output_dir))
    stack.processor.save_pretrained(str(output_dir))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run online GRPO on Qwen3-VL.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reward-arm",
        choices=tuple(arm.value for arm in RewardArm),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--resume", default="auto")
    return parser


if __name__ == "__main__":
    main()
