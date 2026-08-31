from __future__ import annotations

import argparse
from pathlib import Path

from chart_provenance_rl.constants import MODEL_ID
from chart_provenance_rl.evaluation.rewards import RewardArm
from chart_provenance_rl.evaluation.rollouts import (
    bind_completed_rollout_artifact,
    reconcile_rollout_log,
)
from chart_provenance_rl.serialization import sha256_file
from chart_provenance_rl.training.artifacts import (
    validate_adapter_artifact,
    write_adapter_artifact_manifest,
)
from chart_provenance_rl.training.common import (
    RUN_MANIFEST,
    build_immutable_training_manifest,
    finalize_training_manifest,
    resolve_resume_checkpoint,
    validate_resume_manifest,
)
from chart_provenance_rl.training.configuration import grpo_config_kwargs
from chart_provenance_rl.training.data import load_grpo_dataset
from chart_provenance_rl.training.model import (
    assert_frozen_vision_and_summarize,
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
    sft_manifest = sft_adapter / RUN_MANIFEST
    if not sft_manifest.is_file():
        raise FileNotFoundError(
            f"SFT adapter is missing its immutable run manifest: {sft_manifest}"
        )
    sft_adapter_sha256 = validate_adapter_artifact(sft_adapter)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_kwargs = grpo_config_kwargs(
        output_dir,
        seed=seed,
        max_steps=max_steps,
    )
    config_kwargs["run_name"] = f"chart-provenance-grpo-{reward_arm.value}-seed-{seed}"
    immutable = build_immutable_training_manifest(
        stage="grpo",
        seed=seed,
        max_steps=max_steps,
        training_config=config_kwargs,
        extra={
            "reward_arm": reward_arm.value,
            "sft_run_manifest_sha256": sha256_file(sft_manifest),
            "sft_adapter_sha256": sft_adapter_sha256,
            "dataset_manifest_sha256": sha256_file(dataset_root / "dataset.json"),
            "quantization": "4bit-nf4-double-quant",
            "compute_dtype": "float16",
            "vision_frozen": True,
            "num_generations": 2,
            "temperature": 1.0,
            "use_vllm": False,
            "beta": 0.0,
        },
    )
    resume_checkpoint = resolve_resume_checkpoint(output_dir, resume)
    validate_resume_manifest(output_dir, immutable)
    rollout_log = output_dir / "rollout_diagnostics.jsonl"
    reconcile_rollout_log(rollout_log, resume_checkpoint)
    stack = load_model_stack(
        model_source=model_source,
        adapter_path=sft_adapter,
        trainable_adapter=True,
    )
    dataset = load_grpo_dataset(dataset_root)
    config = GRPOConfig(**config_kwargs)
    reward = TrainingReward(
        reward_arm,
        seed=seed,
        rollout_log=rollout_log,
    )
    trainer = GRPOTrainer(
        model=stack.model,
        args=config,
        reward_funcs=[reward],
        train_dataset=dataset,
        processing_class=stack.processor,
    )
    enforce_trainable_parameter_dtype(trainer.model, torch.float16)
    trainable = assert_frozen_vision_and_summarize(trainer.model)
    finalize_training_manifest(
        output_dir,
        immutable=immutable,
        trainable_parameter_names=trainable.names,
        trainable_parameter_count=trainable.count,
    )
    run_manifest_sha256 = sha256_file(output_dir / RUN_MANIFEST)
    reward.run_manifest_sha256 = run_manifest_sha256
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output_dir))
    stack.processor.save_pretrained(str(output_dir))
    adapter_artifact_sha256 = write_adapter_artifact_manifest(output_dir)
    bind_completed_rollout_artifact(
        rollout_log,
        run_manifest_sha256=run_manifest_sha256,
        adapter_artifact_sha256=adapter_artifact_sha256,
    )


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
