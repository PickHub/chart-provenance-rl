from __future__ import annotations

from pathlib import Path


def sft_config_kwargs(
    output_dir: Path,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, object]:
    return {
        **_common(output_dir, seed=seed, max_steps=max_steps),
        "run_name": f"chart-provenance-sft-seed-{seed}",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "max_length": None,
        "completion_only_loss": True,
        "shuffle_dataset": True,
    }


def grpo_config_kwargs(
    output_dir: Path,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, object]:
    return {
        **_common(output_dir, seed=seed, max_steps=max_steps),
        "run_name": f"chart-provenance-grpo-seed-{seed}",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "num_generations": 2,
        "max_completion_length": 256,
        "temperature": 1.0,
        "loss_type": "grpo",
        "beta": 0.04,
        "use_vllm": False,
        "log_completions": False,
    }


def _common(
    output_dir: Path,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, object]:
    return {
        "output_dir": str(output_dir),
        "seed": seed,
        "data_seed": seed,
        "max_steps": max_steps,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 1,
        "save_steps": 25,
        "save_total_limit": 2,
        "report_to": ["mlflow"],
        "remove_unused_columns": False,
    }
