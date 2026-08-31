from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from chart_provenance_rl.schema import BenchmarkRecord
from chart_provenance_rl.training.configuration import grpo_config_kwargs
from chart_provenance_rl.training.data import (
    build_sft_example,
    load_sft_dataset,
)


def test_sft_rows_use_plural_image_lists(
    dataset_root: Path,
    records: list[BenchmarkRecord],
) -> None:
    example = build_sft_example(dataset_root, records[0])
    assert "image" not in example
    assert isinstance(example["images"], list)
    assert len(example["images"]) == 1
    assert example["images"][0].mode == "RGB"


def test_pinned_trl_vlm_collator_and_reward_kwargs_contract(
    dataset_root: Path,
) -> None:
    pytest.importorskip("trl")
    from trl import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer
    from trl.trainer.sft_trainer import (
        DataCollatorForVisionLanguageModeling,
    )

    collator_source = inspect.getsource(
        DataCollatorForVisionLanguageModeling._collate_prompt_completion
    )
    assert 'example["images"]' in collator_source
    reward_source = inspect.getsource(GRPOTrainer._calculate_rewards)
    assert "reward_kwargs" in reward_source
    assert "for key in inputs[0]" in reward_source
    assert "prompts=prompts" in reward_source
    assert "completions=completions" in reward_source
    assert "completion_ids=completion_ids_list" in reward_source
    config_kwargs = grpo_config_kwargs(
        Path("/tmp/chart-provenance-contract"),
        seed=17,
        max_steps=100,
    )
    config_kwargs["use_cpu"] = True
    config_kwargs["fp16"] = False
    config = GRPOConfig(**config_kwargs)
    assert config.remove_unused_columns is False
    assert config.loss_type == "dr_grpo"
    assert config.scale_rewards == "none"
    dataset = load_sft_dataset(dataset_root)
    assert "images" in dataset.column_names
    assert "image" not in dataset.column_names
    assert len(dataset[0]["images"]) == 1
