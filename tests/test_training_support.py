from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chart_provenance_rl.evaluation.rewards import RewardArm
from chart_provenance_rl.training.common import resolve_resume_checkpoint
from chart_provenance_rl.training.configuration import (
    grpo_config_kwargs,
    sft_config_kwargs,
)
from chart_provenance_rl.training.data import parse_provenance_sets
from chart_provenance_rl.training.reward import TrainingReward


def test_grpo_arms_share_action_space_and_budget(tmp_path: Path) -> None:
    first = grpo_config_kwargs(tmp_path / "first", seed=17, max_steps=100)
    second = grpo_config_kwargs(tmp_path / "second", seed=17, max_steps=100)
    comparable_first = {key: value for key, value in first.items() if key != "output_dir"}
    comparable_second = {key: value for key, value in second.items() if key != "output_dir"}
    assert comparable_first == comparable_second
    assert first["num_generations"] == 2
    assert first["per_device_train_batch_size"] == 2
    assert first["max_completion_length"] == 256
    assert first["temperature"] == 1.0
    assert first["use_vllm"] is False
    assert first["fp16"] is True
    assert first["bf16"] is False


def test_sft_configuration_preserves_image_tokens(tmp_path: Path) -> None:
    config = sft_config_kwargs(tmp_path, seed=17, max_steps=50)
    assert config["max_length"] is None
    assert config["completion_only_loss"] is True
    assert config["fp16"] is True


def test_resume_checkpoint_resolution(tmp_path: Path) -> None:
    assert resolve_resume_checkpoint(tmp_path, "auto") is None
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-11").mkdir()
    assert resolve_resume_checkpoint(tmp_path, "auto") == str(tmp_path / "checkpoint-11")
    assert resolve_resume_checkpoint(tmp_path, "none") is None
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(tmp_path, str(tmp_path / "missing"))


def test_training_reward_logs_hidden_metrics(tmp_path: Path) -> None:
    rollout_log = tmp_path / "rollouts.jsonl"
    reward = TrainingReward(RewardArm.NAIVE_RECALL, rollout_log)
    output = (
        '{"answer":"42","evidence":['
        '{"series":"Revenue","category":"Q3"},'
        '{"series":"Revenue","category":"Q4"}]}'
    )
    scores = reward(
        prompts=[[]],
        completions=[output],
        completion_ids=[[]],
        expected_answer=["42"],
        minimal_provenance_json=['[[{"series":"Revenue","category":"Q3"}]]'],
        record_id=["q1"],
        trainer_state=SimpleNamespace(global_step=3),
    )
    assert scores == [1.0]
    content = rollout_log.read_text()
    assert '"hidden_reward"' in content
    assert '"evidence_precision":0.5' in content


def test_training_reward_gates_wrong_answer() -> None:
    reward = TrainingReward(RewardArm.MINIMAL_PROVENANCE)
    kwargs: dict[str, Any] = {
        "prompts": [[]],
        "completions": ['{"answer":"wrong","evidence":[{"series":"Revenue","category":"Q3"}]}'],
        "completion_ids": [[]],
        "expected_answer": ["42"],
        "minimal_provenance_json": ['[[{"series":"Revenue","category":"Q3"}]]'],
        "record_id": ["q1"],
        "trainer_state": SimpleNamespace(global_step=0),
    }
    assert reward(**kwargs) == [0.0]


def test_provenance_deserialization_is_strict() -> None:
    assert (
        parse_provenance_sets('[[{"series":"Revenue","category":"Q3"}]]')[0][0].series == "Revenue"
    )
    with pytest.raises(TypeError):
        parse_provenance_sets("{}")
