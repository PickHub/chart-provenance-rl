from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chart_provenance_rl.evaluation.rewards import RewardArm
from chart_provenance_rl.schema import EvidenceRef
from chart_provenance_rl.serialization import sha256_file, write_json
from chart_provenance_rl.training.common import resolve_resume_checkpoint
from chart_provenance_rl.training.configuration import (
    grpo_config_kwargs,
    sft_config_kwargs,
)
from chart_provenance_rl.training.data import (
    parse_evidence_refs,
    parse_provenance_sets,
)
from chart_provenance_rl.training.model import enforce_trainable_parameter_dtype
from chart_provenance_rl.training.prepare import (
    MODEL_SOURCE_MANIFEST,
    validate_model_snapshot,
)
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
    assert first["beta"] == 0.0
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
        valid_evidence_json=[
            '[{"series":"Revenue","category":"Q3"},{"series":"Revenue","category":"Q4"}]'
        ],
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
        "valid_evidence_json": ['[{"series":"Revenue","category":"Q3"}]'],
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
    assert parse_evidence_refs('[{"series":"Revenue","category":"Q3"}]') == (
        EvidenceRef("Revenue", "Q3"),
    )


def test_model_snapshot_manifest_verification(tmp_path: Path) -> None:
    model_file = tmp_path / "config.json"
    model_file.write_text("{}")
    write_json(
        tmp_path / MODEL_SOURCE_MANIFEST,
        {
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "license": "Apache-2.0",
            "files": [
                {
                    "path": "config.json",
                    "size": model_file.stat().st_size,
                    "sha256": sha256_file(model_file),
                }
            ],
        },
    )
    manifest = validate_model_snapshot(tmp_path, verify_hashes=True)
    assert manifest["files"][0]["path"] == "config.json"
    model_file.write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_model_snapshot(tmp_path, verify_hashes=True)


def test_model_snapshot_rejects_unmanifested_files(tmp_path: Path) -> None:
    model_file = tmp_path / "config.json"
    model_file.write_text("{}")
    write_json(
        tmp_path / MODEL_SOURCE_MANIFEST,
        {
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "license": "Apache-2.0",
            "files": [
                {
                    "path": "config.json",
                    "size": model_file.stat().st_size,
                    "sha256": sha256_file(model_file),
                }
            ],
        },
    )
    (tmp_path / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unmanifested"):
        validate_model_snapshot(tmp_path, verify_hashes=True)


def test_model_snapshot_rejects_parent_traversal(tmp_path: Path) -> None:
    write_json(
        tmp_path / MODEL_SOURCE_MANIFEST,
        {
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "license": "Apache-2.0",
            "files": [
                {
                    "path": "../outside",
                    "size": 0,
                    "sha256": "0" * 64,
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="Unsafe model snapshot path"):
        validate_model_snapshot(tmp_path, verify_hashes=False)


def test_trainable_adapter_dtype_is_enforced() -> None:
    parameter = _FakeParameter()
    count = enforce_trainable_parameter_dtype(
        SimpleNamespace(parameters=lambda: [parameter]),
        "float16",
    )
    assert count == 7
    assert parameter.dtype == "float16"


class _FakeData:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def to(self, dtype: str) -> "_FakeData":
        return _FakeData(dtype)


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True
        self.data = _FakeData("bfloat16")

    @property
    def dtype(self) -> str:
        return self.data.dtype

    def numel(self) -> int:
        return 7
