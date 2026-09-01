import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chart_provenance_rl.evaluation.rewards import RewardArm
from chart_provenance_rl.schema import EvidenceRef
from chart_provenance_rl.serialization import sha256_file, write_json
from chart_provenance_rl.training.artifacts import (
    ADAPTER_ARTIFACT_MANIFEST,
    validate_adapter_artifact,
    write_adapter_artifact_manifest,
)
from chart_provenance_rl.training.common import (
    build_immutable_training_manifest,
    configure_transformers_mlflow,
    finalize_training_manifest,
    resolve_resume_checkpoint,
    source_tree_sha256,
    validate_resume_manifest,
)
from chart_provenance_rl.training.configuration import (
    grpo_config_kwargs,
    sft_config_kwargs,
)
from chart_provenance_rl.training.data import (
    parse_evidence_refs,
    parse_provenance_sets,
)
from chart_provenance_rl.training.inference import _adapter_provenance
from chart_provenance_rl.training.model import (
    assert_frozen_vision_and_summarize,
    configure_image_processor,
    enforce_trainable_parameter_dtype,
    validate_image_grid,
)
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
    assert first["loss_type"] == "dr_grpo"
    assert first["scale_rewards"] == "none"
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
    with pytest.raises(RuntimeError, match="already contains checkpoints"):
        resolve_resume_checkpoint(tmp_path, "none")
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(tmp_path, str(tmp_path / "missing"))
    outside = tmp_path.parent / "checkpoint-99"
    outside.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="must belong"):
        resolve_resume_checkpoint(tmp_path, str(outside))


def test_training_reward_logs_hidden_metrics(tmp_path: Path) -> None:
    rollout_log = tmp_path / "rollouts.jsonl"
    reward = TrainingReward(
        RewardArm.NAIVE_RECALL,
        17,
        rollout_log,
        run_manifest_sha256="d" * 64,
    )
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
    assert '"seed":17' in content
    assert '"reward_call_index":0' in content


def test_training_reward_gates_wrong_answer() -> None:
    reward = TrainingReward(
        RewardArm.MINIMAL_PROVENANCE,
        17,
        run_manifest_sha256="d" * 64,
    )
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


def test_training_reward_requires_run_manifest_binding() -> None:
    reward = TrainingReward(RewardArm.NAIVE_RECALL, 17)
    with pytest.raises(RuntimeError, match="bound to an immutable run manifest"):
        reward(
            prompts=[[]],
            completions=['{"answer":"42","evidence":[]}'],
            completion_ids=[[]],
            expected_answer=["42"],
            minimal_provenance_json=['[[{"series":"Revenue","category":"Q3"}]]'],
            valid_evidence_json=['[{"series":"Revenue","category":"Q3"}]'],
            record_id=["q1"],
            trainer_state=SimpleNamespace(global_step=0),
        )


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


def test_resume_manifest_rejects_config_drift(tmp_path: Path) -> None:
    immutable = {
        "stage": "grpo",
        "seed": 17,
        "reward_arm": "naive_recall",
        "training_config": {"loss_type": "dr_grpo"},
    }
    finalize_training_manifest(
        tmp_path,
        immutable=immutable,
        trainable_parameter_names=("text.q_proj.lora_A",),
        trainable_parameter_count=7,
    )
    (tmp_path / "checkpoint-1").mkdir()
    assert validate_resume_manifest(tmp_path, immutable) is not None
    drifted = {**immutable, "seed": 29}
    with pytest.raises(RuntimeError, match="immutable training configuration"):
        validate_resume_manifest(tmp_path, drifted)
    with pytest.raises(RuntimeError, match="Trainable parameter set changed"):
        finalize_training_manifest(
            tmp_path,
            immutable=immutable,
            trainable_parameter_names=("text.v_proj.lora_A",),
            trainable_parameter_count=7,
        )


def test_checkpoint_without_manifest_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1").mkdir()
    with pytest.raises(RuntimeError, match="without an immutable run manifest"):
        validate_resume_manifest(tmp_path, {"stage": "sft"})


def test_auto_resume_rejects_symlinked_checkpoint(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-checkpoint"
    outside.mkdir(exist_ok=True)
    (tmp_path / "checkpoint-1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked or escaped"):
        resolve_resume_checkpoint(tmp_path, "auto")
    with pytest.raises(ValueError, match="must not be a symlink"):
        resolve_resume_checkpoint(
            tmp_path,
            str(tmp_path / "checkpoint-1"),
        )


def test_immutable_manifest_captures_packages_and_optimizer_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chart_provenance_rl.training.common.version",
        lambda package: f"{package}-version",
    )
    manifest = build_immutable_training_manifest(
        stage="grpo",
        seed=17,
        max_steps=100,
        training_config={
            "output_dir": "/ephemeral/path",
            "run_name": "ephemeral-name",
            "optim": "paged_adamw_8bit",
            "loss_type": "dr_grpo",
            "scale_rewards": "none",
        },
        extra={"reward_arm": "naive_recall"},
    )
    assert "output_dir" not in manifest["training_config"]
    assert "run_name" not in manifest["training_config"]
    assert manifest["training_config"]["loss_type"] == "dr_grpo"
    assert manifest["training_config"]["scale_rewards"] == "none"
    assert manifest["reward_arm"] == "naive_recall"
    assert manifest["packages"]["trl"] == "trl-version"


def test_frozen_vision_is_checked_after_adapter_injection() -> None:
    vision = _FakeParameter(requires_grad=False)
    text = _FakeParameter(requires_grad=True)
    model = _FakeModel(vision=vision, text=text)
    summary = assert_frozen_vision_and_summarize(model)
    assert summary.names == ("model.language_model.q_proj.lora_A",)
    assert summary.count == 7
    vision.requires_grad = True
    with pytest.raises(RuntimeError, match="Vision tower contains trainable"):
        assert_frozen_vision_and_summarize(model)


def test_qwen_processor_pixel_cap_and_grid_validation() -> None:
    image_processor = SimpleNamespace(
        size={"shortest_edge": 3_136, "longest_edge": 16_777_216},
        patch_size=14,
    )
    processor = SimpleNamespace(image_processor=image_processor)
    configure_image_processor(processor)
    assert image_processor.size == {
        "shortest_edge": 3_136,
        "longest_edge": 512 * 384,
    }
    grid = SimpleNamespace(tolist=lambda: [[1, 28, 28]])
    assert (
        validate_image_grid(
            processor,
            {"image_grid_thw": grid},
        )
        == 28 * 14 * 28 * 14
    )
    oversized = SimpleNamespace(tolist=lambda: [[1, 40, 40]])
    with pytest.raises(RuntimeError, match="exceeds pixel cap"):
        validate_image_grid(processor, {"image_grid_thw": oversized})


def test_adapter_prediction_provenance_uses_run_manifest(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    adapter_sha256 = write_adapter_artifact_manifest(tmp_path)
    manifest = {
        "immutable": {
            "stage": "grpo",
            "seed": 17,
            "max_steps": 100,
            "reward_arm": "minimal_provenance",
            "sft_adapter_sha256": "a" * 64,
        },
        "trainable_parameters": {"count": 7, "names": ["q_proj.lora_A"]},
    }
    write_json(tmp_path / "run_manifest.json", manifest)
    provenance = _adapter_provenance(tmp_path)
    assert provenance["adapter_stage"] == "grpo"
    assert provenance["adapter_seed"] == 17
    assert provenance["adapter_reward_arm"] == "minimal_provenance"
    assert provenance["sft_adapter_sha256"] == "a" * 64
    assert provenance["adapter_artifact_sha256"] == adapter_sha256
    assert validate_adapter_artifact(tmp_path) == adapter_sha256
    (tmp_path / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_adapter_artifact(tmp_path)
    assert _adapter_provenance(None)["adapter_stage"] == "prompted_frozen"


def test_source_tree_digest_changes_with_training_code(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    source = tmp_path / "src" / "package"
    source.mkdir(parents=True)
    module = source / "module.py"
    module.write_text("VALUE = 1\n")
    first = source_tree_sha256(tmp_path)
    module.write_text("VALUE = 2\n")
    assert source_tree_sha256(tmp_path) != first


def test_adapter_artifact_and_source_digest_fail_closed(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config"):
        write_adapter_artifact_manifest(empty)
    with pytest.raises(FileNotFoundError, match="pyproject"):
        source_tree_sha256(empty)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    write_adapter_artifact_manifest(adapter)
    (adapter / "adapter_model.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="file set"):
        validate_adapter_artifact(adapter)


@pytest.mark.parametrize(
    ("manifest", "error_type", "message"),
    [
        ([], TypeError, "must be an object"),
        ({"files": []}, ValueError, "contains no files"),
        ({"files": [1]}, TypeError, "entries must be objects"),
    ],
)
def test_adapter_artifact_manifest_shape_is_strict(
    tmp_path: Path,
    manifest: object,
    error_type: type[Exception],
    message: str,
) -> None:
    write_json(tmp_path / ADAPTER_ARTIFACT_MANIFEST, manifest)
    with pytest.raises(error_type, match=message):
        validate_adapter_artifact(tmp_path)


def test_adapter_artifact_rejects_symlinked_weights(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    target = tmp_path / "weights.bin"
    target.write_bytes(b"weights")
    (tmp_path / "adapter_model.safetensors").symlink_to(target)
    with pytest.raises(ValueError, match="must not be symlinks"):
        write_adapter_artifact_manifest(tmp_path)


class _FakeData:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def to(self, dtype: str) -> "_FakeData":
        return _FakeData(dtype)


class _FakeParameter:
    def __init__(self, *, requires_grad: bool = True) -> None:
        self.requires_grad = requires_grad
        self.data = _FakeData("bfloat16")

    @property
    def dtype(self) -> str:
        return self.data.dtype

    def numel(self) -> int:
        return 7


class _FakeModel:
    def __init__(
        self,
        *,
        vision: _FakeParameter,
        text: _FakeParameter,
    ) -> None:
        self.model = SimpleNamespace(visual=SimpleNamespace(parameters=lambda: [vision]))
        self._parameters = (
            ("model.visual.qkv", vision),
            ("model.language_model.q_proj.lora_A", text),
        )

    def named_parameters(self) -> tuple[tuple[str, _FakeParameter], ...]:
        return self._parameters


def test_configure_transformers_mlflow_bounds_azure_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "azureml://example")
    monkeypatch.setenv("MLFLOW_FLATTEN_PARAMS", "FALSE")
    monkeypatch.setenv("MLFLOW_MAX_LOG_PARAMS", "1000")

    configure_transformers_mlflow()

    assert os.environ["MLFLOW_FLATTEN_PARAMS"] == "TRUE"
    assert os.environ["MLFLOW_MAX_LOG_PARAMS"] == "100"


def test_configure_transformers_mlflow_is_inert_without_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_FLATTEN_PARAMS", raising=False)
    monkeypatch.delenv("MLFLOW_MAX_LOG_PARAMS", raising=False)

    configure_transformers_mlflow()

    assert "MLFLOW_FLATTEN_PARAMS" not in os.environ
    assert "MLFLOW_MAX_LOG_PARAMS" not in os.environ
