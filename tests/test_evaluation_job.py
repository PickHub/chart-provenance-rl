from pathlib import Path

import pytest

from chart_provenance_rl.audits import load_dataset_records
from chart_provenance_rl.evaluation.attacks import attack_predictions
from chart_provenance_rl.evaluation.job import (
    _validate_prediction_provenance,
    score_prediction_file,
)
from chart_provenance_rl.serialization import (
    load_jsonl,
    sha256_file,
    write_jsonl,
)


def test_scoring_job_writes_question_level_rows(dataset_root: Path, tmp_path: Path) -> None:
    records = load_dataset_records(dataset_root)["test_composition"]
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        (
            _sft_prediction(
                record.record_id,
                attack_predictions(record)[0]["prediction"],
                seed=17,
                dataset_hash=sha256_file(dataset_root / "dataset.json"),
            )
            for record in records
        ),
    )
    output = tmp_path / "raw"
    metrics = score_prediction_file(
        dataset_root=dataset_root,
        split="test_composition",
        predictions_path=predictions,
        output_dir=output,
        method="sft",
        seed=17,
        repair=False,
    )
    assert metrics["joint_exact_pass"] == 1.0
    rows = load_jsonl(output / "question_metrics.jsonl")
    assert len(rows) == len(records)
    assert all(row["seed"] == 17 for row in rows)
    assert all(row["method"] == "sft" for row in rows)
    assert all(isinstance(row["answer_em"], float) for row in rows)
    assert all(isinstance(row["joint_exact"], float) for row in rows)


def test_scoring_job_can_apply_deterministic_repair(
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    records = load_dataset_records(dataset_root)["test_composition"]
    predictions = tmp_path / "fenced.jsonl"
    write_jsonl(
        predictions,
        (
            _sft_prediction(
                record.record_id,
                "```json\n" + str(attack_predictions(record)[0]["prediction"]) + "\n```",
                seed=29,
                dataset_hash=sha256_file(dataset_root / "dataset.json"),
            )
            for record in records
        ),
    )
    metrics = score_prediction_file(
        dataset_root=dataset_root,
        split="test_composition",
        predictions_path=predictions,
        output_dir=tmp_path / "repaired",
        method="sft_json_repair",
        seed=29,
        repair=True,
    )
    assert metrics["joint_exact_pass"] == 1.0
    assert metrics["postprocessing"] == "deterministic_json_repair"


def test_scoring_job_rejects_relabelled_probe_metadata(
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    record = load_dataset_records(dataset_root)["test_composition"][0]
    row = _sft_prediction(
        record.record_id,
        attack_predictions(record)[0]["prediction"],
        seed=17,
        dataset_hash=sha256_file(dataset_root / "dataset.json"),
    )
    row["image_mode"] = "blank"
    predictions = tmp_path / "relabelled.jsonl"
    write_jsonl(predictions, [row])
    with pytest.raises(ValueError, match="requires image mode normal"):
        score_prediction_file(
            dataset_root=dataset_root,
            split="test_composition",
            predictions_path=predictions,
            output_dir=tmp_path / "invalid",
            method="sft",
            seed=17,
            repair=False,
        )
    with pytest.raises(
        ValueError,
        match="only valid for method sft_json_repair",
    ):
        score_prediction_file(
            dataset_root=dataset_root,
            split="test_composition",
            predictions_path=predictions,
            output_dir=tmp_path / "invalid-repair",
            method="sft",
            seed=17,
            repair=True,
        )


def test_grpo_prediction_provenance_requires_arm_seed_and_sft_source() -> None:
    row = {
        "adapter_artifact_sha256": "a" * 64,
        "run_manifest_sha256": "d" * 64,
        "run_max_steps": 100,
        "adapter_stage": "grpo",
        "adapter_seed": 17,
        "adapter_reward_arm": "minimal_provenance",
        "sft_adapter_sha256": "b" * 64,
        "dataset_manifest_sha256": "c" * 64,
        "image_mode": "blank",
    }
    provenance = _validate_prediction_provenance(
        [row],
        method="grpo_minimal_question_only",
        seed=17,
        dataset_manifest_sha256="c" * 64,
    )
    assert provenance["adapter_reward_arm"] == "minimal_provenance"
    wrong_seed = dict(row, adapter_seed=29)
    with pytest.raises(ValueError, match="adapter seed"):
        _validate_prediction_provenance(
            [wrong_seed],
            method="grpo_minimal_question_only",
            seed=17,
            dataset_manifest_sha256="c" * 64,
        )
    missing_sft = dict(row, sft_adapter_sha256=None)
    with pytest.raises(ValueError, match="source SFT"):
        _validate_prediction_provenance(
            [missing_sft],
            method="grpo_minimal_question_only",
            seed=17,
            dataset_manifest_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="Unsupported evaluation method"):
        _validate_prediction_provenance(
            [row],
            method="invented",
            seed=17,
            dataset_manifest_sha256="c" * 64,
        )


def test_prompted_prediction_provenance_rejects_adapters_and_empty_rows() -> None:
    row = {
        "adapter_artifact_sha256": None,
        "run_manifest_sha256": None,
        "run_max_steps": None,
        "adapter_stage": "prompted_frozen",
        "adapter_seed": None,
        "adapter_reward_arm": None,
        "sft_adapter_sha256": None,
        "dataset_manifest_sha256": "c" * 64,
        "image_mode": "normal",
    }
    provenance = _validate_prediction_provenance(
        [row],
        method="prompted_frozen",
        seed=17,
        dataset_manifest_sha256="c" * 64,
    )
    assert provenance["adapter_artifact_sha256"] is None
    with pytest.raises(ValueError, match="must not use an adapter"):
        _validate_prediction_provenance(
            [dict(row, adapter_artifact_sha256="a" * 64)],
            method="prompted_frozen",
            seed=17,
            dataset_manifest_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="empty"):
        _validate_prediction_provenance(
            [],
            method="prompted_frozen",
            seed=17,
            dataset_manifest_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="constant"):
        _validate_prediction_provenance(
            [row, dict(row, image_mode="blank")],
            method="prompted_frozen",
            seed=17,
            dataset_manifest_sha256="c" * 64,
        )


def _sft_prediction(
    record_id: str,
    prediction: object,
    *,
    seed: int,
    dataset_hash: str,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "prediction": prediction,
        "adapter_artifact_sha256": f"{seed:064x}",
        "run_manifest_sha256": f"{seed + 100:064x}",
        "run_max_steps": 100,
        "adapter_stage": "sft",
        "adapter_seed": seed,
        "adapter_reward_arm": None,
        "sft_adapter_sha256": None,
        "dataset_manifest_sha256": dataset_hash,
        "image_mode": "normal",
    }
