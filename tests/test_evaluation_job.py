from pathlib import Path

from chart_provenance_rl.audits import load_dataset_records
from chart_provenance_rl.evaluation.attacks import attack_predictions
from chart_provenance_rl.evaluation.job import score_prediction_file
from chart_provenance_rl.serialization import load_jsonl, write_jsonl


def test_scoring_job_writes_question_level_rows(dataset_root: Path, tmp_path: Path) -> None:
    records = load_dataset_records(dataset_root)["test_composition"]
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        (
            {
                "record_id": record.record_id,
                "prediction": attack_predictions(record)[0]["prediction"],
            }
            for record in records
        ),
    )
    output = tmp_path / "raw"
    metrics = score_prediction_file(
        dataset_root=dataset_root,
        split="test_composition",
        predictions_path=predictions,
        output_dir=output,
        method="oracle_test",
        seed=17,
        repair=False,
    )
    assert metrics["joint_exact_pass"] == 1.0
    rows = load_jsonl(output / "question_metrics.jsonl")
    assert len(rows) == len(records)
    assert all(row["seed"] == 17 for row in rows)
    assert all(row["method"] == "oracle_test" for row in rows)


def test_scoring_job_can_apply_deterministic_repair(
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    records = load_dataset_records(dataset_root)["test_composition"]
    predictions = tmp_path / "fenced.jsonl"
    write_jsonl(
        predictions,
        (
            {
                "record_id": record.record_id,
                "prediction": (
                    "```json\n" + str(attack_predictions(record)[0]["prediction"]) + "\n```"
                ),
            }
            for record in records
        ),
    )
    metrics = score_prediction_file(
        dataset_root=dataset_root,
        split="test_composition",
        predictions_path=predictions,
        output_dir=tmp_path / "repaired",
        method="repaired",
        seed=29,
        repair=True,
    )
    assert metrics["joint_exact_pass"] == 1.0
    assert metrics["postprocessing"] == "deterministic_json_repair"
