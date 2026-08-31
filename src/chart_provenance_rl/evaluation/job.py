from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chart_provenance_rl.audits import load_dataset_records
from chart_provenance_rl.evaluation.metrics import (
    aggregate_predictions,
    evaluate_prediction,
)
from chart_provenance_rl.evaluation.provenance import (
    PROVENANCE_FIELDS,
    expected_prediction_provenance,
    is_sha256,
)
from chart_provenance_rl.evaluation.repair import repair_output
from chart_provenance_rl.serialization import (
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def main() -> None:
    args = _parser().parse_args()
    score_prediction_file(
        dataset_root=args.dataset_root,
        split=args.split,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        method=args.method,
        seed=args.seed,
        repair=args.repair,
    )


def score_prediction_file(
    *,
    dataset_root: Path,
    split: str,
    predictions_path: Path,
    output_dir: Path,
    method: str,
    seed: int,
    repair: bool,
) -> dict[str, Any]:
    if repair != (method == "sft_json_repair"):
        raise ValueError("Deterministic repair is only valid for method sft_json_repair")
    records_by_split = load_dataset_records(dataset_root)
    try:
        records = records_by_split[split]
    except KeyError as error:
        raise KeyError(f"Unknown dataset split: {split}") from error
    predictions = load_jsonl(predictions_path)
    if repair:
        predictions = _repair_rows(predictions)
    dataset_manifest_sha256 = sha256_file(dataset_root / "dataset.json")
    provenance = _validate_prediction_provenance(
        predictions,
        method=method,
        seed=seed,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    aggregate: dict[str, Any] = dict(aggregate_predictions(records, predictions))
    aggregate["method"] = method
    aggregate["seed"] = seed
    aggregate["split"] = split
    aggregate["postprocessing"] = "deterministic_json_repair" if repair else "none"
    aggregate.update(provenance)
    write_json(output_dir / "metrics.json", aggregate)
    record_by_id = {record.record_id: record for record in records}
    question_rows = []
    for prediction in predictions:
        record_id = prediction["record_id"]
        raw_output = prediction["prediction"]
        if not isinstance(record_id, str) or not isinstance(raw_output, str):
            raise TypeError("Prediction rows require string record_id and prediction")
        metric = evaluate_prediction(
            record_by_id[record_id],
            raw_output,
            train_reward=(
                float(prediction["train_reward"])
                if prediction.get("train_reward") is not None
                else None
            ),
        )
        metric_values = {
            key: float(value) if isinstance(value, bool) else value
            for key, value in asdict(metric).items()
        }
        question_rows.append(
            {
                **metric_values,
                **{field: prediction.get(field) for field in PROVENANCE_FIELDS},
                "method": method,
                "seed": seed,
                "split": split,
                "postprocessing": aggregate["postprocessing"],
            }
        )
    write_jsonl(output_dir / "question_metrics.jsonl", question_rows)
    return aggregate


def _repair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repaired_rows: list[dict[str, Any]] = []
    for row in rows:
        raw_output = row.get("prediction")
        if not isinstance(raw_output, str):
            raise TypeError("prediction must be a string")
        repaired = repair_output(raw_output)
        updated = dict(row)
        if repaired is not None:
            updated["prediction"] = repaired
        updated["postprocessing"] = "deterministic_json_repair"
        repaired_rows.append(updated)
    return repaired_rows


def _validate_prediction_provenance(
    predictions: list[dict[str, Any]],
    *,
    method: str,
    seed: int,
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    expected = expected_prediction_provenance(method)
    if not predictions:
        raise ValueError("Prediction artifact is empty")
    observed_values: dict[str, set[Any]] = {
        field: {row.get(field) for row in predictions} for field in PROVENANCE_FIELDS
    }
    if any(len(values) != 1 for values in observed_values.values()):
        raise ValueError("Prediction provenance must be constant within an artifact")
    observed = {field: values.pop() for field, values in observed_values.items()}
    if observed["dataset_manifest_sha256"] != dataset_manifest_sha256:
        raise ValueError("Prediction artifact does not match the scored dataset")
    if observed["image_mode"] != expected["image_mode"]:
        raise ValueError(f"{method} requires image mode {expected['image_mode']}")
    if observed["adapter_stage"] != expected["adapter_stage"]:
        raise ValueError(f"{method} requires adapter stage {expected['adapter_stage']}")
    if observed["adapter_reward_arm"] != expected["adapter_reward_arm"]:
        raise ValueError(f"{method} requires reward arm {expected['adapter_reward_arm']}")
    expected_seed = None if method == "prompted_frozen" else seed
    if observed["adapter_seed"] != expected_seed:
        raise ValueError(f"{method} adapter seed does not match score seed")
    adapter_hash = observed["adapter_artifact_sha256"]
    if method == "prompted_frozen":
        if adapter_hash is not None:
            raise ValueError("Prompted frozen predictions must not use an adapter")
    elif not is_sha256(adapter_hash):
        raise ValueError(f"{method} requires a valid adapter artifact hash")
    if expected["adapter_stage"] == "grpo" and not is_sha256(observed["sft_adapter_sha256"]):
        raise ValueError("GRPO predictions require the source SFT adapter hash")
    run_manifest_hash = observed["run_manifest_sha256"]
    run_max_steps = observed["run_max_steps"]
    if method == "prompted_frozen":
        if run_manifest_hash is not None:
            raise ValueError("Prompted frozen predictions must not use a run manifest")
        if run_max_steps is not None:
            raise ValueError("Prompted frozen predictions must not report train steps")
    elif not is_sha256(run_manifest_hash):
        raise ValueError(f"{method} requires a valid run manifest hash")
    elif not isinstance(run_max_steps, int) or run_max_steps <= 0:
        raise ValueError(f"{method} requires positive run max_steps")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score one prediction file.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repair", type=_parse_bool, default=False)
    return parser


def _parse_bool(value: str) -> bool:
    normalized = value.casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


if __name__ == "__main__":
    main()
