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
from chart_provenance_rl.evaluation.repair import repair_output
from chart_provenance_rl.serialization import load_jsonl, write_json, write_jsonl


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
    records_by_split = load_dataset_records(dataset_root)
    try:
        records = records_by_split[split]
    except KeyError as error:
        raise KeyError(f"Unknown dataset split: {split}") from error
    predictions = load_jsonl(predictions_path)
    if repair:
        predictions = _repair_rows(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    aggregate: dict[str, Any] = dict(aggregate_predictions(records, predictions))
    aggregate["method"] = method
    aggregate["seed"] = seed
    aggregate["split"] = split
    aggregate["postprocessing"] = "deterministic_json_repair" if repair else "none"
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
        question_rows.append(
            {
                **asdict(metric),
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
