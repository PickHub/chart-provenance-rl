from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from chart_provenance_rl.analysis import compare_methods
from chart_provenance_rl.audits import audit_dataset, load_dataset_records
from chart_provenance_rl.constants import MODEL_ID
from chart_provenance_rl.evaluation.attacks import attack_predictions
from chart_provenance_rl.evaluation.metrics import aggregate_predictions
from chart_provenance_rl.evaluation.repair import repair_output
from chart_provenance_rl.generation.config import default_config, smoke_config
from chart_provenance_rl.generation.generator import generate_dataset
from chart_provenance_rl.schema import BenchmarkRecord
from chart_provenance_rl.serialization import (
    canonical_json,
    load_jsonl,
    write_json,
    write_jsonl,
)
from chart_provenance_rl.smoke import run_cpu_smoke


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    result = args.handler(args)
    if result is not None:
        print(canonical_json(result))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chart-provenance",
        description="Generate and evaluate the chart provenance benchmark.",
    )
    subparsers = parser.add_subparsers(required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--profile", choices=("full", "smoke"), default="full")
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(handler=_generate)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.set_defaults(handler=_audit)

    attacks = subparsers.add_parser("generate-attacks")
    attacks.add_argument("--dataset", type=Path, required=True)
    attacks.add_argument("--split", required=True)
    attacks.add_argument("--output", type=Path, required=True)
    attacks.set_defaults(handler=_attacks)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(handler=_evaluate)

    repair = subparsers.add_parser("repair-predictions")
    repair.add_argument("--input", type=Path, required=True)
    repair.add_argument("--output", type=Path, required=True)
    repair.set_defaults(handler=_repair_predictions)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--treatment", type=Path, required=True)
    analyze.add_argument("--control", type=Path, required=True)
    analyze.add_argument("--metric", default="joint_exact")
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(handler=_analyze)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output", type=Path)
    smoke.set_defaults(handler=_smoke)

    model_smoke = subparsers.add_parser("model-smoke")
    model_smoke.add_argument("--model-path", type=Path)
    model_smoke.add_argument("--local-files-only", action="store_true")
    model_smoke.set_defaults(handler=_model_smoke)
    return parser


def _generate(args: argparse.Namespace) -> dict[str, Any]:
    config = smoke_config() if args.profile == "smoke" else default_config()
    return generate_dataset(args.output, config, overwrite=args.overwrite)


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    report = audit_dataset(args.dataset)
    if report["status"] != "pass":
        raise SystemExit(1)
    return report


def _attacks(args: argparse.Namespace) -> dict[str, int]:
    records = _split_records(args.dataset, args.split)
    rows = [attack for record in records for attack in attack_predictions(record)]
    write_jsonl(args.output, rows)
    return {"question_count": len(records), "attack_count": len(rows)}


def _evaluate(args: argparse.Namespace) -> dict[str, float | int]:
    report = aggregate_predictions(
        _split_records(args.dataset, args.split),
        load_jsonl(args.predictions),
    )
    if args.output:
        write_json(args.output, report)
    return report


def _repair_predictions(args: argparse.Namespace) -> dict[str, int]:
    repaired_rows: list[dict[str, Any]] = []
    repaired_count = 0
    for row in load_jsonl(args.input):
        raw_prediction = row.get("prediction")
        if not isinstance(raw_prediction, str):
            raise TypeError("prediction must be a string")
        repaired = repair_output(raw_prediction)
        if repaired is not None:
            repaired_count += int(repaired != raw_prediction)
            row = dict(row)
            row["prediction"] = repaired
            row["postprocessing"] = "deterministic_json_repair"
        repaired_rows.append(row)
    write_jsonl(args.output, repaired_rows)
    return {"row_count": len(repaired_rows), "repaired_count": repaired_count}


def _analyze(args: argparse.Namespace) -> dict[str, Any]:
    report = compare_methods(
        load_jsonl(args.treatment),
        load_jsonl(args.control),
        metric=args.metric,
    )
    if args.output:
        write_json(args.output, report)
    return report


def _smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.output:
        return run_cpu_smoke(args.output)
    with tempfile.TemporaryDirectory(prefix="chart-provenance-smoke-") as directory:
        return run_cpu_smoke(Path(directory) / "dataset")


def _model_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from chart_provenance_rl.training.model import model_stack_smoke

    return model_stack_smoke(
        model_source=args.model_path if args.model_path else MODEL_ID,
        local_files_only=args.local_files_only,
    )


def _split_records(dataset: Path, split: str) -> list[BenchmarkRecord]:
    records = load_dataset_records(dataset)
    try:
        return records[split]
    except KeyError as error:
        raise KeyError(f"Unknown split: {split}") from error


if __name__ == "__main__":
    main()
