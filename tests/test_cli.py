import json
import sys
from pathlib import Path
from typing import Any

import pytest

from chart_provenance_rl import cli
from chart_provenance_rl.audits import load_dataset_records
from chart_provenance_rl.evaluation.attacks import attack_predictions
from chart_provenance_rl.serialization import write_jsonl


def test_cli_generation_audit_attack_and_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "dataset"
    generated = _run_cli(
        monkeypatch,
        capsys,
        "generate",
        "--output",
        str(dataset),
        "--profile",
        "smoke",
    )
    assert generated["splits"]["train"]["count"] == 8
    audited = _run_cli(
        monkeypatch,
        capsys,
        "audit",
        "--dataset",
        str(dataset),
    )
    assert audited["status"] == "pass"
    attacks_path = tmp_path / "attacks.jsonl"
    attack_report = _run_cli(
        monkeypatch,
        capsys,
        "generate-attacks",
        "--dataset",
        str(dataset),
        "--split",
        "train",
        "--output",
        str(attacks_path),
    )
    assert attack_report["attack_count"] == 48

    records = load_dataset_records(dataset)["train"]
    oracle_path = tmp_path / "oracle.jsonl"
    write_jsonl(
        oracle_path,
        (
            {
                "record_id": record.record_id,
                "prediction": attack_predictions(record)[0]["prediction"],
            }
            for record in records
        ),
    )
    metrics_path = tmp_path / "metrics.json"
    metrics = _run_cli(
        monkeypatch,
        capsys,
        "evaluate",
        "--dataset",
        str(dataset),
        "--split",
        "train",
        "--predictions",
        str(oracle_path),
        "--output",
        str(metrics_path),
    )
    assert metrics["joint_exact_pass"] == 1.0
    assert metrics_path.is_file()


def test_cli_repair_analysis_and_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = tmp_path / "raw.jsonl"
    repaired = tmp_path / "repaired.jsonl"
    write_jsonl(
        raw,
        [
            {
                "record_id": "q1",
                "prediction": '```json\n{"answer":"1","evidence":[],}\n```',
            }
        ],
    )
    repair_report = _run_cli(
        monkeypatch,
        capsys,
        "repair-predictions",
        "--input",
        str(raw),
        "--output",
        str(repaired),
    )
    assert repair_report["repaired_count"] == 1

    treatment = tmp_path / "treatment.jsonl"
    control = tmp_path / "control.jsonl"
    treatment_rows = _seed_rows(1.0)
    control_rows = _seed_rows(0.0)
    write_jsonl(treatment, treatment_rows)
    write_jsonl(control, control_rows)
    analysis = _run_cli(
        monkeypatch,
        capsys,
        "analyze",
        "--treatment",
        str(treatment),
        "--control",
        str(control),
        "--output",
        str(tmp_path / "analysis.json"),
    )
    assert analysis["paired_question_test"]["mean_difference"] == 1.0

    smoke = _run_cli(
        monkeypatch,
        capsys,
        "smoke",
        "--output",
        str(tmp_path / "smoke"),
    )
    assert smoke["audit"]["status"] == "pass"


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *arguments: str,
) -> dict[str, Any]:
    monkeypatch.setattr(sys, "argv", ["chart-provenance", *arguments])
    cli.main()
    value = json.loads(capsys.readouterr().out)
    if not isinstance(value, dict):
        raise TypeError("CLI result must be an object")
    return value


def _seed_rows(score: float) -> list[dict[str, str | int | float]]:
    return [
        {"record_id": f"q{question}", "seed": seed, "joint_exact": score}
        for seed in (17, 29, 43)
        for question in range(4)
    ]
