from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chart_provenance_rl.evaluation.confirmatory import (
    _valid_time_range,
    run_confirmatory_analysis,
)
from chart_provenance_rl.evaluation.rollouts import (
    aggregate_rollout_diagnostics,
    bind_completed_rollout_artifact,
    reconcile_rollout_log,
)
from chart_provenance_rl.serialization import (
    load_jsonl,
    sha256_json,
    write_json,
    write_jsonl,
)

SEEDS = (17, 29, 43)


def test_confirmatory_analysis_passes_only_with_complete_artifacts(
    tmp_path: Path,
) -> None:
    paths = _write_confirmatory_artifacts(tmp_path)
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "pass"
    assert all(criterion["status"] == "pass" for criterion in report["criteria"])
    assert report["rollout_rows_are_independent"] is False
    assert report["rollout_diagnostics"]["statistical_unit"].lower().startswith("non-independent")


def test_confirmatory_analysis_is_incomplete_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    paths = _write_confirmatory_artifacts(tmp_path)
    paths["throughput_path"].unlink()
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "incomplete"
    assert "throughput" in report["missing_artifacts"]
    assert all(criterion["status"] == "incomplete" for criterion in report["criteria"])


def test_confirmatory_analysis_fails_closed_on_bad_mechanism(
    tmp_path: Path,
) -> None:
    paths = _write_confirmatory_artifacts(tmp_path)
    write_jsonl(
        paths["naive_path"],
        _question_rows("grpo_naive_recall", joint=0.0, cite_all=0.05),
    )
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "fail"
    criterion = {item["name"]: item for item in report["criteria"]}["naive_cite_all_vs_minimal"]
    assert criterion["status"] == "fail"


def test_confirmatory_analysis_fails_on_invalid_artifact_shape(
    tmp_path: Path,
) -> None:
    paths = _write_confirmatory_artifacts(tmp_path)
    rows = _question_rows(
        "grpo_minimal_provenance",
        joint=1.0,
        cite_all=0.0,
    )
    rows[0]["seed"] = 999
    rows[0]["adapter_seed"] = 999
    write_jsonl(paths["minimal_path"], rows)
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "fail"
    assert "requires seeds" in report["artifact_validation_error"]
    assert all(criterion["status"] == "incomplete" for criterion in report["criteria"])


def test_confirmatory_throughput_failure_is_explicit(
    tmp_path: Path,
) -> None:
    paths = _write_confirmatory_artifacts(tmp_path)
    throughput = _throughput_report()
    for run in throughput["runs"]:
        if run["seed"] == 43:
            run["status"] = "failed"
            run["steps"] = 50
    throughput["collection_sha256"] = _collection_sha256(throughput)
    write_json(paths["throughput_path"], throughput)
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "fail"
    criterion = {item["name"]: item for item in report["criteria"]}["t4_three_seed_throughput"]
    assert criterion["status"] == "fail"


def test_confirmatory_rejects_self_attested_or_tampered_throughput(
    tmp_path: Path,
) -> None:
    paths = _write_confirmatory_artifacts(tmp_path / "self-attested")
    write_json(
        paths["throughput_path"],
        {"hardware": "Standard_NC4as_T4_v3", "runs": []},
    )
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "fail"
    assert "Azure ML collector" in report["artifact_validation_error"]

    paths = _write_confirmatory_artifacts(tmp_path / "tampered")
    throughput = _throughput_report()
    throughput["runs"][0]["wall_seconds"] = 1
    write_json(paths["throughput_path"], throughput)
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "fail"
    assert "identity hash" in report["artifact_validation_error"]


def test_throughput_time_ranges_require_consistent_zoned_duration() -> None:
    valid = {
        "start_time": "2026-08-31T00:00:00+00:00",
        "end_time": "2026-08-31T01:00:00+00:00",
        "wall_seconds": 3600,
    }
    assert _valid_time_range(valid)
    assert not _valid_time_range({**valid, "wall_seconds": 1})
    assert not _valid_time_range({**valid, "start_time": "not-a-time"})
    assert not _valid_time_range({**valid, "start_time": "2026-08-31T00:00:00"})


def test_confirmatory_rejects_truncated_rollout_trajectory(
    tmp_path: Path,
) -> None:
    paths = _write_confirmatory_artifacts(tmp_path)
    truncated = [row for row in _rollout_rows() if int(row["global_step"]) < 4]
    write_jsonl(paths["rollout_paths"][0], truncated)
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "fail"
    criterion = {item["name"]: item for item in report["criteria"]}["rollout_diagnostics_complete"]
    assert criterion["status"] == "fail"
    assert criterion["observed"]["committed_step_coverage"] is False


def test_confirmatory_rejects_probe_adapter_or_dataset_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_confirmatory_artifacts(tmp_path)
    probe_rows = _question_rows(
        "grpo_minimal_question_only",
        joint=0.0,
        cite_all=0.0,
        answer_em=0.0,
    )
    for row in probe_rows:
        if row["seed"] == 17:
            row["adapter_artifact_sha256"] = "f" * 64
    write_jsonl(paths["question_only_path"], probe_rows)
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "fail"
    assert "same minimal adapter" in report["artifact_validation_error"]


def test_confirmatory_rejects_rows_unlinked_from_audit_or_sft(
    tmp_path: Path,
) -> None:
    paths = _write_confirmatory_artifacts(tmp_path)
    naive_rows = _question_rows(
        "grpo_naive_recall",
        joint=0.0,
        cite_all=0.20,
    )
    for row in naive_rows:
        if row["seed"] == 17:
            row["sft_adapter_sha256"] = "e" * 64
    write_jsonl(paths["naive_path"], naive_rows)
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "fail"
    assert "scored SFT adapters" in report["artifact_validation_error"]

    paths = _write_confirmatory_artifacts(tmp_path / "dataset-mismatch")
    audit = {
        "status": "pass",
        "cross_split_hash_overlap_count": 0,
        "dataset_manifest_sha256": "e" * 64,
        "split_record_ids": {
            "train": [f"train-{question}" for question in range(4)],
            "test_composition": [f"q{question}" for question in range(48)],
        },
    }
    write_json(paths["audit_path"], audit)
    report = run_confirmatory_analysis(**paths)
    assert report["status"] == "fail"
    assert "audited dataset" in report["artifact_validation_error"]


def test_rollout_aggregator_rejects_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.jsonl"
    rows = _rollout_rows()
    write_jsonl(path, [rows[0], rows[0]])
    try:
        aggregate_rollout_diagnostics([path])
    except ValueError as error:
        assert "Duplicate rollout diagnostic row" in str(error)
    else:
        raise AssertionError("Duplicate rollout diagnostics unexpectedly passed")


def test_rollout_aggregator_weights_questions_not_rows(tmp_path: Path) -> None:
    path = tmp_path / "unbalanced-rollouts.jsonl"
    rows = [
        {
            "global_step": index,
            "record_id": "frequent",
            "reward_arm": "naive_recall",
            "seed": 17,
            "run_manifest_sha256": "a" * 64,
            "adapter_artifact_sha256": "b" * 64,
            "reward_call_index": index,
            "completion_index": 0,
            "selected_reward": 1.0,
            "hidden_reward": 0.0,
            "evidence_precision": 0.0,
            "joint_exact": 0.0,
        }
        for index in range(10)
    ]
    rows.append(
        {
            "global_step": 10,
            "record_id": "rare",
            "reward_arm": "naive_recall",
            "seed": 17,
            "run_manifest_sha256": "a" * 64,
            "adapter_artifact_sha256": "b" * 64,
            "reward_call_index": 10,
            "completion_index": 0,
            "selected_reward": 0.0,
            "hidden_reward": 0.0,
            "evidence_precision": 0.0,
            "joint_exact": 0.0,
        }
    )
    write_jsonl(path, rows)
    summary = aggregate_rollout_diagnostics([path])
    group = summary["groups"][0]
    assert group["mean_selected_reward"] == 0.5
    assert group["question_rollout_counts"] == {
        "frequent": 10,
        "rare": 1,
    }


def test_rollout_reconcile_drops_uncommitted_rows(tmp_path: Path) -> None:
    path = tmp_path / "resume-rollouts.jsonl"
    rows = _rollout_rows()[:30]
    write_jsonl(path, rows)
    reconcile_rollout_log(path, str(tmp_path / "checkpoint-3"))
    retained = [row for row in rows if int(row["global_step"]) < 3]
    assert load_jsonl(path) == retained
    reconcile_rollout_log(path, None)
    assert load_jsonl(path) == []
    write_jsonl(path, rows)
    try:
        reconcile_rollout_log(path, str(tmp_path / "bad-checkpoint"))
    except ValueError as error:
        assert "Invalid checkpoint name" in str(error)
    else:
        raise AssertionError("Invalid checkpoint unexpectedly reconciled")


def test_completed_rollout_is_bound_to_adapter_weights(tmp_path: Path) -> None:
    path = tmp_path / "unbound-rollouts.jsonl"
    row = _rollout_rows()[0]
    adapter_hash = str(row.pop("adapter_artifact_sha256"))
    write_jsonl(path, [row])
    bind_completed_rollout_artifact(
        path,
        run_manifest_sha256=str(row["run_manifest_sha256"]),
        adapter_artifact_sha256=adapter_hash,
    )
    assert load_jsonl(path)[0]["adapter_artifact_sha256"] == adapter_hash
    with pytest.raises(ValueError, match="valid run and adapter hashes"):
        bind_completed_rollout_artifact(
            path,
            run_manifest_sha256="bad",
            adapter_artifact_sha256=adapter_hash,
        )
    empty = tmp_path / "empty-rollouts.jsonl"
    write_jsonl(empty, [])
    with pytest.raises(ValueError, match="no rollout diagnostics"):
        bind_completed_rollout_artifact(
            empty,
            run_manifest_sha256="a" * 64,
            adapter_artifact_sha256="b" * 64,
        )
    mismatched = tmp_path / "mismatched-rollouts.jsonl"
    write_jsonl(
        mismatched,
        [{**row, "run_manifest_sha256": "c" * 64}],
    )
    with pytest.raises(ValueError, match="completed run manifest"):
        bind_completed_rollout_artifact(
            mismatched,
            run_manifest_sha256="a" * 64,
            adapter_artifact_sha256="b" * 64,
        )


def test_confirm_cli_exit_codes_follow_report_status(tmp_path: Path) -> None:
    passing = _write_confirmatory_artifacts(tmp_path / "pass")
    pass_result = subprocess.run(
        _confirm_command(passing, tmp_path / "pass-report.json"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert pass_result.returncode == 0
    assert json.loads(pass_result.stdout)["status"] == "pass"

    failing = _write_confirmatory_artifacts(tmp_path / "fail")
    write_jsonl(
        failing["naive_path"],
        _question_rows(
            "grpo_naive_recall",
            joint=0.0,
            cite_all=0.05,
        ),
    )
    fail_result = subprocess.run(
        _confirm_command(failing, tmp_path / "fail-report.json"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert fail_result.returncode == 1
    assert json.loads(fail_result.stdout)["status"] == "fail"

    incomplete = _write_confirmatory_artifacts(tmp_path / "incomplete")
    incomplete["throughput_path"].unlink()
    incomplete_result = subprocess.run(
        _confirm_command(
            incomplete,
            tmp_path / "incomplete-report.json",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert incomplete_result.returncode == 2
    assert json.loads(incomplete_result.stdout)["status"] == "incomplete"


def _write_confirmatory_artifacts(
    root: Path,
) -> dict[str, Any]:
    sft = root / "sft.jsonl"
    naive = root / "naive.jsonl"
    minimal = root / "minimal.jsonl"
    question_only = root / "question-only.jsonl"
    shuffled = root / "shuffled.jsonl"
    audit = root / "audit.json"
    throughput = root / "throughput.json"
    rollouts = root / "rollouts.jsonl"
    write_jsonl(sft, _question_rows("sft", joint=0.25, cite_all=0.05))
    write_jsonl(
        naive,
        _question_rows("grpo_naive_recall", joint=0.0, cite_all=0.20),
    )
    write_jsonl(
        minimal,
        _question_rows("grpo_minimal_provenance", joint=1.0, cite_all=0.0),
    )
    write_jsonl(
        question_only,
        _question_rows(
            "grpo_minimal_question_only",
            joint=0.0,
            cite_all=0.0,
            answer_em=0.0,
        ),
    )
    write_jsonl(
        shuffled,
        _question_rows(
            "grpo_minimal_image_shuffle",
            joint=0.0,
            cite_all=0.0,
            answer_em=0.20,
        ),
    )
    write_json(
        audit,
        {
            "status": "pass",
            "cross_split_hash_overlap_count": 0,
            "dataset_manifest_sha256": "d" * 64,
            "split_record_ids": {
                "train": [f"train-{question}" for question in range(4)],
                "test_composition": [f"q{question}" for question in range(48)],
            },
            "within_split_duplicate_hash_count": {
                "record": 0,
                "image": 0,
                "table": 0,
            },
        },
    )
    write_json(throughput, _throughput_report())
    write_jsonl(rollouts, _rollout_rows())
    return {
        "sft_path": sft,
        "naive_path": naive,
        "minimal_path": minimal,
        "question_only_path": question_only,
        "shuffled_path": shuffled,
        "audit_path": audit,
        "throughput_path": throughput,
        "rollout_paths": [rollouts],
    }


def _confirm_command(
    paths: dict[str, Any],
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "chart_provenance_rl.cli",
        "confirm",
        "--sft",
        str(paths["sft_path"]),
        "--naive",
        str(paths["naive_path"]),
        "--minimal",
        str(paths["minimal_path"]),
        "--question-only",
        str(paths["question_only_path"]),
        "--shuffled",
        str(paths["shuffled_path"]),
        "--dataset-audit",
        str(paths["audit_path"]),
        "--throughput",
        str(paths["throughput_path"]),
        "--rollout-diagnostics",
        *(str(path) for path in paths["rollout_paths"]),
        "--output",
        str(output),
    ]


def _question_rows(
    method: str,
    *,
    joint: float,
    cite_all: float,
    answer_em: float = 0.90,
) -> list[dict[str, Any]]:
    stage = "sft" if method == "sft" else "grpo"
    reward_arm = {
        "sft": None,
        "grpo_naive_recall": "naive_recall",
        "grpo_minimal_provenance": "minimal_provenance",
        "grpo_minimal_question_only": "minimal_provenance",
        "grpo_minimal_image_shuffle": "minimal_provenance",
    }[method]
    image_mode = {
        "sft": "normal",
        "grpo_naive_recall": "normal",
        "grpo_minimal_provenance": "normal",
        "grpo_minimal_question_only": "blank",
        "grpo_minimal_image_shuffle": "shuffled",
    }[method]
    return [
        {
            "record_id": f"q{question}",
            "seed": seed,
            "split": "test_composition",
            "method": method,
            "answer_em": answer_em,
            "joint_exact": joint,
            "cite_all": cite_all,
            "overcitation": cite_all,
            "adapter_artifact_sha256": _adapter_hash(method, seed),
            "run_manifest_sha256": _run_hash(method, seed),
            "run_max_steps": 100,
            "adapter_stage": stage,
            "adapter_seed": seed,
            "adapter_reward_arm": reward_arm,
            "sft_adapter_sha256": (None if method == "sft" else _adapter_hash("sft", seed)),
            "dataset_manifest_sha256": "d" * 64,
            "image_mode": image_mode,
            "postprocessing": "none",
        }
        for seed in SEEDS
        for question in range(48)
    ]


def _rollout_rows() -> list[dict[str, Any]]:
    rows = []
    for arm in ("naive_recall", "minimal_provenance"):
        for seed in SEEDS:
            for step in range(100):
                naive = arm == "naive_recall"
                rows.append(
                    {
                        "global_step": step,
                        "record_id": f"train-{step % 4}",
                        "reward_arm": arm,
                        "seed": seed,
                        "run_manifest_sha256": _run_hash(
                            ("grpo_naive_recall" if naive else "grpo_minimal_provenance"),
                            seed,
                        ),
                        "adapter_artifact_sha256": _adapter_hash(
                            ("grpo_naive_recall" if naive else "grpo_minimal_provenance"),
                            seed,
                        ),
                        "reward_call_index": step,
                        "completion_index": 0,
                        "selected_reward": 1.0 if naive else 0.9,
                        "hidden_reward": 0.4 if naive else 0.85,
                        "answer_exact": True,
                        "evidence_precision": 0.3 if naive else 0.9,
                        "evidence_recall": 1.0,
                        "evidence_f1": 0.45 if naive else 0.9,
                        "joint_exact": 0.3 if naive else 0.8,
                        "valid_schema": True,
                        "overcitation_count": 4 if naive else 0,
                    }
                )
    return rows


def _adapter_hash(method: str, seed: int) -> str:
    prefixes = {
        "sft": 1,
        "grpo_naive_recall": 2,
        "grpo_minimal_provenance": 3,
        "grpo_minimal_question_only": 3,
        "grpo_minimal_image_shuffle": 3,
    }
    return f"{prefixes[method] * 1000 + seed:064x}"


def _run_hash(method: str, seed: int) -> str:
    prefixes = {
        "sft": 4,
        "grpo_naive_recall": 5,
        "grpo_minimal_provenance": 6,
        "grpo_minimal_question_only": 6,
        "grpo_minimal_image_shuffle": 6,
    }
    return f"{prefixes[method] * 1000 + seed:064x}"


def _run_hash_for_stage(stage: str, seed: int) -> str:
    methods = {
        "sft": "sft",
        "naive_recall": "grpo_naive_recall",
        "minimal_provenance": "grpo_minimal_provenance",
    }
    return _run_hash(methods[stage], seed)


def _adapter_hash_for_stage(stage: str, seed: int) -> str:
    methods = {
        "sft": "sft",
        "naive_recall": "grpo_naive_recall",
        "minimal_provenance": "grpo_minimal_provenance",
    }
    return _adapter_hash(methods[stage], seed)


def _throughput_report() -> dict[str, Any]:
    component_names = {
        "sft": "sft",
        "naive_recall": "grpo_naive",
        "minimal_provenance": "grpo_minimal",
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "evidence_source": "azure_ml_v2_cli",
        "compute": {
            "name": "gpu-t4-dedicated",
            "size": "Standard_NC4as_T4_v3",
        },
        "pipeline_jobs": [
            {
                "name": f"pipeline-{seed}",
                "status": "completed",
                "seed": seed,
            }
            for seed in SEEDS
        ],
        "git_commit": "f" * 40,
        "source_sha256": "7" * 64,
        "dataset_manifest_sha256": "d" * 64,
        "runs": [
            {
                "seed": seed,
                "stage": stage,
                "status": "completed",
                "steps": 100,
                "max_steps": 100,
                "wall_seconds": 3600.0,
                "parent_job_name": f"pipeline-{seed}",
                "job_name": f"{component_names[stage]}-{seed}",
                "component_name": component_names[stage],
                "compute": "gpu-t4-dedicated",
                "compute_size": "Standard_NC4as_T4_v3",
                "start_time": "2026-08-31T00:00:00+00:00",
                "end_time": "2026-08-31T01:00:00+00:00",
                "git_commit": "f" * 40,
                "source_sha256": "7" * 64,
                "dataset_manifest_sha256": "d" * 64,
                "training_config_sha256": ("8" * 64 if stage == "sft" else "9" * 64),
                "seed_invariant_training_config_sha256": ("a" * 64 if stage == "sft" else "b" * 64),
                "run_manifest_sha256": _run_hash_for_stage(
                    stage,
                    seed,
                ),
                "adapter_artifact_sha256": _adapter_hash_for_stage(
                    stage,
                    seed,
                ),
            }
            for seed in SEEDS
            for stage in ("sft", "naive_recall", "minimal_provenance")
        ],
    }
    report["collection_sha256"] = _collection_sha256(report)
    return report


def _collection_sha256(report: dict[str, Any]) -> str:
    return sha256_json({key: value for key, value in report.items() if key != "collection_sha256"})
