from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from chart_provenance_rl.evaluation.provenance import is_sha256
from chart_provenance_rl.evaluation.rewards import RewardArm
from chart_provenance_rl.serialization import load_jsonl, write_jsonl


@dataclass(frozen=True)
class _QuestionRolloutSummary:
    record_id: str
    rollout_count: int
    selected_reward: float
    hidden_reward: float
    hidden_joint_exact: float
    hidden_evidence_precision: float


def reconcile_rollout_log(
    path: Path,
    resume_checkpoint: str | None,
) -> None:
    if not path.is_file():
        return
    committed_step = 0
    if resume_checkpoint is not None:
        checkpoint_name = Path(resume_checkpoint).name
        suffix = checkpoint_name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            raise ValueError(f"Invalid checkpoint name: {checkpoint_name}")
        committed_step = int(suffix)
    retained = []
    for row in load_jsonl(path):
        global_step = _required_int(row, "global_step")
        if global_step < committed_step:
            retained.append(row)
    write_jsonl(path, retained)


def bind_completed_rollout_artifact(
    path: Path,
    *,
    run_manifest_sha256: str,
    adapter_artifact_sha256: str,
) -> None:
    if not is_sha256(run_manifest_sha256) or not is_sha256(adapter_artifact_sha256):
        raise ValueError("Rollout binding requires valid run and adapter hashes")
    rows = load_jsonl(path)
    if not rows:
        raise ValueError("Completed GRPO run has no rollout diagnostics")
    bound_rows = []
    for row in rows:
        if row.get("run_manifest_sha256") != run_manifest_sha256:
            raise ValueError("Rollout row does not match the completed run manifest")
        bound_rows.append({**row, "adapter_artifact_sha256": adapter_artifact_sha256})
    write_jsonl(path, bound_rows)


def aggregate_rollout_diagnostics(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one rollout diagnostics artifact is required")
    rows = [row for path in paths for row in load_jsonl(path)]
    if not rows:
        raise ValueError("Rollout diagnostics artifacts are empty")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    rollout_ids: set[tuple[str, int, int, int, int]] = set()
    for row in rows:
        arm = _required_arm(row)
        seed = _required_int(row, "seed")
        global_step = _required_int(row, "global_step")
        call_index = _required_int(row, "reward_call_index")
        completion_index = _required_int(row, "completion_index")
        record_id = _required_string(row, "record_id")
        _required_sha256(
            row,
            "run_manifest_sha256",
        )
        _required_sha256(
            row,
            "adapter_artifact_sha256",
        )
        rollout_id = (
            arm.value,
            seed,
            global_step,
            call_index,
            completion_index,
        )
        if rollout_id in rollout_ids:
            raise ValueError(f"Duplicate rollout diagnostic row: {rollout_id}")
        rollout_ids.add(rollout_id)
        for field in (
            "selected_reward",
            "hidden_reward",
            "evidence_precision",
            "joint_exact",
        ):
            _required_float(row, field)
        if not record_id:
            raise ValueError("Rollout diagnostic record_id must be non-empty")
        grouped[(arm.value, seed)].append(row)
    groups = []
    for (reward_arm, seed), group_rows in sorted(grouped.items()):
        rows_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            rows_by_question[_required_string(row, "record_id")].append(row)
        question_summaries = [
            _QuestionRolloutSummary(
                record_id=record_id,
                rollout_count=len(question_rows),
                selected_reward=fmean(
                    _required_float(row, "selected_reward") for row in question_rows
                ),
                hidden_reward=fmean(_required_float(row, "hidden_reward") for row in question_rows),
                hidden_joint_exact=fmean(
                    _required_float(row, "joint_exact") for row in question_rows
                ),
                hidden_evidence_precision=fmean(
                    _required_float(row, "evidence_precision") for row in question_rows
                ),
            )
            for record_id, question_rows in sorted(rows_by_question.items())
        ]
        mean_selected = fmean(summary.selected_reward for summary in question_summaries)
        mean_joint = fmean(summary.hidden_joint_exact for summary in question_summaries)
        groups.append(
            {
                "reward_arm": reward_arm,
                "seed": seed,
                "run_manifest_sha256": _constant_string(
                    group_rows,
                    "run_manifest_sha256",
                ),
                "adapter_artifact_sha256": _constant_string(
                    group_rows,
                    "adapter_artifact_sha256",
                ),
                "rollout_count": len(group_rows),
                "unique_question_count": len(question_summaries),
                "question_ids": [summary.record_id for summary in question_summaries],
                "question_rollout_counts": {
                    summary.record_id: summary.rollout_count for summary in question_summaries
                },
                "observed_global_steps": sorted(
                    {_required_int(row, "global_step") for row in group_rows}
                ),
                "mean_selected_reward": mean_selected,
                "mean_hidden_reward": fmean(
                    summary.hidden_reward for summary in question_summaries
                ),
                "mean_hidden_joint_exact": mean_joint,
                "mean_hidden_evidence_precision": fmean(
                    summary.hidden_evidence_precision for summary in question_summaries
                ),
                "train_reward_hidden_joint_divergence": (mean_selected - mean_joint),
            }
        )
    return {
        "row_count": len(rows),
        "statistical_unit": (
            "Non-independent rollout diagnostics; never use rows as "
            "question-level inferential samples."
        ),
        "aggregation": (
            "Average completions within question, then average questions "
            "within reward arm and seed."
        ),
        "groups": groups,
    }


def _required_arm(row: dict[str, Any]) -> RewardArm:
    return RewardArm(_required_string(row, "reward_arm"))


def _required_string(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise TypeError(f"Rollout diagnostic {field} must be a string")
    return value


def _required_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Rollout diagnostic {field} must be an integer")
    return value


def _required_float(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"Rollout diagnostic {field} must be numeric")
    return float(value)


def _required_sha256(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not is_sha256(value):
        raise ValueError(f"Rollout diagnostic {field} must be a SHA-256")
    return value


def _constant_string(rows: list[dict[str, Any]], field: str) -> str:
    values = {_required_string(row, field) for row in rows}
    if len(values) != 1:
        raise ValueError(f"Rollout diagnostic {field} must be constant per group")
    return values.pop()
