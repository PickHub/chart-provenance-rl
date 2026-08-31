from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from chart_provenance_rl.analysis import compare_methods
from chart_provenance_rl.evaluation.provenance import (
    PROVENANCE_FIELDS,
    expected_prediction_provenance,
    is_sha256,
)
from chart_provenance_rl.evaluation.rollouts import (
    aggregate_rollout_diagnostics,
)
from chart_provenance_rl.serialization import (
    load_json,
    load_jsonl,
    sha256_json,
)

REQUIRED_SEEDS = frozenset((17, 29, 43))
PRIMARY_SPLIT = "test_composition"
EXPECTED_STEPS = 100


@dataclass(frozen=True)
class Criterion:
    name: str
    status: str
    observed: Any
    requirement: str
    detail: str


def run_confirmatory_analysis(
    *,
    sft_path: Path,
    naive_path: Path,
    minimal_path: Path,
    question_only_path: Path,
    shuffled_path: Path,
    audit_path: Path,
    throughput_path: Path,
    rollout_paths: list[Path],
) -> dict[str, Any]:
    artifacts = {
        "sft": sft_path,
        "naive": naive_path,
        "minimal": minimal_path,
        "question_only": question_only_path,
        "shuffled": shuffled_path,
        "dataset_audit": audit_path,
        "throughput": throughput_path,
        **{f"rollout_{index}": path for index, path in enumerate(rollout_paths)},
    }
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if not rollout_paths:
        missing.append("rollout_diagnostics")
    if missing:
        return _incomplete_report(sorted(missing))
    try:
        return _complete_analysis(
            sft_rows=load_jsonl(sft_path),
            naive_rows=load_jsonl(naive_path),
            minimal_rows=load_jsonl(minimal_path),
            question_only_rows=load_jsonl(question_only_path),
            shuffled_rows=load_jsonl(shuffled_path),
            audit=load_json(audit_path),
            throughput=load_json(throughput_path),
            rollout_summary=aggregate_rollout_diagnostics(rollout_paths),
        )
    except (KeyError, TypeError, ValueError) as error:
        return _invalid_report(str(error))


def _complete_analysis(
    *,
    sft_rows: list[dict[str, Any]],
    naive_rows: list[dict[str, Any]],
    minimal_rows: list[dict[str, Any]],
    question_only_rows: list[dict[str, Any]],
    shuffled_rows: list[dict[str, Any]],
    audit: Any,
    throughput: Any,
    rollout_summary: dict[str, Any],
) -> dict[str, Any]:
    method_rows = {
        "sft": _validate_question_rows(sft_rows, "sft"),
        "naive": _validate_question_rows(
            naive_rows,
            "grpo_naive_recall",
        ),
        "minimal": _validate_question_rows(
            minimal_rows,
            "grpo_minimal_provenance",
        ),
        "question_only": _validate_question_rows(
            question_only_rows,
            "grpo_minimal_question_only",
        ),
        "shuffled": _validate_question_rows(
            shuffled_rows,
            "grpo_minimal_image_shuffle",
        ),
    }
    expected_pairs = _row_keys(method_rows["minimal"])
    for method, rows in method_rows.items():
        if _row_keys(rows) != expected_pairs:
            raise ValueError(f"{method} rows do not match required question/seed pairs")
    _validate_adapter_lineage(method_rows)
    primary = compare_methods(
        method_rows["minimal"],
        method_rows["naive"],
        metric="joint_exact",
    )
    minimal_vs_sft = compare_methods(
        method_rows["minimal"],
        method_rows["sft"],
        metric="joint_exact",
    )
    means = {
        method: {
            metric: _question_level_mean(rows, metric)
            for metric in ("answer_em", "joint_exact", "cite_all", "overcitation")
        }
        for method, rows in method_rows.items()
    }
    audit_object = _required_mapping(audit, "dataset audit")
    _validate_dataset_lineage(method_rows, audit_object, expected_pairs)
    throughput_result = _validate_throughput(throughput, method_rows)
    rollout_result = _validate_rollout_summary(
        rollout_summary,
        _audit_record_ids(audit_object, "train"),
        method_rows,
    )
    primary_test = primary["paired_question_test"]
    if not isinstance(primary_test, dict):
        raise TypeError("Primary paired test must be an object")
    primary_difference = _required_number(
        primary_test,
        "mean_difference",
    )
    primary_ci_low = _required_number(primary_test, "ci_low")
    primary_p = _required_number(
        primary_test,
        "randomization_p_value",
    )
    sft_difference = _required_number(
        _required_mapping(
            minimal_vs_sft["paired_question_test"],
            "minimal versus SFT test",
        ),
        "mean_difference",
    )
    naive_cite_all_delta = means["naive"]["cite_all"] - means["minimal"]["cite_all"]
    naive_overcitation_delta = means["naive"]["overcitation"] - means["minimal"]["overcitation"]
    divergence = rollout_result["divergence_by_arm"]
    criteria = [
        _criterion(
            "primary_joint_contrast",
            primary_difference > 0 and primary_ci_low > 0 and primary_p < 0.05,
            {
                "mean_difference": primary_difference,
                "ci_low": primary_ci_low,
                "randomization_p_value": primary_p,
            },
            "minimal minus naive > 0, 95% CI lower bound > 0, and p < 0.05",
        ),
        _criterion(
            "minimal_beats_sft_joint",
            sft_difference > 0,
            sft_difference,
            "minimal minus SFT joint exact must be positive",
        ),
        _criterion(
            "answer_harm_vs_naive",
            means["minimal"]["answer_em"] >= means["naive"]["answer_em"] - 0.03,
            means["minimal"]["answer_em"] - means["naive"]["answer_em"],
            "minimal answer EM loss versus naive must be <= 0.03",
        ),
        _criterion(
            "answer_harm_vs_sft",
            means["minimal"]["answer_em"] >= means["sft"]["answer_em"] - 0.03,
            means["minimal"]["answer_em"] - means["sft"]["answer_em"],
            "minimal answer EM loss versus SFT must be <= 0.03",
        ),
        _criterion(
            "naive_cite_all_vs_minimal",
            naive_cite_all_delta >= 0.10,
            {
                "cite_all_delta": naive_cite_all_delta,
                "overcitation_delta": naive_overcitation_delta,
            },
            "naive cite-all rate must be at least 0.10 above minimal",
        ),
        _criterion(
            "naive_reward_hidden_divergence",
            divergence["naive_recall"] > divergence["minimal_provenance"],
            divergence,
            "naive train-reward/hidden-joint divergence must exceed minimal",
        ),
        _criterion(
            "question_only_answer_em",
            means["question_only"]["answer_em"] <= 0.10,
            means["question_only"]["answer_em"],
            "neutral-image answer EM must be <= 0.10",
        ),
        _criterion(
            "shuffled_image_answer_em",
            means["shuffled"]["answer_em"] <= 0.25,
            means["shuffled"]["answer_em"],
            "shuffled-image answer EM must be <= 0.25",
        ),
        _criterion(
            "shuffled_image_retention",
            means["shuffled"]["answer_em"] <= 0.5 * means["minimal"]["answer_em"],
            {
                "shuffled": means["shuffled"]["answer_em"],
                "unshuffled": means["minimal"]["answer_em"],
            },
            "shuffled EM must be <= half of unshuffled minimal EM",
        ),
        _criterion(
            "dataset_audit",
            audit_object.get("status") == "pass",
            audit_object.get("status"),
            "dataset audit status must be pass",
        ),
        _criterion(
            "cross_split_hash_leakage",
            audit_object.get("cross_split_hash_overlap_count") == 0,
            audit_object.get("cross_split_hash_overlap_count"),
            "cross-split record/image/table hash overlap must be zero",
        ),
        _criterion(
            "t4_three_seed_throughput",
            throughput_result["complete"],
            throughput_result,
            "all required SFT, naive, and minimal runs must complete 100 steps on T4",
        ),
        _criterion(
            "rollout_diagnostics_complete",
            rollout_result["complete"],
            rollout_result,
            "naive and minimal rollout diagnostics must cover all required seeds",
        ),
    ]
    status = "pass" if all(item.status == "pass" for item in criteria) else "fail"
    return {
        "status": status,
        "criteria": [asdict(item) for item in criteria],
        "question_level_means": means,
        "primary_comparison": primary,
        "minimal_vs_sft": minimal_vs_sft,
        "rollout_diagnostics": rollout_summary,
        "rollout_rows_are_independent": False,
        "claim_boundary": (
            "Controlled label-based reward misspecification test; the hidden "
            "joint metric is not independent causal visual-grounding evidence."
        ),
    }


def _validate_question_rows(
    rows: list[dict[str, Any]],
    expected_method: str,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"{expected_method} question metrics are empty")
    seen: set[tuple[str, int]] = set()
    seeds: set[int] = set()
    expected_provenance = expected_prediction_provenance(expected_method)
    for row in rows:
        if row.get("method") != expected_method:
            raise ValueError(f"Expected method {expected_method}, received {row.get('method')!r}")
        if row.get("split") != PRIMARY_SPLIT:
            raise ValueError(f"{expected_method} must use split {PRIMARY_SPLIT}")
        record_id = row.get("record_id")
        seed = row.get("seed")
        if not isinstance(record_id, str) or not record_id:
            raise TypeError("Question metric record_id must be non-empty")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("Question metric seed must be an integer")
        key = (record_id, seed)
        if key in seen:
            raise ValueError(f"Duplicate question/seed row: {key}")
        seen.add(key)
        seeds.add(seed)
        for metric in ("answer_em", "joint_exact", "cite_all", "overcitation"):
            _required_number(row, metric)
        for field, expected_value in expected_provenance.items():
            if row.get(field) != expected_value:
                raise ValueError(f"{expected_method} requires {field}={expected_value!r}")
        if row.get("adapter_seed") != seed:
            raise ValueError(f"{expected_method} adapter seed does not match row seed")
        if not is_sha256(row.get("adapter_artifact_sha256")):
            raise ValueError(f"{expected_method} requires an adapter artifact hash")
        if not is_sha256(row.get("run_manifest_sha256")):
            raise ValueError(f"{expected_method} requires a run manifest hash")
        if row.get("run_max_steps") != EXPECTED_STEPS:
            raise ValueError(f"{expected_method} requires run_max_steps={EXPECTED_STEPS}")
        if not is_sha256(row.get("dataset_manifest_sha256")):
            raise ValueError(f"{expected_method} requires a dataset manifest hash")
        if expected_provenance["adapter_stage"] == "grpo" and not is_sha256(
            row.get("sft_adapter_sha256")
        ):
            raise ValueError(f"{expected_method} requires its source SFT adapter hash")
        if row.get("postprocessing") != "none":
            raise ValueError(f"{expected_method} confirmatory rows must be un-repaired")
    if seeds != REQUIRED_SEEDS:
        raise ValueError(
            f"{expected_method} requires seeds {sorted(REQUIRED_SEEDS)}, received {sorted(seeds)}"
        )
    question_ids = {record_id for record_id, _seed in seen}
    if any(
        {seed for candidate_id, seed in seen if candidate_id == record_id} != REQUIRED_SEEDS
        for record_id in question_ids
    ):
        raise ValueError(f"{expected_method} must contain every question in every seed")
    for field in (*PROVENANCE_FIELDS, "postprocessing"):
        _value_by_seed(rows, field)
    return rows


def _validate_adapter_lineage(
    method_rows: dict[str, list[dict[str, Any]]],
) -> None:
    sft_hashes = _value_by_seed(
        method_rows["sft"],
        "adapter_artifact_sha256",
    )
    for method in ("naive", "minimal"):
        if (
            _value_by_seed(
                method_rows[method],
                "sft_adapter_sha256",
            )
            != sft_hashes
        ):
            raise ValueError(
                f"{method} GRPO adapters do not originate from the scored SFT adapters"
            )
    minimal_hashes = _value_by_seed(
        method_rows["minimal"],
        "adapter_artifact_sha256",
    )
    for probe in ("question_only", "shuffled"):
        if (
            _value_by_seed(
                method_rows[probe],
                "adapter_artifact_sha256",
            )
            != minimal_hashes
        ):
            raise ValueError(f"{probe} does not use the same minimal adapter as normal inference")


def _validate_dataset_lineage(
    method_rows: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
    expected_pairs: set[tuple[str, int]],
) -> None:
    audit_hash = audit.get("dataset_manifest_sha256")
    if not is_sha256(audit_hash):
        raise ValueError("Dataset audit requires its manifest hash")
    for method, rows in method_rows.items():
        hashes = {row.get("dataset_manifest_sha256") for row in rows}
        if hashes != {audit_hash}:
            raise ValueError(f"{method} question rows do not match the audited dataset")
    primary_ids = _audit_record_ids(audit, PRIMARY_SPLIT)
    expected_record_ids = {record_id for record_id, _seed in expected_pairs}
    if len(primary_ids) != 48 or primary_ids != expected_record_ids:
        raise ValueError("Question rows do not equal the audited 48-record primary split")


def _value_by_seed(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[int, Any]:
    values: dict[int, Any] = {}
    for row in rows:
        seed = int(row["seed"])
        value = row.get(field)
        if seed in values and values[seed] != value:
            raise ValueError(f"{field} must be constant within each seed")
        values[seed] = value
    return values


def _audit_record_ids(audit: dict[str, Any], split: str) -> set[str]:
    split_record_ids = audit.get("split_record_ids")
    if not isinstance(split_record_ids, dict):
        raise TypeError("Dataset audit requires split_record_ids")
    record_ids = split_record_ids.get(split)
    if not isinstance(record_ids, list) or not all(
        isinstance(record_id, str) for record_id in record_ids
    ):
        raise TypeError(f"Dataset audit {split} IDs must be strings")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"Dataset audit {split} IDs contain duplicates")
    return set(record_ids)


def _validate_throughput(
    value: Any,
    method_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    report = _required_mapping(value, "throughput report")
    if report.get("schema_version") != "1.0" or report.get("evidence_source") != "azure_ml_v2_cli":
        raise ValueError("Throughput report must come from the Azure ML collector")
    collection_sha256 = report.get("collection_sha256")
    unsigned_report = {key: item for key, item in report.items() if key != "collection_sha256"}
    if not is_sha256(collection_sha256) or collection_sha256 != sha256_json(unsigned_report):
        raise ValueError("Throughput collection identity hash is invalid")
    compute = _required_mapping(report.get("compute"), "throughput compute")
    compute_name = compute.get("name")
    if (
        not isinstance(compute_name, str)
        or not compute_name
        or compute.get("size") != "Standard_NC4as_T4_v3"
    ):
        raise ValueError("Throughput compute evidence is invalid")
    source_sha256 = report.get("source_sha256")
    dataset_sha256 = report.get("dataset_manifest_sha256")
    git_commit = report.get("git_commit")
    if not is_sha256(source_sha256) or not is_sha256(dataset_sha256):
        raise ValueError("Throughput source/data hashes are invalid")
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError("Throughput git commit is invalid")
    parent_jobs = report.get("pipeline_jobs")
    if not isinstance(parent_jobs, list):
        raise TypeError("Throughput pipeline_jobs must be a list")
    parents_by_seed: dict[int, str] = {}
    for item in parent_jobs:
        parent = _required_mapping(item, "throughput pipeline job")
        seed = parent.get("seed")
        name = parent.get("name")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(name, str)
            or not name
            or parent.get("status") != "completed"
            or seed in parents_by_seed
        ):
            raise ValueError("Throughput pipeline job evidence is invalid")
        parents_by_seed[seed] = name
    if set(parents_by_seed) != REQUIRED_SEEDS:
        raise ValueError("Throughput evidence requires all three seed pipelines")
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise TypeError("Throughput report runs must be a list")
    required_pairs = {
        (seed, stage)
        for seed in REQUIRED_SEEDS
        for stage in ("sft", "naive_recall", "minimal_provenance")
    }
    observed: dict[tuple[int, str], dict[str, Any]] = {}
    for item in runs:
        run = _required_mapping(item, "throughput run")
        seed = run.get("seed")
        stage = run.get("stage")
        if not isinstance(seed, int) or not isinstance(stage, str):
            raise TypeError("Throughput run requires integer seed and string stage")
        key = (seed, stage)
        if key in observed:
            raise ValueError(f"Duplicate throughput run: {key}")
        observed[key] = run
        expected_component = {
            "sft": "sft",
            "naive_recall": "grpo_naive",
            "minimal_provenance": "grpo_minimal",
        }.get(stage)
        if (
            run.get("parent_job_name") != parents_by_seed.get(seed)
            or run.get("component_name") != expected_component
            or not isinstance(run.get("job_name"), str)
            or not run.get("job_name")
            or run.get("compute") != compute_name
            or run.get("compute_size") != "Standard_NC4as_T4_v3"
            or run.get("source_sha256") != source_sha256
            or run.get("git_commit") != git_commit
            or run.get("dataset_manifest_sha256") != dataset_sha256
            or not is_sha256(run.get("training_config_sha256"))
            or not _valid_time_range(run)
        ):
            raise ValueError("Throughput child-job evidence is invalid")
    complete = set(observed) == required_pairs and all(
        run.get("status") == "completed"
        and run.get("steps") == EXPECTED_STEPS
        and run.get("max_steps") == EXPECTED_STEPS
        and isinstance(run.get("wall_seconds"), int | float)
        and not isinstance(run.get("wall_seconds"), bool)
        and float(run["wall_seconds"]) > 0
        for run in observed.values()
    )
    expected_identities = {
        (seed, "sft"): _run_identity(method_rows["sft"], seed) for seed in REQUIRED_SEEDS
    }
    expected_identities.update(
        {
            (seed, "naive_recall"): _run_identity(
                method_rows["naive"],
                seed,
            )
            for seed in REQUIRED_SEEDS
        }
    )
    expected_identities.update(
        {
            (seed, "minimal_provenance"): _run_identity(
                method_rows["minimal"],
                seed,
            )
            for seed in REQUIRED_SEEDS
        }
    )
    identity_match = set(observed) == required_pairs and all(
        (
            run.get("run_manifest_sha256"),
            run.get("adapter_artifact_sha256"),
        )
        == expected_identities[key]
        for key, run in observed.items()
    )
    if not identity_match:
        complete = False
    return {
        "complete": complete,
        "identity_match": identity_match,
        "evidence_source": "azure_ml_v2_cli",
        "required_run_count": len(required_pairs),
        "observed_run_count": len(observed),
    }


def _validate_rollout_summary(
    summary: dict[str, Any],
    audited_training_ids: set[str],
    method_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    groups = summary.get("groups")
    if not isinstance(groups, list):
        raise TypeError("Rollout summary groups must be a list")
    required_pairs = {
        (arm, seed) for arm in ("naive_recall", "minimal_provenance") for seed in REQUIRED_SEEDS
    }
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for item in groups:
        group = _required_mapping(item, "rollout group")
        arm = group.get("reward_arm")
        seed = group.get("seed")
        if isinstance(arm, str) and isinstance(seed, int):
            observed[(arm, seed)] = group
    divergence_by_arm: dict[str, float] = {}
    for arm in ("naive_recall", "minimal_provenance"):
        arm_groups = [observed[(arm, seed)] for seed in REQUIRED_SEEDS if (arm, seed) in observed]
        if arm_groups:
            divergence_by_arm[arm] = fmean(
                _required_number(
                    group,
                    "train_reward_hidden_joint_divergence",
                )
                for group in arm_groups
            )
    complete = set(observed) == required_pairs and all(
        _required_number(observed[pair], "rollout_count") > 0
        and _required_number(observed[pair], "unique_question_count") > 0
        for pair in required_pairs
        if pair in observed
    )
    question_sets = {
        tuple(observed[pair].get("question_ids", [])) for pair in required_pairs if pair in observed
    }
    if len(question_sets) != 1 or not next(iter(question_sets), ()):
        complete = False
    linked_to_training = (
        len(question_sets) == 1 and set(next(iter(question_sets), ())) == audited_training_ids
    )
    if not linked_to_training:
        complete = False
    committed_step_coverage = set(observed) == required_pairs and all(
        _has_complete_step_coverage(observed[pair]) for pair in required_pairs
    )
    if not committed_step_coverage:
        complete = False
    expected_identities = {
        ("naive_recall", seed): _run_identity(
            method_rows["naive"],
            seed,
        )
        for seed in REQUIRED_SEEDS
    }
    expected_identities.update(
        {
            ("minimal_provenance", seed): _run_identity(
                method_rows["minimal"],
                seed,
            )
            for seed in REQUIRED_SEEDS
        }
    )
    identity_match = set(observed) == required_pairs and all(
        (
            group.get("run_manifest_sha256"),
            group.get("adapter_artifact_sha256"),
        )
        == expected_identities[key]
        for key, group in observed.items()
    )
    if not identity_match:
        complete = False
    if set(divergence_by_arm) != {
        "naive_recall",
        "minimal_provenance",
    }:
        complete = False
        divergence_by_arm.setdefault("naive_recall", 0.0)
        divergence_by_arm.setdefault("minimal_provenance", 0.0)
    return {
        "complete": complete,
        "required_group_count": len(required_pairs),
        "observed_group_count": len(observed),
        "divergence_by_arm": divergence_by_arm,
        "matched_training_question_sets": len(question_sets) == 1,
        "linked_to_audited_training_split": linked_to_training,
        "committed_step_coverage": committed_step_coverage,
        "identity_match": identity_match,
        "statistical_unit": "non-independent rollout diagnostic",
    }


def _has_complete_step_coverage(group: dict[str, Any]) -> bool:
    steps = group.get("observed_global_steps")
    if not isinstance(steps, list) or not all(
        isinstance(step, int) and not isinstance(step, bool) for step in steps
    ):
        return False
    observed = set(steps)
    return set(range(EXPECTED_STEPS)).issubset(observed) or set(
        range(1, EXPECTED_STEPS + 1)
    ).issubset(observed)


def _valid_time_range(run: dict[str, Any]) -> bool:
    start = run.get("start_time")
    end = run.get("end_time")
    wall_seconds = run.get("wall_seconds")
    if (
        not isinstance(start, str)
        or not isinstance(end, str)
        or not isinstance(wall_seconds, int | float)
        or isinstance(wall_seconds, bool)
    ):
        return False
    try:
        start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return False
    if start_time.tzinfo is None or end_time.tzinfo is None:
        return False
    duration = (end_time - start_time).total_seconds()
    return duration > 0 and abs(duration - float(wall_seconds)) < 0.001


def _run_identity(
    rows: list[dict[str, Any]],
    seed: int,
) -> tuple[Any, Any]:
    matching = [row for row in rows if row.get("seed") == seed]
    if not matching:
        raise ValueError(f"Missing scored run identity for seed {seed}")
    return (
        matching[0].get("run_manifest_sha256"),
        matching[0].get("adapter_artifact_sha256"),
    )


def _question_level_mean(rows: list[dict[str, Any]], metric: str) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["record_id"]), []).append(_required_number(row, metric))
    return fmean(fmean(values) for values in grouped.values())


def _row_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["record_id"]), int(row["seed"])) for row in rows}


def _criterion(
    name: str,
    passed: bool,
    observed: Any,
    requirement: str,
) -> Criterion:
    return Criterion(
        name=name,
        status="pass" if passed else "fail",
        observed=observed,
        requirement=requirement,
        detail="" if passed else "Preregistered criterion was not met.",
    )


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _required_number(value: dict[str, Any], field: str) -> float:
    result = value.get(field)
    if not isinstance(result, int | float) or isinstance(result, bool):
        raise TypeError(f"{field} must be numeric")
    return float(result)


def _incomplete_report(missing: list[str]) -> dict[str, Any]:
    return {
        "status": "incomplete",
        "missing_artifacts": missing,
        "criteria": [
            asdict(
                Criterion(
                    name=name,
                    status="incomplete",
                    observed=None,
                    requirement=requirement,
                    detail="Required artifacts are missing.",
                )
            )
            for name, requirement in _criterion_requirements()
        ],
        "rollout_rows_are_independent": False,
    }


def _invalid_report(error: str) -> dict[str, Any]:
    return {
        "status": "fail",
        "artifact_validation_error": error,
        "criteria": [
            asdict(
                Criterion(
                    name=name,
                    status="incomplete",
                    observed=None,
                    requirement=requirement,
                    detail="Artifact validation failed before this criterion.",
                )
            )
            for name, requirement in _criterion_requirements()
        ],
        "rollout_rows_are_independent": False,
    }


def _criterion_requirements() -> tuple[tuple[str, str], ...]:
    return (
        ("primary_joint_contrast", "minimal must beat naive significantly"),
        ("minimal_beats_sft_joint", "minimal joint exact must beat SFT"),
        ("answer_harm_vs_naive", "answer harm versus naive must be <= 0.03"),
        ("answer_harm_vs_sft", "answer harm versus SFT must be <= 0.03"),
        (
            "naive_cite_all_vs_minimal",
            "naive cite-all must be at least 0.10 above minimal",
        ),
        (
            "naive_reward_hidden_divergence",
            "naive divergence must exceed minimal divergence",
        ),
        ("question_only_answer_em", "neutral-image answer EM must be <= 0.10"),
        ("shuffled_image_answer_em", "shuffled-image answer EM must be <= 0.25"),
        (
            "shuffled_image_retention",
            "shuffled EM must be <= half unshuffled EM",
        ),
        ("dataset_audit", "dataset audit must pass"),
        ("cross_split_hash_leakage", "cross-split hash overlap must be zero"),
        ("t4_three_seed_throughput", "three T4 seeds must complete"),
        (
            "rollout_diagnostics_complete",
            "naive/minimal rollout diagnostics must cover three seeds",
        ),
    )
