from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any

from chart_provenance_rl.audits import audit_dataset, load_dataset_records
from chart_provenance_rl.decimal_utils import canonical_answer
from chart_provenance_rl.evaluation.attacks import attack_predictions
from chart_provenance_rl.evaluation.diagnostics import counterfactual_diagnostic
from chart_provenance_rl.evaluation.leakage import (
    image_shuffle_map,
    question_only_predictions,
    renderer_family_classifier,
)
from chart_provenance_rl.evaluation.metrics import (
    aggregate_predictions,
    evaluate_prediction,
    reward_hacking_attack_success,
)
from chart_provenance_rl.evaluation.rewards import RewardArm, score_record
from chart_provenance_rl.evaluation.statistics import (
    aggregate_three_seeds,
    paired_question_test,
)
from chart_provenance_rl.generation.config import smoke_config
from chart_provenance_rl.generation.generator import generate_dataset
from chart_provenance_rl.schema import BenchmarkRecord
from chart_provenance_rl.serialization import write_json, write_jsonl


def run_cpu_smoke(root: Path) -> dict[str, Any]:
    generate_dataset(root, smoke_config(), overwrite=True)
    audit = audit_dataset(root)
    if audit["status"] != "pass":
        raise RuntimeError(f"Generated dataset failed audit: {audit['errors']}")
    records_by_split = load_dataset_records(root)
    records = [record for split_records in records_by_split.values() for record in split_records]
    attacks = [attack for record in records for attack in attack_predictions(record)]
    write_jsonl(root / "probes" / "attacks.jsonl", attacks)
    rewards = _reward_grid(records, attacks)
    oracle_rows = _attack_rows(attacks, "oracle_minimal")
    cite_all_rows = _attack_rows(attacks, "cite_all")
    citation_wrong_rows = _attack_rows(
        attacks,
        "answer_correct_citation_wrong",
    )
    oracle_metrics = aggregate_predictions(records, oracle_rows)
    cite_all_metrics = aggregate_predictions(records, cite_all_rows)
    diagnostics = [counterfactual_diagnostic(record) for record in records]
    paired = paired_question_test(
        {
            record.record_id: evaluate_prediction(
                record,
                _prediction_for(cite_all_rows, record.record_id),
            ).hidden_score
            for record in records
        },
        {
            record.record_id: evaluate_prediction(
                record,
                _prediction_for(citation_wrong_rows, record.record_id),
            ).hidden_score
            for record in records
        },
        seed=901,
        bootstrap_iterations=200,
        randomization_iterations=500,
    )
    seeded_rows = [
        {
            "record_id": record.record_id,
            "seed": seed,
            "joint_exact": 1.0,
        }
        for seed in (17, 29, 43)
        for record in records
    ]
    composition = records_by_split["test_composition"]
    question_only = question_only_predictions(
        records_by_split["train"],
        composition,
    )
    question_only_answer_em = fmean(
        canonical_answer(question_only[record.record_id]) == canonical_answer(record.answer)
        for record in composition
    )
    shuffle = image_shuffle_map(composition)
    write_json(root / "probes" / "image_shuffle.json", shuffle)
    report = {
        "report_kind": "synthetic_cpu_smoke_not_model_results",
        "audit": audit,
        "reward_arms": rewards,
        "oracle_metrics": oracle_metrics,
        "cite_all_metrics": cite_all_metrics,
        "reward_hacking_attack_success": reward_hacking_attack_success(
            records,
            attacks,
        ),
        "counterfactual_diagnostic": {
            "question_count": len(diagnostics),
            "mean_necessity": fmean(diagnostic.necessity for diagnostic in diagnostics),
            "mean_sufficiency": fmean(diagnostic.sufficiency for diagnostic in diagnostics),
            "interpretation": (
                "Executable provenance-set diagnostic only; it does not measure whether "
                "the VLM causally used image regions."
            ),
        },
        "paired_question_test": paired.to_dict(),
        "three_seed_aggregation": aggregate_three_seeds(
            seeded_rows,
            metric="joint_exact",
        ),
        "question_only_answer_em": question_only_answer_em,
        "renderer_family_classifier": renderer_family_classifier(root, records),
        "image_shuffle_count": len(shuffle),
    }
    write_json(root / "smoke_report.json", report)
    return report


def _reward_grid(
    records: list[BenchmarkRecord],
    attacks: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    records_by_id = {record.record_id: record for record in records}
    grouped: dict[str, dict[str, list[float]]] = {arm.value: {} for arm in RewardArm}
    for attack in attacks:
        attack_name = str(attack["attack"])
        record = records_by_id[str(attack["record_id"])]
        prediction = str(attack["prediction"])
        for arm in RewardArm:
            reward = score_record(record, prediction, arm).reward
            grouped[arm.value].setdefault(attack_name, []).append(reward)
    return {
        arm: {attack: fmean(values) for attack, values in sorted(attack_values.items())}
        for arm, attack_values in grouped.items()
    }


def _attack_rows(
    attacks: list[dict[str, Any]],
    attack_name: str,
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": str(row["record_id"]),
            "prediction": str(row["prediction"]),
        }
        for row in attacks
        if row["attack"] == attack_name
    ]


def _prediction_for(rows: list[dict[str, Any]], record_id: str) -> str:
    matches = [str(row["prediction"]) for row in rows if row["record_id"] == record_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one prediction for {record_id}, received {len(matches)}")
    return matches[0]
