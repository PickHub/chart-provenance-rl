from pathlib import Path

import pytest

from chart_provenance_rl.analysis import compare_methods
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
from chart_provenance_rl.evaluation.repair import repair_output
from chart_provenance_rl.evaluation.statistics import (
    aggregate_three_seeds,
    paired_question_test,
)
from chart_provenance_rl.schema import BenchmarkRecord, EvidenceRef
from chart_provenance_rl.training.inference import (
    ImageMode,
    _inference_image,
)


def test_metrics_use_one_row_per_question(records: list[BenchmarkRecord]) -> None:
    oracle = [
        {
            "record_id": record.record_id,
            "prediction": attack_predictions(record)[0]["prediction"],
        }
        for record in records
    ]
    report = aggregate_predictions(records, oracle)
    assert report["answer_em"] == 1.0
    assert report["joint_exact_pass"] == 1.0
    with pytest.raises(ValueError, match="rollout rows"):
        aggregate_predictions(records, [*oracle, oracle[0]])
    with pytest.raises(ValueError, match="Missing predictions"):
        aggregate_predictions(records, oracle[:-1])


def test_cite_all_attack_exposes_reward_hacking(
    records: list[BenchmarkRecord],
) -> None:
    attacks = [attack for record in records for attack in attack_predictions(record)]
    assert reward_hacking_attack_success(records, attacks) == 1.0


def test_unknown_and_duplicate_references_fail_shared_schema(
    records: list[BenchmarkRecord],
) -> None:
    record = records[0]
    unknown = f'{{"answer":"{record.answer}","evidence":[{{"series":"Bogus","category":"Bogus"}}]}}'
    unknown_metrics = evaluate_prediction(record, unknown)
    assert unknown_metrics.answer_em
    assert not unknown_metrics.valid_schema
    assert not unknown_metrics.joint_exact
    reference = record.table.cells()[0]
    duplicate = (
        f'{{"answer":"{record.answer}","evidence":['
        f"{_reference_json(reference)},{_reference_json(reference)}]}}"
    )
    duplicate_metrics = evaluate_prediction(record, duplicate)
    assert duplicate_metrics.answer_em
    assert not duplicate_metrics.valid_schema
    assert not duplicate_metrics.joint_exact
    all_cells = record.table.cells()
    malformed_cite_all = (
        f'{{"answer":"{record.answer}","evidence":['
        + ",".join(_reference_json(reference) for reference in all_cells)
        + f",{_reference_json(all_cells[0])}]}}"
    )
    cite_all_metrics = evaluate_prediction(record, malformed_cite_all)
    assert cite_all_metrics.answer_em
    assert not cite_all_metrics.valid_schema
    assert not cite_all_metrics.cite_all


def test_question_only_image_shuffle_and_renderer_probes(
    dataset_root: Path,
    records: list[BenchmarkRecord],
) -> None:
    train = [record for record in records if record.split == "train"]
    composition = [record for record in records if record.split == "test_composition"]
    predictions = question_only_predictions(train, composition)
    assert set(predictions) == {record.record_id for record in composition}
    shuffle = image_shuffle_map(composition)
    assert all(shuffle[record.record_id] != record.image_path for record in composition)
    classifier = renderer_family_classifier(dataset_root, records)
    assert classifier["evaluated_images"] > 0
    assert 0.0 <= classifier["accuracy"] <= 1.0
    blank, blank_source = _inference_image(
        dataset_root,
        composition[0].record_id,
        composition[0].image_path,
        ImageMode.BLANK,
        {},
    )
    assert blank_source == "neutral_blank"
    assert blank.getbbox() == (0, 0, blank.width, blank.height)
    shuffled_image, shuffled_source = _inference_image(
        dataset_root,
        composition[0].record_id,
        composition[0].image_path,
        ImageMode.SHUFFLED,
        shuffle,
    )
    assert shuffled_source != composition[0].image_path
    assert shuffled_image.size == blank.size


def test_counterfactual_is_program_level(records: list[BenchmarkRecord]) -> None:
    diagnostic = counterfactual_diagnostic(records[0])
    assert 0.0 <= diagnostic.necessity <= 1.0
    assert 0.0 <= diagnostic.sufficiency <= 1.0
    assert diagnostic.to_dict()["provenance_set_index"] >= 0


def test_deterministic_json_repair_is_conservative() -> None:
    raw = 'prefix ```json\n{"answer":"1","evidence":[],}\n``` suffix'
    assert repair_output(raw) == '{"answer":"1","evidence":[]}'
    assert repair_output("no object") is None
    assert repair_output('{"answer":1,"evidence":[]}') is None


def test_paired_bootstrap_and_randomization_are_question_level() -> None:
    treatment = {f"q{index}": 1.0 for index in range(12)}
    control = {f"q{index}": 0.0 for index in range(12)}
    result = paired_question_test(
        treatment,
        control,
        seed=7,
        bootstrap_iterations=100,
        randomization_iterations=200,
    )
    assert result.question_count == 12
    assert result.mean_difference == 1.0
    assert result.ci_low == 1.0
    assert result.randomization_p_value < 0.05
    with pytest.raises(ValueError):
        paired_question_test({"q1": 1.0}, {"q2": 0.0}, seed=1)


def test_three_seed_aggregation_and_comparison() -> None:
    treatment = _seed_rows(1.0, "grpo_minimal_provenance")
    control = _seed_rows(0.0, "grpo_naive_recall")
    summary = aggregate_three_seeds(treatment, metric="joint_exact")
    assert summary["seed_count"] == 3
    assert summary["question_count"] == 4
    comparison = compare_methods(
        treatment,
        control,
        metric="joint_exact",
        seed=13,
    )
    assert comparison["paired_question_test"]["mean_difference"] == 1.0
    with pytest.raises(ValueError, match="exactly three seeds"):
        aggregate_three_seeds(treatment[:4], metric="joint_exact")
    mismatched = [dict(row) for row in control]
    mismatched[0]["seed"] = 101
    with pytest.raises(ValueError, match="identical question/seed"):
        compare_methods(treatment, mismatched, metric="joint_exact")
    same_method = [dict(row, method="grpo_minimal_provenance") for row in control]
    with pytest.raises(ValueError, match="methods must differ"):
        compare_methods(treatment, same_method, metric="joint_exact")
    wrong_split = [dict(row) for row in control]
    wrong_split[0]["split"] = "validation"
    with pytest.raises(ValueError, match="one split"):
        compare_methods(treatment, wrong_split, metric="joint_exact")
    missing_method = [dict(row) for row in control]
    for row in missing_method:
        row.pop("method")
    with pytest.raises(TypeError, match="non-empty string method"):
        compare_methods(treatment, missing_method, metric="joint_exact")
    with pytest.raises(ValueError, match="Duplicate question"):
        compare_methods(
            [*treatment, treatment[0]],
            [*control, control[0]],
            metric="joint_exact",
        )


def _seed_rows(
    score: float,
    method: str,
) -> list[dict[str, str | int | float]]:
    return [
        {
            "record_id": f"q{question}",
            "seed": seed,
            "joint_exact": score,
            "method": method,
            "split": "test_composition",
        }
        for seed in (17, 29, 43)
        for question in range(4)
    ]


def _reference_json(reference: EvidenceRef) -> str:
    return f'{{"series":"{reference.series}","category":"{reference.category}"}}'
