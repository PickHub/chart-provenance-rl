from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any

from chart_provenance_rl.evaluation.statistics import (
    aggregate_three_seeds,
    paired_question_test,
)


def compare_methods(
    treatment_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    *,
    metric: str,
    seed: int = 202_701,
) -> dict[str, Any]:
    if _row_keys(treatment_rows) != _row_keys(control_rows):
        raise ValueError("Treatment and control require identical question/seed pairs")
    treatment_method = _constant_string(treatment_rows, "method")
    control_method = _constant_string(control_rows, "method")
    if treatment_method == control_method:
        raise ValueError("Treatment and control methods must differ")
    treatment_split = _constant_string(treatment_rows, "split")
    control_split = _constant_string(control_rows, "split")
    if treatment_split != control_split:
        raise ValueError("Treatment and control splits must match")
    treatment = _question_means(treatment_rows, metric)
    control = _question_means(control_rows, metric)
    return {
        "metric": metric,
        "treatment_method": treatment_method,
        "control_method": control_method,
        "split": treatment_split,
        "treatment": aggregate_three_seeds(treatment_rows, metric=metric),
        "control": aggregate_three_seeds(control_rows, metric=metric),
        "paired_question_test": paired_question_test(
            treatment,
            control,
            seed=seed,
        ).to_dict(),
    }


def _row_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["record_id"]), int(row["seed"])) for row in rows}


def _constant_string(rows: list[dict[str, Any]], field: str) -> str:
    values = {row.get(field) for row in rows}
    if len(values) != 1:
        raise ValueError(f"Analysis rows require one {field}")
    value = values.pop()
    if not isinstance(value, str) or not value:
        raise TypeError(f"Analysis rows require a non-empty string {field}")
    return value


def _question_means(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        record_id = str(row["record_id"])
        run_seed = int(row["seed"])
        if run_seed in grouped[record_id]:
            raise ValueError(f"Duplicate question {record_id} for seed {run_seed}")
        grouped[record_id][run_seed] = float(row[metric])
    seeds = {seed for values in grouped.values() for seed in values}
    if len(seeds) != 3:
        raise ValueError("Main comparisons require three seeds")
    if any(set(values) != seeds for values in grouped.values()):
        raise ValueError("Every question must be present in every seed")
    return {record_id: fmean(seed_scores.values()) for record_id, seed_scores in grouped.items()}
