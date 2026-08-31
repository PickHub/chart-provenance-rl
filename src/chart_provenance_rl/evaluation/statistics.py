from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean, stdev


@dataclass(frozen=True)
class PairedTestResult:
    question_count: int
    mean_difference: float
    ci_low: float
    ci_high: float
    randomization_p_value: float
    bootstrap_iterations: int
    randomization_iterations: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "question_count": self.question_count,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "randomization_p_value": self.randomization_p_value,
            "bootstrap_iterations": self.bootstrap_iterations,
            "randomization_iterations": self.randomization_iterations,
        }


def paired_question_test(
    treatment: dict[str, float],
    control: dict[str, float],
    *,
    seed: int,
    bootstrap_iterations: int = 2_000,
    randomization_iterations: int = 5_000,
) -> PairedTestResult:
    if set(treatment) != set(control):
        raise ValueError("Paired tests require identical question IDs")
    if not treatment:
        raise ValueError("Paired tests require at least one question")
    question_ids = sorted(treatment)
    differences = [treatment[key] - control[key] for key in question_ids]
    observed = fmean(differences)
    rng = random.Random(seed)
    bootstrapped = sorted(
        fmean(rng.choice(differences) for _ in differences)
        for _iteration in range(bootstrap_iterations)
    )
    ci_low = _quantile(bootstrapped, 0.025)
    ci_high = _quantile(bootstrapped, 0.975)
    extreme = 0
    absolute_observed = abs(observed)
    for _iteration in range(randomization_iterations):
        randomized = fmean(
            difference if rng.random() < 0.5 else -difference for difference in differences
        )
        if abs(randomized) >= absolute_observed:
            extreme += 1
    p_value = (extreme + 1) / (randomization_iterations + 1)
    return PairedTestResult(
        question_count=len(question_ids),
        mean_difference=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        randomization_p_value=p_value,
        bootstrap_iterations=bootstrap_iterations,
        randomization_iterations=randomization_iterations,
    )


def aggregate_three_seeds(
    rows: list[dict[str, str | int | float]],
    *,
    metric: str,
) -> dict[str, float | int | dict[str, float]]:
    seed_scores: dict[int, list[float]] = {}
    question_seed_scores: dict[str, dict[int, float]] = {}
    for row in rows:
        seed = int(row["seed"])
        question_id = str(row["record_id"])
        score = float(row[metric])
        if seed in question_seed_scores.setdefault(question_id, {}):
            raise ValueError(f"Duplicate question {question_id} for seed {seed}")
        question_seed_scores[question_id][seed] = score
        seed_scores.setdefault(seed, []).append(score)
    if len(seed_scores) != 3:
        raise ValueError(f"Main claims require exactly three seeds, received {len(seed_scores)}")
    expected_seeds = set(seed_scores)
    if any(set(scores) != expected_seeds for scores in question_seed_scores.values()):
        raise ValueError("Every question must be present in all three seeds")
    per_seed = {str(seed): fmean(scores) for seed, scores in sorted(seed_scores.items())}
    seed_means = list(per_seed.values())
    per_question = [fmean(scores.values()) for scores in question_seed_scores.values()]
    return {
        "question_count": len(per_question),
        "seed_count": len(seed_scores),
        "mean": fmean(seed_means),
        "seed_standard_deviation": stdev(seed_means),
        "per_seed": per_seed,
    }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction
