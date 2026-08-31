from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from chart_provenance_rl.constants import MAX_EVIDENCE_REFS
from chart_provenance_rl.decimal_utils import canonical_answer
from chart_provenance_rl.evaluation.parsing import ParsedOutput, evidence_key, parse_output
from chart_provenance_rl.schema import BenchmarkRecord, EvidenceRef


class RewardArm(StrEnum):
    ANSWER_ONLY = "answer_only"
    NAIVE_RECALL = "naive_recall"
    MINIMAL_PROVENANCE = "minimal_provenance"


@dataclass(frozen=True)
class EvidenceScore:
    precision: float
    recall: float
    f1: float
    exact: bool
    predicted_size: int
    gold_size: int
    intersection_size: int


@dataclass(frozen=True)
class RewardBreakdown:
    arm: RewardArm
    reward: float
    answer_exact: bool
    evidence: EvidenceScore
    overcitation_count: int
    parsed: ParsedOutput


def score_reward(
    raw_output: str,
    expected_answer: str,
    valid_provenance_sets: tuple[tuple[EvidenceRef, ...], ...],
    arm: RewardArm,
    *,
    citation_cap: int = MAX_EVIDENCE_REFS,
    valid_evidence_refs: tuple[EvidenceRef, ...] | None = None,
) -> RewardBreakdown:
    parsed = parse_output(raw_output, citation_cap=citation_cap)
    empty_score = EvidenceScore(0.0, 0.0, 0.0, False, 0, 0, 0)
    if parsed.schema_valid and valid_evidence_refs is not None:
        visible_keys = {evidence_key(reference) for reference in valid_evidence_refs}
        if any(evidence_key(reference) not in visible_keys for reference in parsed.evidence):
            parsed = replace(
                parsed,
                schema_valid=False,
                error="unknown_evidence_reference",
            )
    if not parsed.schema_valid or parsed.answer is None:
        return RewardBreakdown(arm, 0.0, False, empty_score, 0, parsed)
    answer_exact = canonical_answer(parsed.answer) == canonical_answer(expected_answer)
    scoring_mode = "recall" if arm is RewardArm.NAIVE_RECALL else "f1"
    evidence = score_evidence(parsed.evidence, valid_provenance_sets, mode=scoring_mode)
    overcitation = max(0, evidence.predicted_size - evidence.gold_size)
    if arm is RewardArm.ANSWER_ONLY:
        reward = float(answer_exact)
    elif arm is RewardArm.NAIVE_RECALL:
        reward = float(answer_exact) * (0.5 + 0.5 * evidence.recall)
    elif not answer_exact:
        reward = 0.0
    else:
        reward = 0.5 + 0.5 * evidence.f1
    return RewardBreakdown(
        arm=arm,
        reward=reward,
        answer_exact=answer_exact,
        evidence=evidence,
        overcitation_count=overcitation,
        parsed=parsed,
    )


def score_record(
    record: BenchmarkRecord,
    raw_output: str,
    arm: RewardArm,
) -> RewardBreakdown:
    return score_reward(
        raw_output,
        record.answer,
        record.minimal_provenance_sets,
        arm,
        valid_evidence_refs=record.table.cells(),
    )


def score_evidence(
    predicted: tuple[EvidenceRef, ...],
    valid_provenance_sets: tuple[tuple[EvidenceRef, ...], ...],
    *,
    mode: str = "f1",
) -> EvidenceScore:
    if not valid_provenance_sets:
        raise ValueError("At least one provenance set is required")
    if mode not in {"f1", "recall"}:
        raise ValueError(f"Unknown evidence scoring mode: {mode}")
    candidates = [_score_single_set(predicted, gold) for gold in valid_provenance_sets]
    if mode == "recall":
        return max(
            candidates,
            key=lambda score: (score.recall, score.f1, score.precision, -score.gold_size),
        )
    return max(
        candidates,
        key=lambda score: (score.f1, score.exact, score.recall, -score.gold_size),
    )


def _score_single_set(
    predicted: tuple[EvidenceRef, ...],
    gold: tuple[EvidenceRef, ...],
) -> EvidenceScore:
    predicted_keys = {evidence_key(reference) for reference in predicted}
    gold_keys = {evidence_key(reference) for reference in gold}
    intersection = len(predicted_keys & gold_keys)
    precision = intersection / len(predicted_keys) if predicted_keys else 0.0
    recall = intersection / len(gold_keys) if gold_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvidenceScore(
        precision=precision,
        recall=recall,
        f1=f1,
        exact=predicted_keys == gold_keys,
        predicted_size=len(predicted_keys),
        gold_size=len(gold_keys),
        intersection_size=intersection,
    )
