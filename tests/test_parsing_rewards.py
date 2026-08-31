import pytest

from chart_provenance_rl.constants import MAX_EVIDENCE_REFS
from chart_provenance_rl.evaluation.parsing import (
    completion_to_text,
    parse_output,
)
from chart_provenance_rl.evaluation.rewards import (
    RewardArm,
    score_evidence,
    score_reward,
)
from chart_provenance_rl.schema import EvidenceRef
from chart_provenance_rl.serialization import canonical_json

GOLD = ((EvidenceRef("Revenue", "Q3"),),)


def test_strict_parser_normalizes_evidence() -> None:
    parsed = parse_output('{"answer":"12.0","evidence":[{"series":" Revenue ","category":" Q3 "}]}')
    assert parsed.schema_valid
    assert parsed.evidence == (EvidenceRef("Revenue", "Q3"),)


def test_duplicate_evidence_is_deduplicated_and_rejected() -> None:
    parsed = parse_output(
        '{"answer":"12.0","evidence":['
        '{"series":"Revenue","category":"Q3"},'
        '{"series":"revenue","category":" Q3 "}]}'
    )
    assert parsed.evidence == (EvidenceRef("Revenue", "Q3"),)
    assert parsed.duplicate_evidence_count == 1
    assert not parsed.schema_valid
    assert parsed.error == "duplicate_evidence_reference"


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("not json", "malformed_json"),
        ('{"answer":"1","answer":"2","evidence":[]}', "duplicate_json_key"),
        ('{"answer":"1","evidence":[],"extra":1}', "unexpected_top_level_keys"),
        ('{"answer":1,"evidence":[]}', "answer_not_string"),
        ('{"answer":"","evidence":[]}', "empty_answer"),
        ('{"answer":"1","evidence":{}}', "evidence_not_list"),
        ('{"answer":"1","evidence":[1]}', "evidence_item_not_object"),
    ],
)
def test_strict_parser_rejects_invalid_output(raw: str, error: str) -> None:
    parsed = parse_output(raw)
    assert not parsed.schema_valid
    assert parsed.error == error


def test_citation_cap_applies_before_deduplication() -> None:
    evidence = [{"series": "Revenue", "category": "Q3"} for _ in range(MAX_EVIDENCE_REFS + 1)]
    parsed = parse_output(canonical_json({"answer": "12", "evidence": evidence}))
    assert parsed.citation_cap_exceeded
    assert not parsed.schema_valid


def test_reward_arms_isolate_recall_from_precision() -> None:
    all_cells = tuple(
        EvidenceRef(f"Series {index // 4}", f"Category {index % 4}") for index in range(8)
    )
    gold = ((all_cells[0],),)
    output = canonical_json(
        {
            "answer": "42",
            "evidence": [reference.to_dict() for reference in all_cells],
        }
    )
    naive = score_reward(output, "42", gold, RewardArm.NAIVE_RECALL)
    minimal = score_reward(output, "42", gold, RewardArm.MINIMAL_PROVENANCE)
    assert naive.reward == 1.0
    assert minimal.reward < 1.0
    assert minimal.evidence.precision == 0.125
    assert minimal.overcitation_count == 7


@pytest.mark.parametrize(
    "arm",
    [RewardArm.NAIVE_RECALL, RewardArm.MINIMAL_PROVENANCE],
)
def test_wrong_answer_gates_all_evidence_credit(arm: RewardArm) -> None:
    output = '{"answer":"wrong","evidence":[{"series":"Revenue","category":"Q3"}]}'
    assert score_reward(output, "42", GOLD, arm).reward == 0.0


def test_malformed_and_cap_violations_receive_zero() -> None:
    assert score_reward("{", "42", GOLD, RewardArm.ANSWER_ONLY).reward == 0.0
    refs = [
        {"series": f"S{index}", "category": f"C{index}"} for index in range(MAX_EVIDENCE_REFS + 1)
    ]
    raw = canonical_json({"answer": "42", "evidence": refs})
    assert score_reward(raw, "42", GOLD, RewardArm.NAIVE_RECALL).reward == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        (
            '{"answer":"42","evidence":['
            '{"series":"Revenue","category":"Q3"},'
            '{"series":"Revenue","category":"Q3"}]}'
        ),
        '{"answer":"42","evidence":[{"series":"Unknown","category":"Q3"}]}',
        canonical_json(
            {
                "answer": "42",
                "evidence": [
                    {"series": f"S{index}", "category": f"C{index}"}
                    for index in range(MAX_EVIDENCE_REFS + 1)
                ],
            }
        ),
        '{"answer":"42","evidence":[1]}',
        '{"answer":"42","evidence":{}}',
        '{"answer":"42","evidence":[],"unexpected":true}',
    ],
)
def test_answer_exact_is_independent_of_invalid_evidence(raw: str) -> None:
    breakdown = score_reward(
        raw,
        "42",
        GOLD,
        RewardArm.ANSWER_ONLY,
        valid_evidence_refs=(EvidenceRef("Revenue", "Q3"),),
    )
    assert breakdown.answer_exact
    assert breakdown.reward == 0.0
    assert not breakdown.parsed.schema_valid


def test_malformed_json_has_no_trustworthy_answer() -> None:
    breakdown = score_reward(
        '{"answer":"42","evidence":',
        "42",
        GOLD,
        RewardArm.ANSWER_ONLY,
    )
    assert not breakdown.answer_exact
    assert breakdown.reward == 0.0


def test_extreme_numeric_answer_is_a_safe_non_match() -> None:
    breakdown = score_reward(
        '{"answer":"1e1000000","evidence":[]}',
        "42",
        GOLD,
        RewardArm.ANSWER_ONLY,
    )
    assert not breakdown.answer_exact
    assert breakdown.reward == 0.0


def test_best_valid_provenance_set_is_used() -> None:
    alternatives = (
        (EvidenceRef("Revenue", "Q1"),),
        (EvidenceRef("Revenue", "Q2"),),
    )
    predicted = (EvidenceRef("revenue", "q2"),)
    score = score_evidence(predicted, alternatives)
    assert score.exact
    assert score.f1 == 1.0


def test_completion_text_supports_conversational_shapes() -> None:
    assert completion_to_text("raw") == "raw"
    assert completion_to_text([{"role": "assistant", "content": "answer"}]) == "answer"
    assert (
        completion_to_text(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                }
            ]
        )
        == "answer"
    )
    with pytest.raises(TypeError):
        completion_to_text([])
