from __future__ import annotations

from typing import Any

from chart_provenance_rl.constants import MAX_EVIDENCE_REFS
from chart_provenance_rl.schema import BenchmarkRecord, EvidenceRef
from chart_provenance_rl.serialization import canonical_json


def attack_predictions(record: BenchmarkRecord) -> list[dict[str, Any]]:
    gold = record.minimal_provenance_sets[0]
    all_cells = record.table.cells()
    wrong = tuple(cell for cell in all_cells if cell not in gold)
    wrong_reference = wrong[0] if wrong else EvidenceRef("Unknown series", "Unknown category")
    return [
        _prediction(record, "oracle_minimal", record.answer, gold),
        _prediction(record, "answer_only", record.answer, ()),
        _prediction(record, "cite_all", record.answer, all_cells),
        _prediction(record, "answer_correct_citation_wrong", record.answer, (wrong_reference,)),
        {
            "record_id": record.record_id,
            "attack": "malformed_json",
            "prediction": '{"answer":',
        },
        _prediction(
            record,
            "citation_cap_violation",
            record.answer,
            tuple(
                EvidenceRef(f"Series {index}", f"Category {index}")
                for index in range(MAX_EVIDENCE_REFS + 1)
            ),
        ),
    ]


def _prediction(
    record: BenchmarkRecord,
    attack: str,
    answer: str,
    evidence: tuple[EvidenceRef, ...],
) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "attack": attack,
        "prediction": canonical_json(
            {
                "answer": answer,
                "evidence": [reference.to_dict() for reference in evidence],
            }
        ),
    }
