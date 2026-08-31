from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from chart_provenance_rl.constants import MAX_EVIDENCE_REFS
from chart_provenance_rl.schema import EvidenceRef


@dataclass(frozen=True)
class ParsedOutput:
    json_valid: bool
    schema_valid: bool
    answer: str | None
    evidence: tuple[EvidenceRef, ...]
    raw_evidence_count: int
    duplicate_evidence_count: int
    citation_cap_exceeded: bool
    error: str | None = None


class DuplicateJsonKeyError(ValueError):
    pass


def parse_output(raw_output: str, *, citation_cap: int = MAX_EVIDENCE_REFS) -> ParsedOutput:
    try:
        value = json.loads(raw_output, object_pairs_hook=_unique_object)
    except json.JSONDecodeError:
        return _invalid("malformed_json", json_valid=False)
    except DuplicateJsonKeyError:
        return _invalid("duplicate_json_key", json_valid=False)
    if not isinstance(value, dict):
        return _invalid("root_not_object")
    answer = _candidate_answer(value)
    if set(value) != {"answer", "evidence"}:
        return _invalid("unexpected_top_level_keys", answer=answer)
    answer_value = value["answer"]
    evidence = value["evidence"]
    if not isinstance(answer_value, str):
        return _invalid("answer_not_string")
    if not answer_value.strip():
        return _invalid("empty_answer")
    answer = answer_value
    if not isinstance(evidence, list):
        return _invalid("evidence_not_list", answer=answer)
    raw_count = len(evidence)
    if raw_count > citation_cap:
        return ParsedOutput(
            json_valid=True,
            schema_valid=False,
            answer=answer,
            evidence=(),
            raw_evidence_count=raw_count,
            duplicate_evidence_count=0,
            citation_cap_exceeded=True,
            error="citation_cap_exceeded",
        )
    normalized: list[EvidenceRef] = []
    seen: set[tuple[str, str]] = set()
    duplicate_count = 0
    for raw_reference in evidence:
        if not isinstance(raw_reference, dict):
            return _invalid(
                "evidence_item_not_object",
                raw_count=raw_count,
                answer=answer,
            )
        try:
            reference = EvidenceRef.from_dict(raw_reference)
        except (TypeError, ValueError):
            return _invalid(
                "invalid_evidence_item",
                raw_count=raw_count,
                answer=answer,
            )
        normalized_reference = EvidenceRef(
            series=_normalize_label(reference.series),
            category=_normalize_label(reference.category),
        )
        key = evidence_key(normalized_reference)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        normalized.append(normalized_reference)
    return ParsedOutput(
        json_valid=True,
        schema_valid=duplicate_count == 0,
        answer=answer,
        evidence=tuple(normalized),
        raw_evidence_count=raw_count,
        duplicate_evidence_count=duplicate_count,
        citation_cap_exceeded=False,
        error="duplicate_evidence_reference" if duplicate_count else None,
    )


def evidence_key(reference: EvidenceRef) -> tuple[str, str]:
    return (reference.series.casefold(), reference.category.casefold())


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list) or not completion:
        raise TypeError("Completion must be text or a non-empty conversational completion")
    final_message = completion[-1]
    if not isinstance(final_message, dict):
        raise TypeError("Conversational completion messages must be objects")
    content = final_message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "".join(text_parts)
    raise TypeError("Assistant completion does not contain text")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _normalize_label(value: str) -> str:
    return " ".join(value.strip().split())


def _candidate_answer(value: dict[str, Any]) -> str | None:
    answer = value.get("answer")
    return answer if isinstance(answer, str) and answer.strip() else None


def _invalid(
    error: str,
    *,
    json_valid: bool = True,
    raw_count: int = 0,
    answer: str | None = None,
) -> ParsedOutput:
    return ParsedOutput(
        json_valid=json_valid,
        schema_valid=False,
        answer=answer,
        evidence=(),
        raw_evidence_count=raw_count,
        duplicate_evidence_count=0,
        citation_cap_exceeded=False,
        error=error,
    )
