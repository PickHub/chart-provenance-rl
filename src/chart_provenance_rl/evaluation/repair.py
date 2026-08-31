from __future__ import annotations

import re

from chart_provenance_rl.evaluation.parsing import parse_output
from chart_provenance_rl.serialization import canonical_json


def repair_output(raw_output: str) -> str | None:
    candidates = [
        raw_output.strip(),
        _strip_code_fence(raw_output),
        _extract_first_object(raw_output),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = re.sub(r",(\s*[}\]])", r"\1", candidate.strip())
        parsed = parse_output(normalized)
        if parsed.schema_valid and parsed.answer is not None:
            return canonical_json(
                {
                    "answer": parsed.answer,
                    "evidence": [reference.to_dict() for reference in parsed.evidence],
                }
            )
    return None


def _strip_code_fence(value: str) -> str | None:
    stripped = value.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1) if match else None


def _extract_first_object(value: str) -> str | None:
    start = value.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(value[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return None
