from __future__ import annotations

from typing import Any

from chart_provenance_rl.constants import SYSTEM_PROMPT
from chart_provenance_rl.schema import EvidenceRef
from chart_provenance_rl.serialization import canonical_json


def build_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}"


def training_prompt(question: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def training_completion(answer: str, evidence: tuple[EvidenceRef, ...]) -> list[dict[str, Any]]:
    content = canonical_json(
        {
            "answer": answer,
            "evidence": [item.to_dict() for item in evidence],
        }
    )
    return [{"role": "assistant", "content": content}]
