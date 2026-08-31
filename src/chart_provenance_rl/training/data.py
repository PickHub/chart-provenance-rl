from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from chart_provenance_rl.audits import load_dataset_records
from chart_provenance_rl.prompts import training_completion, training_prompt
from chart_provenance_rl.schema import BenchmarkRecord, EvidenceRef
from chart_provenance_rl.serialization import canonical_json


def load_sft_dataset(root: Path, split: str = "train") -> Any:
    from datasets import Dataset

    records = _records(root, split)
    examples = []
    for record in records:
        provenance = record.minimal_provenance_sets[
            record.generation_index % len(record.minimal_provenance_sets)
        ]
        examples.append(
            {
                "record_id": record.record_id,
                "prompt": training_prompt(record.question),
                "completion": training_completion(record.answer, provenance),
                "image": _load_image(root / record.image_path),
            }
        )
    return Dataset.from_list(examples)


def load_grpo_dataset(root: Path, split: str = "train") -> Any:
    from datasets import Dataset

    records = _records(root, split)
    return Dataset.from_list(
        [
            {
                "record_id": record.record_id,
                "prompt": training_prompt(record.question),
                "image": _load_image(root / record.image_path),
                "expected_answer": record.answer,
                "minimal_provenance_json": canonical_json(
                    [
                        [reference.to_dict() for reference in provenance]
                        for provenance in record.minimal_provenance_sets
                    ]
                ),
                "valid_evidence_json": canonical_json(
                    [reference.to_dict() for reference in record.table.cells()]
                ),
            }
            for record in records
        ]
    )


def parse_provenance_sets(
    serialized: str,
) -> tuple[tuple[EvidenceRef, ...], ...]:
    import json

    value = json.loads(serialized)
    if not isinstance(value, list):
        raise TypeError("Serialized provenance sets must be a list")
    provenance_sets: list[tuple[EvidenceRef, ...]] = []
    for provenance in value:
        if not isinstance(provenance, list):
            raise TypeError("Each provenance set must be a list")
        references: list[EvidenceRef] = []
        for reference in provenance:
            if not isinstance(reference, dict):
                raise TypeError("Provenance references must be objects")
            references.append(EvidenceRef.from_dict(reference))
        provenance_sets.append(tuple(references))
    if not provenance_sets:
        raise ValueError("At least one provenance set is required")
    return tuple(provenance_sets)


def parse_evidence_refs(serialized: str) -> tuple[EvidenceRef, ...]:
    import json

    value = json.loads(serialized)
    if not isinstance(value, list):
        raise TypeError("Serialized evidence references must be a list")
    references = []
    for reference in value:
        if not isinstance(reference, dict):
            raise TypeError("Evidence references must be objects")
        references.append(EvidenceRef.from_dict(reference))
    if not references:
        raise ValueError("Visible evidence universe must not be empty")
    return tuple(references)


def _records(root: Path, split: str) -> list[BenchmarkRecord]:
    records = load_dataset_records(root)
    try:
        return records[split]
    except KeyError as error:
        raise KeyError(f"Unknown dataset split: {split}") from error


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
