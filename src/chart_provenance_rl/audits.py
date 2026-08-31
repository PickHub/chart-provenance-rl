from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from chart_provenance_rl.programs import execute_program
from chart_provenance_rl.schema import BenchmarkRecord, ProgramKind
from chart_provenance_rl.serialization import (
    canonical_json,
    load_json,
    load_jsonl,
    sha256_bytes,
    sha256_json,
)


def load_dataset_records(root: Path) -> dict[str, list[BenchmarkRecord]]:
    dataset = load_json(root / "dataset.json")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("splits"), dict):
        raise TypeError("dataset.json is missing split metadata")
    records: dict[str, list[BenchmarkRecord]] = {}
    for split_name, split_metadata in dataset["splits"].items():
        if not isinstance(split_name, str) or not isinstance(split_metadata, dict):
            raise TypeError("Invalid split metadata")
        manifest_path = split_metadata.get("manifest")
        if not isinstance(manifest_path, str):
            raise TypeError("Split manifest path must be a string")
        records[split_name] = [
            BenchmarkRecord.from_dict(item) for item in load_jsonl(root / manifest_path)
        ]
    return records


def audit_dataset(root: Path) -> dict[str, Any]:
    dataset = load_json(root / "dataset.json")
    if not isinstance(dataset, dict):
        raise TypeError("dataset.json must be an object")
    records_by_split = load_dataset_records(root)
    errors: list[str] = []
    warnings: list[str] = []
    hashes_by_split: dict[str, dict[str, set[str]]] = {}
    answer_leaks = 0

    for split, records in records_by_split.items():
        split_metadata = dataset["splits"][split]
        manifest_path = root / split_metadata["manifest"]
        if sha256_bytes(manifest_path.read_bytes()) != split_metadata["manifest_sha256"]:
            errors.append(f"{split}: manifest hash mismatch")
        if len(records) != split_metadata["count"]:
            errors.append(f"{split}: manifest count mismatch")
        hashes_by_split[split] = {
            "record": set(),
            "image": set(),
            "table": set(),
        }
        for record in records:
            _audit_record(root, record, errors)
            hashes_by_split[split]["record"].add(record.record_id)
            hashes_by_split[split]["image"].add(record.image_sha256)
            hashes_by_split[split]["table"].add(record.table_sha256)
            if _answer_appears_in_prompt(record):
                answer_leaks += 1
                errors.append(f"{record.record_id}: expected answer appears in prompt")

    _audit_cross_split_hashes(hashes_by_split, errors)
    _audit_split_design(records_by_split, errors)
    duplicate_records = sum(
        len(records) - len({record.record_id for record in records})
        for records in records_by_split.values()
    )
    if duplicate_records:
        errors.append(f"Found {duplicate_records} duplicate records within splits")
    exact_questions: dict[str, set[str]] = defaultdict(set)
    for split, records in records_by_split.items():
        for record in records:
            exact_questions[record.question].add(split)
    overlapping_questions = sum(1 for splits in exact_questions.values() if len(splits) > 1)
    if overlapping_questions:
        warnings.append(
            f"{overlapping_questions} exact question strings cross splits; "
            "values and images remain disjoint"
        )
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "split_counts": {split: len(records) for split, records in records_by_split.items()},
        "answer_leak_count": answer_leaks,
        "cross_split_exact_question_count": overlapping_questions,
    }


def _audit_record(root: Path, record: BenchmarkRecord, errors: list[str]) -> None:
    prefix = record.record_id
    if record.record_id != sha256_json(record.to_dict(include_record_id=False)):
        errors.append(f"{prefix}: record hash mismatch")
    if record.table_sha256 != sha256_json(record.table.to_dict()):
        errors.append(f"{prefix}: table hash mismatch")
    expected_image_path = f"images/{record.image_sha256}.png"
    if record.image_path != expected_image_path:
        errors.append(f"{prefix}: image filename is not content-addressed")
    image_path = root / record.image_path
    if not image_path.is_file():
        errors.append(f"{prefix}: image is missing")
    elif sha256_bytes(image_path.read_bytes()) != record.image_sha256:
        errors.append(f"{prefix}: image hash mismatch")
    sidecar_path = root / "records" / f"{record.record_id}.json"
    if not sidecar_path.is_file():
        errors.append(f"{prefix}: sidecar is missing")
    else:
        sidecar = load_json(sidecar_path)
        if canonical_json(sidecar) != canonical_json(record.to_dict()):
            errors.append(f"{prefix}: sidecar differs from manifest")
    result = execute_program(record.table, record.program)
    if result.answer != record.answer:
        errors.append(f"{prefix}: executable answer mismatch")
    if result.minimal_provenance_sets != record.minimal_provenance_sets:
        errors.append(f"{prefix}: executable provenance mismatch")
    if record.program.kind is ProgramKind.LOOKUP and record.program.categories:
        lookup_values = {
            record.table.get(record.program.series, category)
            for category in record.program.categories
        }
        if len(lookup_values) != 1:
            errors.append(f"{prefix}: alternative lookup witnesses disagree")
    visible = set(record.table.cells())
    if any(
        reference not in visible
        for provenance in record.minimal_provenance_sets
        for reference in provenance
    ):
        errors.append(f"{prefix}: provenance references a non-visible entity")
    forbidden = ("record_id", "minimal_provenance", "table_sha256", "image_sha256", "program")
    if any(token in record.prompt.casefold() for token in forbidden):
        errors.append(f"{prefix}: prompt contains hidden metadata terminology")
    if re.search(r"\b[0-9a-f]{64}\b", record.prompt.casefold()):
        errors.append(f"{prefix}: prompt contains a content hash")


def _audit_cross_split_hashes(
    hashes_by_split: dict[str, dict[str, set[str]]],
    errors: list[str],
) -> None:
    split_names = sorted(hashes_by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            for hash_kind in ("record", "image", "table"):
                overlap = hashes_by_split[left][hash_kind] & hashes_by_split[right][hash_kind]
                if overlap:
                    errors.append(f"{left}/{right}: {len(overlap)} shared {hash_kind} hashes")


def _audit_split_design(
    records_by_split: dict[str, list[BenchmarkRecord]],
    errors: list[str],
) -> None:
    train = records_by_split.get("train", [])
    composition = records_by_split.get("test_composition", [])
    train_programs = {record.program.kind for record in train}
    composition_programs = {record.program.kind for record in composition}
    required_compositions = {ProgramKind.DIFFERENCE, ProgramKind.SUM_SELECTED}
    if train_programs & required_compositions:
        errors.append("Training split contains held-out program compositions")
    if composition_programs != required_compositions:
        errors.append("Primary OOD split does not cover both held-out compositions")
    train_renderers = {record.renderer_family for record in train}
    heldout_renderers = {
        record.renderer_family for record in records_by_split.get("stress_renderer_holdout", [])
    }
    if train_renderers & heldout_renderers:
        errors.append("Renderer holdout overlaps training renderer families")
    train_seeds = {(record.grammar_seed, record.config_seed) for record in train}
    heldout_seeds = {
        (record.grammar_seed, record.config_seed)
        for split, records in records_by_split.items()
        if split != "train"
        for record in records
    }
    if train_seeds & heldout_seeds:
        errors.append("Training and held-out grammar/config seeds overlap")
    duplicate_records = records_by_split.get("stress_duplicate_decoy", [])
    if not any(len(record.minimal_provenance_sets) > 1 for record in duplicate_records):
        errors.append("Duplicate stress split lacks alternative valid provenance sets")


def _answer_appears_in_prompt(record: BenchmarkRecord) -> bool:
    escaped = re.escape(record.answer.casefold())
    return re.search(rf"(?<![\w.]){escaped}(?![\w.])", record.question.casefold()) is not None
