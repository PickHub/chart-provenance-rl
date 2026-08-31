from __future__ import annotations

import re
from collections import Counter, defaultdict
from math import log2
from pathlib import Path
from typing import Any

from PIL import Image

from chart_provenance_rl.constants import DEFAULT_IMAGE_SIZE
from chart_provenance_rl.decimal_utils import canonical_decimal
from chart_provenance_rl.generation.renderer import (
    MIN_TEXT_CONTRAST,
    boxes_overlap,
    contrast_ratio,
)
from chart_provenance_rl.programs import execute_program
from chart_provenance_rl.schema import (
    BenchmarkRecord,
    ProgramKind,
    TextRole,
)
from chart_provenance_rl.serialization import (
    canonical_json,
    load_json,
    load_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_json,
)


def load_dataset_records(root: Path) -> dict[str, list[BenchmarkRecord]]:
    report = audit_dataset(root)
    if report["status"] != "pass":
        raise ValueError(f"Dataset integrity verification failed: {report['errors']}")
    return _load_dataset_records_unchecked(root)


def _load_dataset_records_unchecked(
    root: Path,
) -> dict[str, list[BenchmarkRecord]]:
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
    records_by_split = _load_dataset_records_unchecked(root)
    errors: list[str] = []
    warnings: list[str] = []
    hashes_by_split: dict[str, dict[str, set[str]]] = {}
    within_split_duplicates = {"record": 0, "image": 0, "table": 0}
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
        for hash_kind, attribute in (
            ("record", "record_id"),
            ("image", "image_sha256"),
            ("table", "table_sha256"),
        ):
            counts = Counter(getattr(record, attribute) for record in records)
            duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
            within_split_duplicates[hash_kind] += duplicate_count
            if duplicate_count:
                errors.append(
                    f"{split}: {duplicate_count} unexpected duplicate "
                    f"{hash_kind} hashes within split"
                )

    cross_split_hash_overlap_count = _audit_cross_split_hashes(
        hashes_by_split,
        errors,
    )
    factor_contingency = _audit_split_design(records_by_split, errors)
    program_answer_distributions = _audit_answer_distributions(
        records_by_split,
        errors,
    )
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
        "cross_split_hash_overlap_count": cross_split_hash_overlap_count,
        "within_split_duplicate_hash_count": within_split_duplicates,
        "factor_contingency": factor_contingency,
        "program_answer_distributions": program_answer_distributions,
        "dataset_manifest_sha256": sha256_file(root / "dataset.json"),
        "split_record_ids": {
            split: sorted(record.record_id for record in records)
            for split, records in records_by_split.items()
        },
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
    else:
        with Image.open(image_path) as image:
            _audit_render_visibility(record, image, errors)
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
) -> int:
    overlap_count = 0
    split_names = sorted(hashes_by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            for hash_kind in ("record", "image", "table"):
                overlap = hashes_by_split[left][hash_kind] & hashes_by_split[right][hash_kind]
                if overlap:
                    overlap_count += len(overlap)
                    errors.append(f"{left}/{right}: {len(overlap)} shared {hash_kind} hashes")
    return overlap_count


def _audit_split_design(
    records_by_split: dict[str, list[BenchmarkRecord]],
    errors: list[str],
) -> dict[str, Any]:
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
    seed_owners: dict[tuple[int, int], str] = {}
    for split, records in records_by_split.items():
        for record in records:
            seed_pair = (record.grammar_seed, record.config_seed)
            prior_owner = seed_owners.setdefault(seed_pair, split)
            if prior_owner != split:
                errors.append(f"{prior_owner}/{split}: grammar/config seed pair overlaps")
    duplicate_records = records_by_split.get("stress_duplicate_decoy", [])
    if not any(len(record.minimal_provenance_sets) > 1 for record in duplicate_records):
        errors.append("Duplicate stress split lacks alternative valid provenance sets")
    contingency = {
        "train": _factor_contingency(train),
        "test_composition": _factor_contingency(composition),
    }
    _audit_factor_coverage(train, "train", errors)
    _audit_factor_coverage(composition, "test_composition", errors)
    return contingency


def _audit_render_visibility(
    record: BenchmarkRecord,
    image: Image.Image,
    errors: list[str],
) -> None:
    prefix = record.record_id
    rgb_image = image.convert("RGB")
    image_size = rgb_image.size
    width, height = image_size
    if image_size != DEFAULT_IMAGE_SIZE:
        errors.append(f"{prefix}: unexpected image size {image_size}")
    for element in record.render_text:
        left, top, right, bottom = element.bbox
        if left < 0 or top < 0 or right > width or bottom > height:
            errors.append(f"{prefix}: clipped {element.role.value} label {element.text!r}")
        actual_contrast = contrast_ratio(element.foreground, element.background)
        if actual_contrast != element.contrast_ratio:
            errors.append(
                f"{prefix}: stored contrast differs for {element.role.value} label {element.text!r}"
            )
        if actual_contrast < MIN_TEXT_CONTRAST:
            errors.append(f"{prefix}: low-contrast {element.role.value} label {element.text!r}")
        background_rgb = _hex_rgb(element.background)
        corners = (
            (left, top),
            (right - 1, top),
            (left, bottom - 1),
            (right - 1, bottom - 1),
        )
        if any(rgb_image.getpixel(point) != background_rgb for point in corners):
            errors.append(
                f"{prefix}: rendered pixels disagree with {element.role.value} "
                f"background metadata for {element.text!r}"
            )
        foreground_rgb = _hex_rgb(element.foreground)
        crop_pixels = rgb_image.crop(element.bbox).get_flattened_data()
        if min(_rgb_distance(pixel, foreground_rgb) for pixel in crop_pixels) > 96:
            errors.append(
                f"{prefix}: rendered pixels lack {element.role.value} "
                f"foreground for {element.text!r}"
            )
    for index, left_element in enumerate(record.render_text):
        for right_element in record.render_text[index + 1 :]:
            if boxes_overlap(left_element.bbox, right_element.bbox):
                errors.append(
                    f"{prefix}: overlapping {left_element.role.value}/"
                    f"{right_element.role.value} labels {left_element.text!r} "
                    f"and {right_element.text!r}"
                )
    value_elements = [element for element in record.render_text if element.role is TextRole.VALUE]
    value_references = {
        element.evidence for element in value_elements if element.evidence is not None
    }
    if len(value_elements) != len(record.table.cells()):
        errors.append(f"{prefix}: expected one visible value label per semantic cell")
    if value_references != set(record.table.cells()):
        errors.append(f"{prefix}: visible value labels do not cover all semantic cells")
    for element in value_elements:
        if element.evidence is None:
            continue
        expected = canonical_decimal(
            record.table.get(element.evidence.series, element.evidence.category)
        )
        if element.text != expected:
            errors.append(f"{prefix}: visible value text mismatch for {element.evidence}")
    category_labels = {
        element.text for element in record.render_text if element.role is TextRole.CATEGORY
    }
    legend_labels = {
        element.text for element in record.render_text if element.role is TextRole.LEGEND
    }
    if category_labels != set(record.table.categories):
        errors.append(f"{prefix}: visible category labels do not match the table")
    if legend_labels != set(record.table.series):
        errors.append(f"{prefix}: visible legend labels do not match the table")


def _factor_contingency(
    records: list[BenchmarkRecord],
) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for factor in _factor_names():
        factor_counts: dict[str, dict[str, int]] = defaultdict(dict)
        for record in records:
            value = _factor_value(record, factor)
            program = record.program.kind.value
            factor_counts[value][program] = factor_counts[value].get(program, 0) + 1
        result[factor] = dict(sorted(factor_counts.items()))
    return result


def _audit_factor_coverage(
    records: list[BenchmarkRecord],
    split: str,
    errors: list[str],
) -> None:
    programs = {record.program.kind for record in records}
    if len(programs) < 2:
        return
    for factor in _factor_names():
        factor_levels = {_factor_value(record, factor) for record in records}
        if not factor_levels:
            continue
        by_program = {
            program: {
                _factor_value(record, factor)
                for record in records
                if record.program.kind is program
            }
            for program in programs
        }
        minimum_program_count = min(
            sum(record.program.kind is program for record in records) for program in programs
        )
        if minimum_program_count >= len(factor_levels):
            for program, observed in by_program.items():
                if observed != factor_levels:
                    errors.append(f"{split}: {program.value} does not span all {factor} levels")
        if len(records) >= len(programs) * len(factor_levels):
            for level in factor_levels:
                programs_at_level = {
                    record.program.kind
                    for record in records
                    if _factor_value(record, factor) == level
                }
                if programs_at_level != programs:
                    errors.append(f"{split}: {factor}={level} predicts program kind")


def _audit_answer_distributions(
    records_by_split: dict[str, list[BenchmarkRecord]],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for split, records in records_by_split.items():
        split_report: dict[str, Any] = {}
        for program in sorted(
            {record.program.kind for record in records},
            key=lambda kind: kind.value,
        ):
            answers = Counter(record.answer for record in records if record.program.kind is program)
            total = sum(answers.values())
            entropy = -sum((count / total) * log2(count / total) for count in answers.values())
            split_report[program.value] = {
                "count": total,
                "unique_answer_count": len(answers),
                "entropy_bits": round(entropy, 6),
                "answer_counts": dict(sorted(answers.items())),
            }
            required_unique = 3 if program is ProgramKind.COUNT and total >= 3 else 2
            if total >= required_unique and len(answers) < required_unique:
                errors.append(
                    f"{split}: {program.value} answer distribution collapsed "
                    f"to {len(answers)} unique values"
                )
            if program is ProgramKind.COUNT and total >= 3:
                count_values = [answers.get(answer, 0) for answer in ("1", "2", "3")]
                if set(answers) != {"1", "2", "3"} or max(count_values) - min(count_values) > 1:
                    errors.append(f"{split}: count answers are not balanced across 1, 2, and 3")
        report[split] = split_report
    return report


def _factor_names() -> tuple[str, ...]:
    return (
        "chart_type",
        "renderer_family",
        "category_family",
        "series_family",
        "target_series_index",
        "title",
    )


def _factor_value(record: BenchmarkRecord, factor: str) -> str:
    values = {
        "chart_type": record.chart_type.value,
        "renderer_family": record.renderer_family,
        "category_family": record.category_family,
        "series_family": record.series_family,
        "target_series_index": str(record.target_series_index),
        "title": record.table.title,
    }
    return values[factor]


def _hex_rgb(color: str) -> tuple[int, int, int]:
    if not color.startswith("#") or len(color) != 7:
        raise ValueError(f"Expected RGB hex color, received {color!r}")
    return (
        int(color[1:3], 16),
        int(color[3:5], 16),
        int(color[5:7], 16),
    )


def _rgb_distance(
    left: object,
    right: tuple[int, int, int],
) -> int:
    if (
        not isinstance(left, tuple)
        or len(left) != 3
        or not all(isinstance(channel, int) for channel in left)
    ):
        raise TypeError(f"Expected an RGB pixel, received {left!r}")
    return sum(
        abs(left_channel - right_channel)
        for left_channel, right_channel in zip(left, right, strict=True)
    )


def _answer_appears_in_prompt(record: BenchmarkRecord) -> bool:
    escaped = re.escape(record.answer.casefold())
    return re.search(rf"(?<![\w.]){escaped}(?![\w.])", record.prompt.casefold()) is not None
