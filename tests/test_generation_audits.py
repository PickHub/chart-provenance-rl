from pathlib import Path

import pytest
from PIL import Image

from chart_provenance_rl.audits import audit_dataset, load_dataset_records
from chart_provenance_rl.constants import DEFAULT_IMAGE_SIZE
from chart_provenance_rl.generation.config import smoke_config
from chart_provenance_rl.generation.generator import generate_dataset
from chart_provenance_rl.generation.renderer import (
    MIN_TEXT_CONTRAST,
    boxes_overlap,
)
from chart_provenance_rl.schema import ProgramKind, TextRole
from chart_provenance_rl.serialization import (
    load_json,
    load_jsonl,
    sha256_bytes,
    write_json,
    write_jsonl,
)


def test_generated_dataset_passes_integrity_and_leakage_audits(
    dataset_root: Path,
) -> None:
    report = audit_dataset(dataset_root)
    assert report["status"] == "pass"
    assert report["answer_leak_count"] == 0
    assert load_json(dataset_root / "dataset.json")["license"] == "CC0-1.0"


def test_generation_is_byte_deterministic(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    generate_dataset(left, smoke_config())
    generate_dataset(right, smoke_config())
    assert (left / "dataset.json").read_bytes() == (right / "dataset.json").read_bytes()
    left_manifests = sorted(path.name for path in (left / "manifests").iterdir())
    for name in left_manifests:
        assert (left / "manifests" / name).read_bytes() == (right / "manifests" / name).read_bytes()


def test_split_design_and_visible_evidence_universe(dataset_root: Path) -> None:
    records = load_dataset_records(dataset_root)
    assert {record.program.kind for record in records["test_composition"]} == {
        ProgramKind.DIFFERENCE,
        ProgramKind.SUM_SELECTED,
    }
    assert not {
        ProgramKind.DIFFERENCE,
        ProgramKind.SUM_SELECTED,
    } & {record.program.kind for record in records["train"]}
    assert all(len(record.table.cells()) == 8 for record in records["train"])
    assert any(
        len(record.minimal_provenance_sets) > 1 for record in records["stress_duplicate_decoy"]
    )
    assert {record.renderer_family for record in records["stress_renderer_holdout"]} == {
        "monochrome"
    }


def test_images_are_content_addressed_and_bounded(dataset_root: Path) -> None:
    records = load_dataset_records(dataset_root)
    for record in records["train"]:
        assert Path(record.image_path).stem == record.image_sha256
        with Image.open(dataset_root / record.image_path) as image:
            assert image.size == DEFAULT_IMAGE_SIZE
            assert image.width <= 512 and image.height <= 512


def test_renderer_metadata_proves_all_cells_visible(
    full_dataset_root: Path,
) -> None:
    records = load_dataset_records(full_dataset_root)
    for record in records["train"] + records["test_composition"] + records["stress_label_shuffle"]:
        values = [element for element in record.render_text if element.role is TextRole.VALUE]
        assert len(values) == 8
        assert {element.evidence for element in values} == set(record.table.cells())
        assert all(element.contrast_ratio >= MIN_TEXT_CONTRAST for element in record.render_text)
        for index, left in enumerate(record.render_text):
            assert left.bbox[0] >= 0
            assert left.bbox[1] >= 0
            assert left.bbox[2] <= DEFAULT_IMAGE_SIZE[0]
            assert left.bbox[3] <= DEFAULT_IMAGE_SIZE[1]
            assert all(
                not boxes_overlap(left.bbox, right.bbox)
                for right in record.render_text[index + 1 :]
            )


def test_balanced_factor_schedule_and_answer_entropy(
    full_dataset_root: Path,
) -> None:
    report = audit_dataset(full_dataset_root)
    assert report["status"] == "pass"
    train_contingency = report["factor_contingency"]["train"]
    programs = {"lookup", "comparison", "argmax", "count"}
    for factor in (
        "chart_type",
        "renderer_family",
        "category_family",
        "series_family",
        "target_series_index",
        "title",
    ):
        for level_counts in train_contingency[factor].values():
            assert set(level_counts) == programs
    train_records = load_dataset_records(full_dataset_root)["train"]
    assert {(record.chart_type.value, record.renderer_family) for record in train_records} == {
        (chart, renderer)
        for chart in ("bar", "line", "table")
        for renderer in ("classic", "pastel", "high_contrast")
    }
    assert {(record.category_family, record.series_family) for record in train_records} == {
        (category, series)
        for category in ("quarter", "month", "region", "product")
        for series in ("finance", "sales", "operations", "score")
    }
    composition = report["factor_contingency"]["test_composition"]
    for factor_counts in composition.values():
        for level_counts in factor_counts.values():
            assert set(level_counts) == {"difference", "sum_selected"}
    count_distribution = report["program_answer_distributions"]["train"]["count"]
    assert count_distribution["answer_counts"] == {"1": 8, "2": 8, "3": 8}
    assert count_distribution["entropy_bits"] > 1.5


def test_audit_detects_image_tampering(tmp_path: Path) -> None:
    root = tmp_path / "tampered"
    generate_dataset(root, smoke_config())
    record = load_dataset_records(root)["train"][0]
    image = root / record.image_path
    image.write_bytes(image.read_bytes() + b"tamper")
    report = audit_dataset(root)
    assert report["status"] == "fail"
    assert any("image hash mismatch" in error for error in report["errors"])
    with pytest.raises(ValueError, match="Dataset integrity verification failed"):
        load_dataset_records(root)


def test_overwrite_refuses_unrecognized_directories(tmp_path: Path) -> None:
    root = tmp_path / "not-a-dataset"
    root.mkdir()
    important = root / "important.txt"
    important.write_text("keep")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        generate_dataset(root, smoke_config(), overwrite=True)
    assert important.read_text() == "keep"


def test_audit_rejects_within_split_image_and_table_duplicates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "duplicates"
    generate_dataset(root, smoke_config())
    dataset = load_json(root / "dataset.json")
    manifest_path = root / dataset["splits"]["train"]["manifest"]
    rows = load_jsonl(manifest_path)
    rows.append(rows[0])
    write_jsonl(manifest_path, rows)
    dataset["splits"]["train"]["count"] = len(rows)
    dataset["splits"]["train"]["manifest_sha256"] = sha256_bytes(manifest_path.read_bytes())
    write_json(root / "dataset.json", dataset)
    report = audit_dataset(root)
    assert report["status"] == "fail"
    assert report["within_split_duplicate_hash_count"] == {
        "record": 1,
        "image": 1,
        "table": 1,
    }
