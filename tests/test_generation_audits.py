from pathlib import Path

import pytest
from PIL import Image

from chart_provenance_rl.audits import audit_dataset, load_dataset_records
from chart_provenance_rl.constants import DEFAULT_IMAGE_SIZE
from chart_provenance_rl.generation.config import smoke_config
from chart_provenance_rl.generation.generator import generate_dataset
from chart_provenance_rl.schema import ProgramKind
from chart_provenance_rl.serialization import load_json


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


def test_audit_detects_image_tampering(tmp_path: Path) -> None:
    root = tmp_path / "tampered"
    generate_dataset(root, smoke_config())
    record = load_dataset_records(root)["train"][0]
    image = root / record.image_path
    image.write_bytes(image.read_bytes() + b"tamper")
    report = audit_dataset(root)
    assert report["status"] == "fail"
    assert any("image hash mismatch" in error for error in report["errors"])


def test_overwrite_refuses_unrecognized_directories(tmp_path: Path) -> None:
    root = tmp_path / "not-a-dataset"
    root.mkdir()
    important = root / "important.txt"
    important.write_text("keep")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        generate_dataset(root, smoke_config(), overwrite=True)
    assert important.read_text() == "keep"
