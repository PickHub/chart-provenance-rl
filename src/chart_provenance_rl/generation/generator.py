from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from chart_provenance_rl.constants import MODEL_ID, MODEL_LICENSE, MODEL_REVISION, SCHEMA_VERSION
from chart_provenance_rl.generation.config import DatasetConfig, SplitSpec
from chart_provenance_rl.generation.renderer import render_chart
from chart_provenance_rl.programs import execute_program, question_for_program
from chart_provenance_rl.prompts import build_prompt
from chart_provenance_rl.schema import (
    BenchmarkRecord,
    ChartType,
    ComparisonKind,
    ProgramKind,
    ProgramSpec,
    TableData,
)
from chart_provenance_rl.serialization import (
    load_json,
    sha256_bytes,
    sha256_json,
    write_json,
    write_jsonl,
)

CATEGORY_FAMILIES = (
    ("quarter", ("Q1", "Q2", "Q3", "Q4")),
    ("month", ("Jan", "Feb", "Mar", "Apr")),
    ("region", ("North", "South", "East", "West")),
    ("product", ("Atlas", "Beacon", "Cobalt", "Delta")),
)
SERIES_FAMILIES = (
    ("finance", ("Revenue", "Cost")),
    ("sales", ("Sales", "Returns")),
    ("operations", ("Volume", "Target")),
    ("score", ("Score", "Baseline")),
)
TITLES = ("Quarterly summary", "Operational metrics", "Regional comparison", "Product results")


@dataclass(frozen=True)
class _FactorAssignment:
    program: ProgramKind
    program_occurrence: int
    chart_type: ChartType
    renderer_family: str
    category_family_index: int
    series_family_index: int
    title_index: int
    target_series_index: int


def generate_dataset(
    root: Path,
    config: DatasetConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset root already exists: {root}")
        _clear_generated_dataset(root)
    image_dir = root / "images"
    record_dir = root / "records"
    manifest_dir = root / "manifests"
    image_dir.mkdir(parents=True)
    record_dir.mkdir()
    manifest_dir.mkdir()

    split_records: dict[str, list[BenchmarkRecord]] = {}
    for split in config.splits:
        records = [_generate_record(root, split, index) for index in range(split.count)]
        split_records[split.name] = records
        write_jsonl(
            manifest_dir / f"{split.name}.jsonl",
            (record.to_dict() for record in records),
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "license": "CC0-1.0",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
        },
        "config": config.to_dict(),
        "splits": {
            split: {
                "count": len(records),
                "manifest": f"manifests/{split}.jsonl",
                "manifest_sha256": sha256_bytes((manifest_dir / f"{split}.jsonl").read_bytes()),
            }
            for split, records in split_records.items()
        },
    }
    write_json(root / "dataset.json", manifest)
    return manifest


def _generate_record(root: Path, split: SplitSpec, index: int) -> BenchmarkRecord:
    factors = _factor_assignment(split, index)
    table_rng = _seeded_rng(split.config_seed, index, 1_009)
    program_rng = _seeded_rng(split.config_seed, index, 2_003)
    table, category_family, series_family = _generate_table(
        split,
        factors,
        table_rng,
    )
    program = _generate_program(factors, table, program_rng, split.tags)
    result = execute_program(table, program)
    grammar_rng = _seeded_rng(split.grammar_seed, index, 3_007)
    question = question_for_program(program, grammar_rng.randrange(2))
    rendered = render_chart(
        table,
        factors.chart_type,
        factors.renderer_family,
    )
    image = rendered.image_bytes
    image_sha256 = sha256_bytes(image)
    image_path = f"images/{image_sha256}.png"
    image_target = root / image_path
    if not image_target.exists():
        image_target.write_bytes(image)
    table_sha256 = sha256_json(table.to_dict())
    provisional = BenchmarkRecord(
        record_id="0" * 64,
        split=split.name,
        question=question,
        prompt=build_prompt(question),
        chart_type=factors.chart_type,
        renderer_family=factors.renderer_family,
        image_sha256=image_sha256,
        image_path=image_path,
        table_sha256=table_sha256,
        table=table,
        program=program,
        answer=result.answer,
        minimal_provenance_sets=result.minimal_provenance_sets,
        grammar_seed=split.grammar_seed,
        config_seed=split.config_seed,
        generation_index=index,
        category_family=category_family,
        series_family=series_family,
        target_series_index=factors.target_series_index,
        render_text=rendered.text_elements,
        tags=split.tags,
    )
    record_id = sha256_json(provisional.to_dict(include_record_id=False))
    record = replace(provisional, record_id=record_id)
    write_json(root / "records" / f"{record_id}.json", record.to_dict())
    return record


def _generate_table(
    split: SplitSpec,
    factors: _FactorAssignment,
    rng: random.Random,
) -> tuple[TableData, str, str]:
    category_family, category_labels = CATEGORY_FAMILIES[factors.category_family_index]
    series_family, series_labels = SERIES_FAMILIES[factors.series_family_index]
    categories = list(category_labels)
    series = list(series_labels)
    if "label_shuffle" in split.tags:
        categories = [f"Group {label}" for label in rng.sample(("A", "C", "F", "H"), 4)]
        series = [f"Measure {label}" for label in rng.sample(("K", "R"), 2)]
        category_family = "shuffled"
        series_family = "shuffled"
    values = [[_random_value(rng) for _category in categories] for _series in series]
    if "duplicate_value_decoy" in split.tags:
        decoy_series_index = 1 - factors.target_series_index
        decoy_start = 0 if decoy_series_index == 0 else 2
        values[decoy_series_index][decoy_start + 1] = values[decoy_series_index][decoy_start]
        if factors.program is ProgramKind.LOOKUP:
            target_start = 0 if factors.target_series_index == 0 else 2
            values[factors.target_series_index][target_start + 1] = values[
                factors.target_series_index
            ][target_start]
    if factors.program is ProgramKind.COUNT:
        _make_values_unique(values[factors.target_series_index])
    if "category_series_order_perturbation" in split.tags:
        category_order = list(range(len(categories)))
        rng.shuffle(category_order)
        categories = [categories[position] for position in category_order]
        values = [[row[position] for position in category_order] for row in values]
        if rng.random() < 0.5:
            series.reverse()
            values.reverse()
    return (
        TableData(
            title=TITLES[factors.title_index],
            categories=tuple(categories),
            series=tuple(series),
            values=tuple(tuple(row) for row in values),
        ),
        category_family,
        series_family,
    )


def _generate_program(
    factors: _FactorAssignment,
    table: TableData,
    rng: random.Random,
    tags: tuple[str, ...],
) -> ProgramSpec:
    kind = factors.program
    series = table.series[factors.target_series_index]
    category, other = rng.sample(list(table.categories), k=2)
    if kind is ProgramKind.LOOKUP:
        alternatives: tuple[str, ...] = ()
        if "duplicate_value_decoy" in tags:
            alternatives = (
                table.categories[:2] if factors.target_series_index == 0 else table.categories[2:]
            )
            category = alternatives[0]
        return ProgramSpec(
            kind=kind,
            series=series,
            category=category,
            categories=alternatives,
        )
    if kind is ProgramKind.COMPARISON:
        category, other = _non_equal_category_pair(table, series, rng)
        left = table.get(series, category)
        right = table.get(series, other)
        true_comparison = ComparisonKind.GREATER if left > right else ComparisonKind.LESS
        false_comparison = (
            ComparisonKind.LESS
            if true_comparison is ComparisonKind.GREATER
            else ComparisonKind.GREATER
        )
        return ProgramSpec(
            kind=kind,
            series=series,
            category=category,
            other_category=other,
            comparison=(
                true_comparison if factors.program_occurrence % 2 == 0 else false_comparison
            ),
        )
    if kind is ProgramKind.ARGMAX:
        return ProgramSpec(kind=kind, series=series)
    if kind is ProgramKind.COUNT:
        ordered = sorted(table.get(series, label) for label in table.categories)
        valid_thresholds = [
            (count, (ordered[boundary] + ordered[boundary + 1]) / 2)
            for count in (1, 2, 3)
            if (boundary := len(ordered) - count - 1) >= 0
            and ordered[boundary] < ordered[boundary + 1]
        ]
        if not valid_thresholds:
            raise ValueError("Count program requires at least two distinct values")
        desired_count = 1 + factors.program_occurrence % 3
        _count, threshold = min(
            valid_thresholds,
            key=lambda candidate: (
                abs(candidate[0] - desired_count),
                candidate[0],
            ),
        )
        return ProgramSpec(
            kind=kind,
            series=series,
            comparison=ComparisonKind.GREATER,
            threshold=threshold,
        )
    if kind is ProgramKind.DIFFERENCE:
        return ProgramSpec(
            kind=kind,
            series=series,
            category=category,
            other_category=other,
        )
    selected = rng.sample(list(table.categories), k=3)
    return ProgramSpec(
        kind=kind,
        series=series,
        categories=tuple(selected),
    )


def _random_value(rng: random.Random) -> Decimal:
    whole = rng.randint(8, 96)
    fraction = rng.choice(("0", "0.25", "0.5", "0.75"))
    return Decimal(whole) + Decimal(fraction)


def _make_values_unique(values: list[Decimal]) -> None:
    seen: set[Decimal] = set()
    for index, value in enumerate(values):
        while value in seen:
            value += Decimal("0.25")
        values[index] = value
        seen.add(value)


def _factor_assignment(split: SplitSpec, index: int) -> _FactorAssignment:
    program_index = index % len(split.programs)
    occurrence = index // len(split.programs)
    return _FactorAssignment(
        program=split.programs[program_index],
        program_occurrence=occurrence,
        chart_type=_balanced_choice(
            split.chart_types,
            occurrence,
            program_index,
            split.config_seed,
            101,
        ),
        renderer_family=_balanced_choice(
            split.renderer_families,
            occurrence,
            program_index,
            split.config_seed,
            211,
        ),
        category_family_index=_balanced_choice(
            tuple(range(len(CATEGORY_FAMILIES))),
            occurrence,
            program_index,
            split.config_seed,
            307,
        ),
        series_family_index=_balanced_choice(
            tuple(range(len(SERIES_FAMILIES))),
            occurrence,
            program_index,
            split.config_seed,
            401,
        ),
        title_index=_balanced_choice(
            tuple(range(len(TITLES))),
            occurrence,
            program_index,
            split.config_seed,
            503,
        ),
        target_series_index=_balanced_choice(
            (0, 1),
            occurrence,
            program_index,
            split.config_seed,
            601,
        ),
    )


def _balanced_choice[T](
    values: tuple[T, ...],
    occurrence: int,
    program_index: int,
    seed: int,
    salt: int,
) -> T:
    cycle, offset = divmod(occurrence, len(values))
    order = list(values)
    _seeded_rng(seed + program_index * 65_537, cycle, salt).shuffle(order)
    return order[offset]


def _seeded_rng(seed: int, index: int, salt: int) -> random.Random:
    return random.Random(seed + index * 104_729 + salt * 1_000_003)


def _non_equal_category_pair(
    table: TableData,
    series: str,
    rng: random.Random,
) -> tuple[str, str]:
    pairs = [
        (left, right)
        for left in table.categories
        for right in table.categories
        if left != right and table.get(series, left) != table.get(series, right)
    ]
    if not pairs:
        raise ValueError("Comparison program requires distinct visible values")
    return rng.choice(pairs)


def _clear_generated_dataset(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("Refusing to overwrite a symlinked dataset root")
    children = list(root.iterdir())
    if not children:
        return
    if any(child.is_symlink() for child in children):
        raise ValueError("Refusing to overwrite a dataset containing symlinks")
    marker_path = root / "dataset.json"
    if not marker_path.is_file():
        raise ValueError("Refusing to overwrite a directory without dataset.json")
    marker = load_json(marker_path)
    if not isinstance(marker, dict) or marker.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Refusing to overwrite an unrecognized dataset directory")
    allowed = {
        "dataset.json",
        "audit_report.json",
        "images",
        "manifests",
        "probes",
        "records",
        "smoke_report.json",
    }
    unexpected = sorted(child.name for child in children if child.name not in allowed)
    if unexpected:
        raise ValueError(f"Refusing to delete unexpected dataset paths: {unexpected}")
    for directory_name in ("images", "manifests", "probes", "records"):
        directory = root / directory_name
        if directory.is_dir():
            shutil.rmtree(directory)
    for file_name in ("audit_report.json", "dataset.json", "smoke_report.json"):
        file_path = root / file_name
        if file_path.is_file():
            file_path.unlink()
