from __future__ import annotations

import random
import shutil
from dataclasses import replace
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
    ComparisonKind,
    ProgramKind,
    ProgramSpec,
    TableData,
)
from chart_provenance_rl.serialization import sha256_bytes, sha256_json, write_json, write_jsonl

CATEGORY_FAMILIES = (
    ("Q1", "Q2", "Q3", "Q4"),
    ("Jan", "Feb", "Mar", "Apr"),
    ("North", "South", "East", "West"),
    ("Atlas", "Beacon", "Cobalt", "Delta"),
)
SERIES_FAMILIES = (
    ("Revenue", "Cost"),
    ("Sales", "Returns"),
    ("Volume", "Target"),
    ("Score", "Baseline"),
)
TITLES = ("Quarterly summary", "Operational metrics", "Regional comparison", "Product results")


def generate_dataset(
    root: Path,
    config: DatasetConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset root already exists: {root}")
        shutil.rmtree(root)
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
        "license": "Apache-2.0",
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
    rng = random.Random(split.config_seed + index * 104_729)
    table = _generate_table(split, index, rng)
    program = _generate_program(split, index, table, rng)
    result = execute_program(table, program)
    question = question_for_program(program, split.grammar_seed + index)
    chart_type = split.chart_types[index % len(split.chart_types)]
    renderer = split.renderer_families[index % len(split.renderer_families)]
    image = render_chart(table, chart_type, renderer)
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
        chart_type=chart_type,
        renderer_family=renderer,
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
        tags=split.tags,
    )
    record_id = sha256_json(provisional.to_dict(include_record_id=False))
    record = replace(provisional, record_id=record_id)
    write_json(root / "records" / f"{record_id}.json", record.to_dict())
    return record


def _generate_table(split: SplitSpec, index: int, rng: random.Random) -> TableData:
    categories = list(CATEGORY_FAMILIES[index % len(CATEGORY_FAMILIES)])
    series = list(SERIES_FAMILIES[(index // len(CATEGORY_FAMILIES)) % len(SERIES_FAMILIES)])
    if "label_shuffle" in split.tags:
        categories = [f"Group {label}" for label in rng.sample(("A", "C", "F", "H"), 4)]
        series = [f"Measure {label}" for label in rng.sample(("K", "R"), 2)]
    values = [[_random_value(rng) for _category in categories] for _series in series]
    if "duplicate_value_decoy" in split.tags:
        values[0][1] = values[0][0]
        values[1][2] = values[1][3]
    if "category_series_order_perturbation" in split.tags:
        category_order = list(range(len(categories)))
        rng.shuffle(category_order)
        categories = [categories[position] for position in category_order]
        values = [[row[position] for position in category_order] for row in values]
        if rng.random() < 0.5:
            series.reverse()
            values.reverse()
    return TableData(
        title=TITLES[index % len(TITLES)],
        categories=tuple(categories),
        series=tuple(series),
        values=tuple(tuple(row) for row in values),
    )


def _generate_program(
    split: SplitSpec,
    index: int,
    table: TableData,
    rng: random.Random,
) -> ProgramSpec:
    kind = split.programs[index % len(split.programs)]
    series = table.series[index % len(table.series)]
    category = table.categories[index % len(table.categories)]
    other = table.categories[(index + 1) % len(table.categories)]
    if kind is ProgramKind.LOOKUP:
        alternatives: tuple[str, ...] = ()
        if "duplicate_value_decoy" in split.tags and series == table.series[0]:
            category = table.categories[0]
            alternatives = table.categories[:2]
        return ProgramSpec(
            kind=kind,
            series=series,
            category=category,
            categories=alternatives,
        )
    if kind is ProgramKind.COMPARISON:
        return ProgramSpec(
            kind=kind,
            series=series,
            category=category,
            other_category=other,
            comparison=(ComparisonKind.GREATER, ComparisonKind.LESS)[index % 2],
        )
    if kind is ProgramKind.ARGMAX:
        return ProgramSpec(kind=kind, series=series)
    if kind is ProgramKind.COUNT:
        ordered = sorted(table.get(series, label) for label in table.categories)
        threshold = (ordered[1] + ordered[2]) / 2
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
