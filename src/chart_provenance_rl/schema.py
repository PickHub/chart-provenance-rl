from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from chart_provenance_rl.constants import SCHEMA_VERSION
from chart_provenance_rl.decimal_utils import canonical_decimal, parse_decimal

type JsonMapping = Mapping[str, Any]


class ChartType(StrEnum):
    BAR = "bar"
    LINE = "line"
    TABLE = "table"


class ProgramKind(StrEnum):
    LOOKUP = "lookup"
    COMPARISON = "comparison"
    ARGMAX = "argmax"
    COUNT = "count"
    DIFFERENCE = "difference"
    SUM_SELECTED = "sum_selected"


class ComparisonKind(StrEnum):
    GREATER = "greater"
    LESS = "less"
    EQUAL = "equal"


class TextRole(StrEnum):
    TITLE = "title"
    CATEGORY = "category"
    LEGEND = "legend"
    VALUE = "value"


@dataclass(frozen=True, order=True)
class EvidenceRef:
    series: str
    category: str

    def __post_init__(self) -> None:
        if not self.series.strip() or not self.category.strip():
            raise ValueError("Evidence labels must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"series": self.series, "category": self.category}

    @classmethod
    def from_dict(cls, value: JsonMapping) -> EvidenceRef:
        if set(value) != {"series", "category"}:
            raise ValueError("Evidence objects require exactly series and category")
        series = value["series"]
        category = value["category"]
        if not isinstance(series, str) or not isinstance(category, str):
            raise TypeError("Evidence labels must be strings")
        return cls(series=series, category=category)


@dataclass(frozen=True)
class TableData:
    title: str
    categories: tuple[str, ...]
    series: tuple[str, ...]
    values: tuple[tuple[Decimal, ...], ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Table title must be non-empty")
        if not self.categories or not self.series:
            raise ValueError("A table requires categories and series")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("Category labels must be unique")
        if len(set(self.series)) != len(self.series):
            raise ValueError("Series labels must be unique")
        if len(self.values) != len(self.series):
            raise ValueError("Values must contain one row per series")
        if any(len(row) != len(self.categories) for row in self.values):
            raise ValueError("Every value row must match the category count")
        if any(not value.is_finite() for row in self.values for value in row):
            raise ValueError("Table values must be finite")

    def get(self, series: str, category: str) -> Decimal:
        try:
            series_index = self.series.index(series)
            category_index = self.categories.index(category)
        except ValueError as error:
            raise KeyError(f"Unknown cell: {series!r}, {category!r}") from error
        return self.values[series_index][category_index]

    def cells(self) -> tuple[EvidenceRef, ...]:
        return tuple(
            EvidenceRef(series=series, category=category)
            for series in self.series
            for category in self.categories
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "categories": list(self.categories),
            "series": list(self.series),
            "values": [[canonical_decimal(value) for value in row] for row in self.values],
        }

    @classmethod
    def from_dict(cls, value: JsonMapping) -> TableData:
        categories = _string_tuple(value["categories"], "categories")
        series = _string_tuple(value["series"], "series")
        rows = value["values"]
        if not isinstance(rows, list):
            raise TypeError("values must be a list")
        parsed_rows: list[tuple[Decimal, ...]] = []
        for row in rows:
            if not isinstance(row, list):
                raise TypeError("value rows must be lists")
            parsed_rows.append(tuple(parse_decimal(item) for item in row))
        title = value["title"]
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        return cls(
            title=title,
            categories=categories,
            series=series,
            values=tuple(parsed_rows),
        )


@dataclass(frozen=True)
class ProgramSpec:
    kind: ProgramKind
    series: str
    category: str | None = None
    other_category: str | None = None
    categories: tuple[str, ...] = ()
    comparison: ComparisonKind | None = None
    threshold: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.series.strip():
            raise ValueError("Program series must be non-empty")
        required: dict[ProgramKind, tuple[str, ...]] = {
            ProgramKind.LOOKUP: ("category",),
            ProgramKind.COMPARISON: ("category", "other_category", "comparison"),
            ProgramKind.ARGMAX: (),
            ProgramKind.COUNT: ("threshold", "comparison"),
            ProgramKind.DIFFERENCE: ("category", "other_category"),
            ProgramKind.SUM_SELECTED: ("categories",),
        }
        for field_name in required[self.kind]:
            field_value = getattr(self, field_name)
            if field_value is None or field_value == ():
                raise ValueError(f"{self.kind.value} requires {field_name}")
        if self.kind is ProgramKind.SUM_SELECTED and len(self.categories) < 2:
            raise ValueError("sum_selected requires at least two categories")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind.value, "series": self.series}
        if self.category is not None:
            result["category"] = self.category
        if self.other_category is not None:
            result["other_category"] = self.other_category
        if self.categories:
            result["categories"] = list(self.categories)
        if self.comparison is not None:
            result["comparison"] = self.comparison.value
        if self.threshold is not None:
            result["threshold"] = canonical_decimal(self.threshold)
        return result

    @classmethod
    def from_dict(cls, value: JsonMapping) -> ProgramSpec:
        return cls(
            kind=ProgramKind(str(value["kind"])),
            series=str(value["series"]),
            category=_optional_string(value.get("category")),
            other_category=_optional_string(value.get("other_category")),
            categories=_string_tuple(value.get("categories", []), "categories"),
            comparison=(
                ComparisonKind(str(value["comparison"]))
                if value.get("comparison") is not None
                else None
            ),
            threshold=(
                parse_decimal(value["threshold"]) if value.get("threshold") is not None else None
            ),
        )


@dataclass(frozen=True)
class ProgramResult:
    answer: str
    minimal_provenance_sets: tuple[tuple[EvidenceRef, ...], ...]

    def __post_init__(self) -> None:
        if not self.minimal_provenance_sets:
            raise ValueError("At least one valid provenance set is required")
        if any(not provenance for provenance in self.minimal_provenance_sets):
            raise ValueError("Provenance sets must be non-empty")


@dataclass(frozen=True)
class RenderText:
    role: TextRole
    text: str
    bbox: tuple[int, int, int, int]
    foreground: str
    background: str
    contrast_ratio: Decimal
    evidence: EvidenceRef | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Rendered text must be non-empty")
        left, top, right, bottom = self.bbox
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError(f"Invalid rendered text bounding box: {self.bbox}")
        if self.contrast_ratio < 1:
            raise ValueError("Rendered text contrast ratio must be at least 1")
        if self.role is TextRole.VALUE and self.evidence is None:
            raise ValueError("Rendered values require a semantic evidence reference")
        if self.role is not TextRole.VALUE and self.evidence is not None:
            raise ValueError("Only rendered values may carry evidence references")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role.value,
            "text": self.text,
            "bbox": list(self.bbox),
            "foreground": self.foreground,
            "background": self.background,
            "contrast_ratio": canonical_decimal(self.contrast_ratio),
        }
        if self.evidence is not None:
            result["evidence"] = self.evidence.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: JsonMapping) -> RenderText:
        raw_bbox = value["bbox"]
        if (
            not isinstance(raw_bbox, list | tuple)
            or len(raw_bbox) != 4
            or not all(isinstance(coordinate, int) for coordinate in raw_bbox)
        ):
            raise TypeError("Rendered text bbox must contain four integers")
        raw_evidence = value.get("evidence")
        return cls(
            role=TextRole(str(value["role"])),
            text=str(value["text"]),
            bbox=tuple(raw_bbox),
            foreground=str(value["foreground"]),
            background=str(value["background"]),
            contrast_ratio=parse_decimal(value["contrast_ratio"]),
            evidence=(
                EvidenceRef.from_dict(_mapping(raw_evidence)) if raw_evidence is not None else None
            ),
        )


@dataclass(frozen=True)
class BenchmarkRecord:
    record_id: str
    split: str
    question: str
    prompt: str
    chart_type: ChartType
    renderer_family: str
    image_sha256: str
    image_path: str
    table_sha256: str
    table: TableData
    program: ProgramSpec
    answer: str
    minimal_provenance_sets: tuple[tuple[EvidenceRef, ...], ...]
    grammar_seed: int
    config_seed: int
    generation_index: int
    category_family: str
    series_family: str
    target_series_index: int
    render_text: tuple[RenderText, ...]
    tags: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        for hash_value in (self.record_id, self.image_sha256, self.table_sha256):
            if len(hash_value) != 64 or any(char not in "0123456789abcdef" for char in hash_value):
                raise ValueError(f"Invalid SHA-256 value: {hash_value!r}")
        if not self.question.strip() or not self.prompt.strip():
            raise ValueError("Question and prompt must be non-empty")
        if not self.category_family or not self.series_family:
            raise ValueError("Generator family labels must be non-empty")
        if self.target_series_index not in range(len(self.table.series)):
            raise ValueError("Target series index is outside the table")
        if self.table.series[self.target_series_index] != self.program.series:
            raise ValueError("Target series index does not identify the program series")
        if not self.render_text:
            raise ValueError("Rendered text metadata must be retained")

    def to_dict(self, *, include_record_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "split": self.split,
            "question": self.question,
            "prompt": self.prompt,
            "chart_type": self.chart_type.value,
            "renderer_family": self.renderer_family,
            "image_sha256": self.image_sha256,
            "image_path": self.image_path,
            "table_sha256": self.table_sha256,
            "table": self.table.to_dict(),
            "program": self.program.to_dict(),
            "answer": self.answer,
            "minimal_provenance_sets": [
                [evidence.to_dict() for evidence in provenance]
                for provenance in self.minimal_provenance_sets
            ],
            "grammar_seed": self.grammar_seed,
            "config_seed": self.config_seed,
            "generation_index": self.generation_index,
            "category_family": self.category_family,
            "series_family": self.series_family,
            "target_series_index": self.target_series_index,
            "render_text": [element.to_dict() for element in self.render_text],
            "tags": list(self.tags),
        }
        if include_record_id:
            result["record_id"] = self.record_id
        return result

    @classmethod
    def from_dict(cls, value: JsonMapping) -> BenchmarkRecord:
        raw_sets = value["minimal_provenance_sets"]
        if not isinstance(raw_sets, list):
            raise TypeError("minimal_provenance_sets must be a list")
        provenance_sets: list[tuple[EvidenceRef, ...]] = []
        for raw_set in raw_sets:
            if not isinstance(raw_set, list):
                raise TypeError("Each provenance set must be a list")
            provenance_sets.append(tuple(EvidenceRef.from_dict(_mapping(item)) for item in raw_set))
        tags = _string_tuple(value.get("tags", []), "tags")
        return cls(
            schema_version=str(value["schema_version"]),
            record_id=str(value["record_id"]),
            split=str(value["split"]),
            question=str(value["question"]),
            prompt=str(value["prompt"]),
            chart_type=ChartType(str(value["chart_type"])),
            renderer_family=str(value["renderer_family"]),
            image_sha256=str(value["image_sha256"]),
            image_path=str(value["image_path"]),
            table_sha256=str(value["table_sha256"]),
            table=TableData.from_dict(_mapping(value["table"])),
            program=ProgramSpec.from_dict(_mapping(value["program"])),
            answer=str(value["answer"]),
            minimal_provenance_sets=tuple(provenance_sets),
            grammar_seed=int(value["grammar_seed"]),
            config_seed=int(value["config_seed"]),
            generation_index=int(value["generation_index"]),
            category_family=str(value["category_family"]),
            series_family=str(value["series_family"]),
            target_series_index=int(value["target_series_index"]),
            render_text=tuple(
                RenderText.from_dict(_mapping(element))
                for element in _mapping_list(value["render_text"], "render_text")
            ),
            tags=tags,
        )


def _mapping(value: Any) -> JsonMapping:
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping")
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must contain strings")
    return tuple(value)


def _mapping_list(value: Any, field_name: str) -> list[JsonMapping]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [_mapping(item) for item in value]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Expected a string")
    return value
