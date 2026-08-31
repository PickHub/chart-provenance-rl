from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from chart_provenance_rl.decimal_utils import canonical_decimal
from chart_provenance_rl.schema import (
    ComparisonKind,
    EvidenceRef,
    ProgramKind,
    ProgramResult,
    ProgramSpec,
    TableData,
)


def execute_program(table: TableData, program: ProgramSpec) -> ProgramResult:
    handlers: dict[ProgramKind, Callable[[TableData, ProgramSpec], ProgramResult]] = {
        ProgramKind.LOOKUP: _execute_lookup,
        ProgramKind.COMPARISON: _execute_comparison,
        ProgramKind.ARGMAX: _execute_argmax,
        ProgramKind.COUNT: _execute_count,
        ProgramKind.DIFFERENCE: _execute_difference,
        ProgramKind.SUM_SELECTED: _execute_sum_selected,
    }
    return handlers[program.kind](table, program)


def question_for_program(program: ProgramSpec, variant: int) -> str:
    series = program.series
    if program.kind is ProgramKind.ARGMAX:
        templates = (
            f"Which category has the highest {series} value?",
            f"At which category does {series} reach its maximum?",
        )
    elif program.kind is ProgramKind.COUNT:
        templates = (
            f"How many categories have {series} {_threshold_phrase(program)}?",
            f"Count the categories where {series} is {_threshold_phrase(program)}.",
        )
    elif program.kind is ProgramKind.SUM_SELECTED:
        templates = (
            f"What is the sum of {series} for {_join_labels(program.categories)}?",
            f"Add the {series} values at {_join_labels(program.categories)}.",
        )
    else:
        category = _required(program.category, "category")
        other = program.other_category
        if program.kind is ProgramKind.LOOKUP and program.categories:
            labels = _join_labels(program.categories)
            return (
                f"{labels} show the same {series} value. What is that value?"
                if variant % 2 == 0
                else f"What shared {series} value is visible at {labels}?"
            )
        if program.kind is ProgramKind.LOOKUP:
            templates = (
                f"What value is shown for {series} in {category}?",
                f"Read the {series} value at {category}.",
            )
        elif program.kind is ProgramKind.COMPARISON:
            other = _required(other, "other_category")
            templates = (
                f"Is {series} in {category} {_comparison_phrase(program)} {other}?",
                f"For {series}, is {category} {_comparison_phrase(program)} {other}?",
            )
        else:
            other = _required(other, "other_category")
            templates = (
                f"What is {series} in {category} minus {series} in {other}?",
                f"Compute {series} at {category} minus {other}, in that order.",
            )
    return templates[variant % 2]


def _execute_lookup(table: TableData, program: ProgramSpec) -> ProgramResult:
    category = _required(program.category, "category")
    value = table.get(program.series, category)
    candidates = program.categories or (category,)
    if category not in candidates:
        raise ValueError("The primary lookup category must be an evidence candidate")
    provenance = tuple(
        (EvidenceRef(series=program.series, category=candidate),) for candidate in candidates
    )
    return ProgramResult(answer=canonical_decimal(value), minimal_provenance_sets=provenance)


def _execute_comparison(table: TableData, program: ProgramSpec) -> ProgramResult:
    category = _required(program.category, "category")
    other = _required(program.other_category, "other_category")
    comparison = _required(program.comparison, "comparison")
    left = table.get(program.series, category)
    right = table.get(program.series, other)
    answer = "yes" if _compare(left, right, comparison) else "no"
    provenance = (
        EvidenceRef(series=program.series, category=category),
        EvidenceRef(series=program.series, category=other),
    )
    return ProgramResult(answer=answer, minimal_provenance_sets=(provenance,))


def _execute_argmax(table: TableData, program: ProgramSpec) -> ProgramResult:
    values = [table.get(program.series, category) for category in table.categories]
    maximum = max(values)
    winners = [
        category
        for category, value in zip(table.categories, values, strict=True)
        if value == maximum
    ]
    answer = ", ".join(winners)
    provenance = tuple(
        EvidenceRef(series=program.series, category=category) for category in table.categories
    )
    return ProgramResult(answer=answer, minimal_provenance_sets=(provenance,))


def _execute_count(table: TableData, program: ProgramSpec) -> ProgramResult:
    threshold = _required(program.threshold, "threshold")
    comparison = _required(program.comparison, "comparison")
    count = sum(
        _compare(table.get(program.series, category), threshold, comparison)
        for category in table.categories
    )
    provenance = tuple(
        EvidenceRef(series=program.series, category=category) for category in table.categories
    )
    return ProgramResult(answer=str(count), minimal_provenance_sets=(provenance,))


def _execute_difference(table: TableData, program: ProgramSpec) -> ProgramResult:
    category = _required(program.category, "category")
    other = _required(program.other_category, "other_category")
    answer = table.get(program.series, category) - table.get(program.series, other)
    provenance = (
        EvidenceRef(series=program.series, category=category),
        EvidenceRef(series=program.series, category=other),
    )
    return ProgramResult(
        answer=canonical_decimal(answer),
        minimal_provenance_sets=(provenance,),
    )


def _execute_sum_selected(table: TableData, program: ProgramSpec) -> ProgramResult:
    answer = sum(
        (table.get(program.series, category) for category in program.categories),
        start=Decimal(0),
    )
    provenance = tuple(
        EvidenceRef(series=program.series, category=category) for category in program.categories
    )
    return ProgramResult(
        answer=canonical_decimal(answer),
        minimal_provenance_sets=(provenance,),
    )


def _compare(left: Decimal, right: Decimal, comparison: ComparisonKind) -> bool:
    if comparison is ComparisonKind.GREATER:
        return left > right
    if comparison is ComparisonKind.LESS:
        return left < right
    return left == right


def _comparison_phrase(program: ProgramSpec) -> str:
    comparison = _required(program.comparison, "comparison")
    return {
        ComparisonKind.GREATER: "greater than",
        ComparisonKind.LESS: "less than",
        ComparisonKind.EQUAL: "equal to",
    }[comparison]


def _threshold_phrase(program: ProgramSpec) -> str:
    comparison = _required(program.comparison, "comparison")
    threshold = canonical_decimal(_required(program.threshold, "threshold"))
    return {
        ComparisonKind.GREATER: f"greater than {threshold}",
        ComparisonKind.LESS: f"less than {threshold}",
        ComparisonKind.EQUAL: f"equal to {threshold}",
    }[comparison]


def _join_labels(labels: tuple[str, ...]) -> str:
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _required[T](value: T | None, field_name: str) -> T:
    if value is None:
        raise ValueError(f"Program is missing {field_name}")
    return value
