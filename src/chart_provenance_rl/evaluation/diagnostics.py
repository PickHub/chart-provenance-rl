from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from chart_provenance_rl.programs import execute_program
from chart_provenance_rl.schema import BenchmarkRecord, EvidenceRef, TableData


@dataclass(frozen=True)
class CounterfactualDiagnostic:
    necessity: float
    sufficiency: float
    provenance_set_index: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "necessity": self.necessity,
            "sufficiency": self.sufficiency,
            "provenance_set_index": self.provenance_set_index,
        }


def counterfactual_diagnostic(record: BenchmarkRecord) -> CounterfactualDiagnostic:
    diagnostics = [
        _diagnose_set(record, provenance, index)
        for index, provenance in enumerate(record.minimal_provenance_sets)
    ]
    return max(
        diagnostics,
        key=lambda diagnostic: (
            diagnostic.necessity + diagnostic.sufficiency,
            -diagnostic.provenance_set_index,
        ),
    )


def _diagnose_set(
    record: BenchmarkRecord,
    provenance: tuple[EvidenceRef, ...],
    index: int,
) -> CounterfactualDiagnostic:
    maximum = max(abs(value) for row in record.table.values for value in row)
    magnitude = maximum + Decimal(1_000)
    necessary = 0
    for reference in provenance:
        if _can_change_answer(record, reference, magnitude):
            necessary += 1
    non_evidence = tuple(cell for cell in record.table.cells() if cell not in provenance)
    perturbed = record.table
    for offset, reference in enumerate(non_evidence, start=1):
        value = magnitude + Decimal(offset * 17)
        perturbed = _replace_cell(perturbed, reference, value)
    sufficiency = float(execute_program(perturbed, record.program).answer == record.answer)
    return CounterfactualDiagnostic(
        necessity=necessary / len(provenance),
        sufficiency=sufficiency,
        provenance_set_index=index,
    )


def _can_change_answer(
    record: BenchmarkRecord,
    reference: EvidenceRef,
    magnitude: Decimal,
) -> bool:
    original = record.table.get(reference.series, reference.category)
    candidates = (magnitude, -magnitude, original + Decimal("0.5"))
    return any(
        execute_program(
            _replace_cell(record.table, reference, candidate),
            record.program,
        ).answer
        != record.answer
        for candidate in candidates
    )


def _replace_cell(
    table: TableData,
    reference: EvidenceRef,
    value: Decimal,
) -> TableData:
    series_index = table.series.index(reference.series)
    category_index = table.categories.index(reference.category)
    rows = [list(row) for row in table.values]
    rows[series_index][category_index] = value
    return TableData(
        title=table.title,
        categories=table.categories,
        series=table.series,
        values=tuple(tuple(row) for row in rows),
    )
