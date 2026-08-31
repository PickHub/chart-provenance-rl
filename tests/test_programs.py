from decimal import Decimal

import pytest

from chart_provenance_rl.programs import execute_program, question_for_program
from chart_provenance_rl.schema import (
    ComparisonKind,
    EvidenceRef,
    ProgramKind,
    ProgramSpec,
    TableData,
)


@pytest.mark.parametrize(
    ("program", "answer", "provenance_size"),
    [
        (
            ProgramSpec(ProgramKind.LOOKUP, "Revenue", category="Q1"),
            "10",
            1,
        ),
        (
            ProgramSpec(
                ProgramKind.COMPARISON,
                "Revenue",
                category="Q4",
                other_category="Q3",
                comparison=ComparisonKind.GREATER,
            ),
            "yes",
            2,
        ),
        (ProgramSpec(ProgramKind.ARGMAX, "Revenue"), "Q4", 4),
        (
            ProgramSpec(
                ProgramKind.COUNT,
                "Revenue",
                comparison=ComparisonKind.GREATER,
                threshold=Decimal(20),
            ),
            "2",
            4,
        ),
        (
            ProgramSpec(
                ProgramKind.DIFFERENCE,
                "Revenue",
                category="Q4",
                other_category="Q3",
            ),
            "10",
            2,
        ),
        (
            ProgramSpec(
                ProgramKind.SUM_SELECTED,
                "Revenue",
                categories=("Q1", "Q3", "Q4"),
            ),
            "80",
            3,
        ),
    ],
)
def test_program_execution(
    program: ProgramSpec,
    answer: str,
    provenance_size: int,
) -> None:
    result = execute_program(_table(), program)
    assert result.answer == answer
    assert len(result.minimal_provenance_sets[0]) == provenance_size
    assert question_for_program(program, 0)
    assert question_for_program(program, 1)


def test_lookup_supports_alternative_minimal_witnesses() -> None:
    program = ProgramSpec(
        ProgramKind.LOOKUP,
        "Revenue",
        category="Q1",
        categories=("Q1", "Q2"),
    )
    result = execute_program(_table(), program)
    assert result.minimal_provenance_sets == (
        (EvidenceRef("Revenue", "Q1"),),
        (EvidenceRef("Revenue", "Q2"),),
    )
    assert "same" in question_for_program(program, 0)


def test_argmax_reports_ties_without_hidden_tiebreak() -> None:
    table = TableData(
        "Tie",
        ("Q1", "Q2"),
        ("Revenue",),
        ((Decimal(10), Decimal(10)),),
    )
    result = execute_program(table, ProgramSpec(ProgramKind.ARGMAX, "Revenue"))
    assert result.answer == "Q1, Q2"


def test_unknown_cell_fails_explicitly() -> None:
    with pytest.raises(KeyError):
        execute_program(
            _table(),
            ProgramSpec(ProgramKind.LOOKUP, "Revenue", category="Q9"),
        )


def _table() -> TableData:
    return TableData(
        "Example",
        ("Q1", "Q2", "Q3", "Q4"),
        ("Revenue", "Cost"),
        (
            (Decimal(10), Decimal(10), Decimal(30), Decimal(40)),
            (Decimal(8), Decimal(12), Decimal(17), Decimal(22)),
        ),
    )
