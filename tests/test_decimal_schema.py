from decimal import Decimal

import pytest

from chart_provenance_rl.decimal_utils import (
    canonical_answer,
    canonical_decimal,
    parse_decimal,
)
from chart_provenance_rl.generation.renderer import render_chart
from chart_provenance_rl.schema import (
    BenchmarkRecord,
    ChartType,
    ComparisonKind,
    EvidenceRef,
    ProgramKind,
    ProgramSpec,
    TableData,
)
from chart_provenance_rl.serialization import sha256_json


def test_decimal_canonicalization() -> None:
    assert canonical_decimal("12.3400") == "12.34"
    assert canonical_decimal("-0.00") == "0"
    assert canonical_answer("  YES ") == "yes"
    assert canonical_answer("01.500") == "1.5"
    assert parse_decimal(3) == Decimal(3)
    with pytest.raises(ValueError):
        parse_decimal("NaN")
    with pytest.raises(ValueError, match="exponent"):
        canonical_decimal("1e1000000")
    with pytest.raises(ValueError, match="256 characters"):
        parse_decimal("1" * 257)


def test_schema_round_trip() -> None:
    table = _table()
    program = ProgramSpec(
        ProgramKind.COMPARISON,
        "Revenue",
        category="Q1",
        other_category="Q2",
        comparison=ComparisonKind.EQUAL,
    )
    table_hash = sha256_json(table.to_dict())
    rendered = render_chart(table, ChartType.TABLE, "classic")
    record = BenchmarkRecord(
        record_id="a" * 64,
        split="train",
        question="Are the values equal?",
        prompt="Return JSON.",
        chart_type=ChartType.TABLE,
        renderer_family="classic",
        image_sha256="b" * 64,
        image_path=f"images/{'b' * 64}.png",
        table_sha256=table_hash,
        table=table,
        program=program,
        answer="yes",
        minimal_provenance_sets=(
            (
                EvidenceRef("Revenue", "Q1"),
                EvidenceRef("Revenue", "Q2"),
            ),
        ),
        grammar_seed=1,
        config_seed=2,
        generation_index=0,
        category_family="quarter",
        series_family="finance",
        target_series_index=0,
        render_text=rendered.text_elements,
    )
    assert BenchmarkRecord.from_dict(record.to_dict()) == record
    assert record.schema_version == "1.1"
    legacy = record.to_dict()
    legacy["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="Unsupported schema version"):
        BenchmarkRecord.from_dict(legacy)
    assert TableData.from_dict(table.to_dict()) == table
    assert ProgramSpec.from_dict(program.to_dict()) == program


def test_schema_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        EvidenceRef("", "Q1")
    with pytest.raises(ValueError):
        TableData(
            "Bad",
            ("Q1", "Q1"),
            ("Revenue",),
            ((Decimal(1), Decimal(2)),),
        )
    with pytest.raises(ValueError):
        ProgramSpec(ProgramKind.DIFFERENCE, "Revenue", category="Q1")
    with pytest.raises(TypeError):
        EvidenceRef.from_dict({"series": "Revenue", "category": 3})


def _table() -> TableData:
    return TableData(
        "Example",
        ("Q1", "Q2"),
        ("Revenue",),
        ((Decimal(10), Decimal(10)),),
    )
