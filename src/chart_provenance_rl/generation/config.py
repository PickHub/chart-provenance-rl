from __future__ import annotations

from dataclasses import dataclass

from chart_provenance_rl.schema import ChartType, ProgramKind


@dataclass(frozen=True)
class SplitSpec:
    name: str
    count: int
    grammar_seed: int
    config_seed: int
    programs: tuple[ProgramKind, ...]
    chart_types: tuple[ChartType, ...]
    renderer_families: tuple[str, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetConfig:
    master_seed: int
    splits: tuple[SplitSpec, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "master_seed": self.master_seed,
            "splits": [
                {
                    "name": split.name,
                    "count": split.count,
                    "grammar_seed": split.grammar_seed,
                    "config_seed": split.config_seed,
                    "programs": [program.value for program in split.programs],
                    "chart_types": [chart_type.value for chart_type in split.chart_types],
                    "renderer_families": list(split.renderer_families),
                    "tags": list(split.tags),
                }
                for split in self.splits
            ],
        }


ATOMIC_PROGRAMS = (
    ProgramKind.LOOKUP,
    ProgramKind.COMPARISON,
    ProgramKind.ARGMAX,
    ProgramKind.COUNT,
)
COMPOSITION_PROGRAMS = (ProgramKind.DIFFERENCE, ProgramKind.SUM_SELECTED)
ALL_PROGRAMS = ATOMIC_PROGRAMS + COMPOSITION_PROGRAMS
STANDARD_RENDERERS = ("classic", "pastel", "high_contrast")
STANDARD_CHARTS = (ChartType.BAR, ChartType.LINE, ChartType.TABLE)


def default_config() -> DatasetConfig:
    return _config_with_counts(
        {
            "train": 96,
            "validation": 24,
            "test_composition": 48,
            "stress_renderer_holdout": 24,
            "stress_duplicate_decoy": 24,
            "stress_label_shuffle": 24,
            "stress_order_perturbation": 24,
            "stress_citation_probe": 24,
        }
    )


def smoke_config() -> DatasetConfig:
    return _config_with_counts(
        {
            "train": 8,
            "validation": 4,
            "test_composition": 4,
            "stress_renderer_holdout": 3,
            "stress_duplicate_decoy": 3,
            "stress_label_shuffle": 3,
            "stress_order_perturbation": 3,
            "stress_citation_probe": 3,
        }
    )


def _config_with_counts(counts: dict[str, int]) -> DatasetConfig:
    return DatasetConfig(
        master_seed=174_221,
        splits=(
            SplitSpec(
                "train",
                counts["train"],
                11_003,
                21_007,
                ATOMIC_PROGRAMS,
                STANDARD_CHARTS,
                STANDARD_RENDERERS,
            ),
            SplitSpec(
                "validation",
                counts["validation"],
                31_019,
                41_021,
                ATOMIC_PROGRAMS,
                STANDARD_CHARTS,
                STANDARD_RENDERERS,
            ),
            SplitSpec(
                "test_composition",
                counts["test_composition"],
                51_031,
                61_033,
                COMPOSITION_PROGRAMS,
                STANDARD_CHARTS,
                STANDARD_RENDERERS,
                ("primary_ood", "held_out_program_composition"),
            ),
            SplitSpec(
                "stress_renderer_holdout",
                counts["stress_renderer_holdout"],
                71_039,
                81_043,
                ALL_PROGRAMS,
                STANDARD_CHARTS,
                ("monochrome",),
                ("renderer_holdout",),
            ),
            SplitSpec(
                "stress_duplicate_decoy",
                counts["stress_duplicate_decoy"],
                91_049,
                101_051,
                ALL_PROGRAMS,
                STANDARD_CHARTS,
                STANDARD_RENDERERS,
                ("duplicate_value_decoy",),
            ),
            SplitSpec(
                "stress_label_shuffle",
                counts["stress_label_shuffle"],
                111_061,
                121_063,
                ALL_PROGRAMS,
                STANDARD_CHARTS,
                STANDARD_RENDERERS,
                ("label_shuffle",),
            ),
            SplitSpec(
                "stress_order_perturbation",
                counts["stress_order_perturbation"],
                131_071,
                141_073,
                ALL_PROGRAMS,
                STANDARD_CHARTS,
                STANDARD_RENDERERS,
                ("category_series_order_perturbation",),
            ),
            SplitSpec(
                "stress_citation_probe",
                counts["stress_citation_probe"],
                151_079,
                161_083,
                ALL_PROGRAMS,
                STANDARD_CHARTS,
                STANDARD_RENDERERS,
                ("answer_correct_citation_wrong_probe",),
            ),
        ),
    )
