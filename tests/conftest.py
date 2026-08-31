from pathlib import Path

import pytest

from chart_provenance_rl.audits import load_dataset_records
from chart_provenance_rl.generation.config import default_config, smoke_config
from chart_provenance_rl.generation.generator import generate_dataset
from chart_provenance_rl.schema import BenchmarkRecord


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("benchmark") / "data"
    generate_dataset(root, smoke_config())
    return root


@pytest.fixture(scope="session")
def records(dataset_root: Path) -> list[BenchmarkRecord]:
    records_by_split = load_dataset_records(dataset_root)
    return [record for split_records in records_by_split.values() for record in split_records]


@pytest.fixture(scope="session")
def full_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("full-benchmark") / "data"
    generate_dataset(root, default_config())
    return root
