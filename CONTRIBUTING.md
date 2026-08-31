# Contributing

Use Python 3.12 and uv. Keep generated release artifacts out of `data/` and place them
under `artifacts/releases/`.

```bash
uv sync --frozen --group dev
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run mypy src tests
uv run pytest
uv build
```

The pinned TRL source-contract test can run on non-CUDA hosts without bitsandbytes:

```bash
uv run \
  --with 'trl==0.26.2' \
  --with 'transformers==4.57.3' \
  --with 'torch==2.8.0' \
  pytest -q tests/test_trl_contract.py -o addopts=''
```

Changes to rewards, splits, statistics, model revision, or training budgets require a
new preregistration version. Do not add model results without the run manifest,
question-level outputs, environment version, source commit, and Azure job IDs.
