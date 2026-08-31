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

Changes to rewards, splits, statistics, model revision, or training budgets require a
new preregistration version. Do not add model results without the run manifest,
question-level outputs, environment version, source commit, and Azure job IDs.
