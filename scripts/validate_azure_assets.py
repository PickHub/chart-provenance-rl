from __future__ import annotations

from pathlib import Path

from azure.ai.ml import load_component, load_compute, load_environment, load_job

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_environment(source=ROOT / "azure/environment/environment.yml")
    load_compute(source=ROOT / "azure/compute/t4.yml")
    for path in sorted((ROOT / "azure/components").glob("*.yml")):
        load_component(source=path)
    load_job(source=ROOT / "azure/pipeline.yml")
    print("Azure ML assets loaded successfully")


if __name__ == "__main__":
    main()
