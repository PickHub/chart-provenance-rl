import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_azure_compute_matches_live_validated_t4() -> None:
    compute = _yaml(ROOT / "azure/compute/t4.yml")
    assert compute["name"] == "gpu-t4-dedicated"
    assert compute["size"] == "Standard_NC4as_T4_v3"
    assert compute["location"] == "spaincentral"
    assert compute["min_instances"] == 0
    assert compute["max_instances"] == 1
    assert compute["idle_time_before_scale_down"] == 300
    assert compute["ssh_public_access_enabled"] is False
    azure_text = "\n".join(
        path.read_text() for path in (ROOT / "azure").rglob("*.yml") if path.name != "pipeline.yml"
    )
    assert "Standard_NV36ads_A10_v5" not in azure_text


def test_azure_environment_uses_immutable_image_and_locked_stack() -> None:
    environment = _yaml(ROOT / "azure/environment/environment.yml")
    assert environment["image"].endswith(
        "@sha256:2f88524bcaaaeecd457236fe6ab35965a780640449402c57aab14de91f5569b9"
    )
    conda = _yaml(ROOT / "azure/environment/conda.yml")
    pip_dependencies = conda["dependencies"][2]["pip"]
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    gpu_dependencies = pyproject["project"]["optional-dependencies"]["gpu"]
    assert set(pip_dependencies) >= set(gpu_dependencies)


def test_pipeline_contains_matched_arms_and_all_baselines() -> None:
    pipeline = _yaml(ROOT / "azure/pipeline.yml")
    jobs = pipeline["jobs"]
    assert jobs["grpo_answer"]["inputs"]["sft_adapter"] == ("${{parent.jobs.sft.outputs.adapter}}")
    assert jobs["grpo_naive"]["inputs"]["sft_adapter"] == ("${{parent.jobs.sft.outputs.adapter}}")
    assert jobs["grpo_minimal"]["inputs"]["sft_adapter"] == ("${{parent.jobs.sft.outputs.adapter}}")
    assert jobs["score_sft_repaired"]["inputs"]["repair"] is True
    assert jobs["score_prompted"]["inputs"]["method"] == "prompted_frozen"
    assert jobs["score_grpo_answer"]["inputs"]["method"] == "grpo_answer_only"
    assert jobs["score_grpo_naive"]["inputs"]["method"] == "grpo_naive_recall"
    assert jobs["score_grpo_minimal"]["inputs"]["method"] == ("grpo_minimal_provenance")


def test_ci_actions_are_immutable_and_dependabot_covers_both_ecosystems() -> None:
    workflow = _yaml(ROOT / ".github/workflows/ci.yml")
    uses = [step["uses"] for step in workflow["jobs"]["test"]["steps"] if "uses" in step]
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", value) for value in uses)
    dependabot = _yaml(ROOT / ".github/dependabot.yml")
    assert {update["package-ecosystem"] for update in dependabot["updates"]} == {
        "pip",
        "github-actions",
    }


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML object")
    return value
