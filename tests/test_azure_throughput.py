from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chart_provenance_rl.azure_throughput import (
    AzureCliEvidenceClient,
    _find_adapter_artifact,
    _job_time,
    _pipeline_seed,
    _single_hash,
    _validate_child_job,
    collect_azure_throughput,
)
from chart_provenance_rl.serialization import load_json, sha256_json, write_json
from chart_provenance_rl.training.artifacts import (
    write_adapter_artifact_manifest,
)

SEEDS = (17, 29, 43)
NODES = {
    "sft": ("sft", None),
    "grpo_naive": ("grpo", "naive_recall"),
    "grpo_minimal": ("grpo", "minimal_provenance"),
}


def test_collector_binds_live_jobs_compute_times_and_artifacts(
    tmp_path: Path,
) -> None:
    client = _FakeAzureClient(tmp_path)
    output = tmp_path / "throughput.json"
    report = collect_azure_throughput(
        client,
        [f"pipeline-{seed}" for seed in SEEDS],
        output=output,
    )
    assert report["evidence_source"] == "azure_ml_v2_cli"
    assert report["compute"] == {
        "name": "gpu-t4-dedicated",
        "size": "Standard_NC4as_T4_v3",
    }
    assert {item["seed"] for item in report["pipeline_jobs"]} == set(SEEDS)
    assert len(report["runs"]) == 9
    assert all(run["wall_seconds"] == 3600 for run in report["runs"])
    assert all(run["source_sha256"] == "a" * 64 for run in report["runs"])
    assert all(run["dataset_manifest_sha256"] == "b" * 64 for run in report["runs"])
    unsigned = {key: value for key, value in report.items() if key != "collection_sha256"}
    assert report["collection_sha256"] == sha256_json(unsigned)
    assert json.loads(output.read_text()) == report
    schema = load_json(
        Path(__file__).resolve().parents[1] / "schemas" / "t4-throughput.schema.json"
    )
    assert set(schema["required"]).issubset(report)
    assert "hardware" not in report


def test_collector_fails_on_missing_or_nonterminal_azure_evidence(
    tmp_path: Path,
) -> None:
    client = _FakeAzureClient(tmp_path)
    client.jobs["pipeline-29"]["status"] = "Running"
    with pytest.raises(ValueError, match="not completed"):
        collect_azure_throughput(
            client,
            [f"pipeline-{seed}" for seed in SEEDS],
        )
    client.jobs["pipeline-29"]["status"] = "Completed"
    client.children["pipeline-29"] = [
        child for child in client.children["pipeline-29"] if child["display_name"] != "grpo_minimal"
    ]
    with pytest.raises(ValueError, match="one grpo_minimal child"):
        collect_azure_throughput(
            client,
            [f"pipeline-{seed}" for seed in SEEDS],
        )


def test_collector_rejects_compute_and_config_drift(tmp_path: Path) -> None:
    client = _FakeAzureClient(tmp_path)
    client.compute["size"] = "Standard_DS3_v2"
    with pytest.raises(ValueError, match="Expected Standard_NC4as_T4_v3"):
        collect_azure_throughput(
            client,
            [f"pipeline-{seed}" for seed in SEEDS],
        )
    client.compute["size"] = "Standard_NC4as_T4_v3"
    manifest_path = client.artifacts["grpo_naive-29"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["immutable"]["training_config"]["temperature"] = 0.5
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="config differs across seeds"):
        collect_azure_throughput(
            client,
            [f"pipeline-{seed}" for seed in SEEDS],
        )


def test_collector_requires_exact_three_pipelines(tmp_path: Path) -> None:
    client = _FakeAzureClient(tmp_path)
    with pytest.raises(ValueError, match="Exactly three distinct"):
        collect_azure_throughput(
            client,
            ["pipeline-17", "pipeline-29"],
        )


def test_collector_rejects_pipeline_child_and_manifest_mismatch(
    tmp_path: Path,
) -> None:
    client = _FakeAzureClient(tmp_path / "pipeline")
    client.jobs["pipeline-17"]["type"] = "command"
    with pytest.raises(ValueError, match="not a pipeline"):
        collect_azure_throughput(
            client,
            [f"pipeline-{seed}" for seed in SEEDS],
        )

    client = _FakeAzureClient(tmp_path / "child")
    client.jobs["sft-17"]["properties"]["_azureml.ClusterName"] = "other"
    with pytest.raises(ValueError, match="cluster property mismatch"):
        collect_azure_throughput(
            client,
            [f"pipeline-{seed}" for seed in SEEDS],
        )

    client = _FakeAzureClient(tmp_path / "manifest")
    manifest_path = client.artifacts["grpo_minimal-17"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["immutable"]["stage"] = "sft"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="stage mismatch"):
        collect_azure_throughput(
            client,
            [f"pipeline-{seed}" for seed in SEEDS],
        )


def test_azure_cli_queries_children_by_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        calls.append(command)
        if "download" in command:
            return SimpleNamespace(stdout="")
        if "compute" in command:
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "name": "compute",
                        "size": "Standard_NC4as_T4_v3",
                    }
                )
            )
        if "show" in command:
            return SimpleNamespace(stdout='{"name":"job"}')
        return SimpleNamespace(stdout="[]")

    monkeypatch.setattr(
        "chart_provenance_rl.azure_throughput.subprocess.run",
        fake_run,
    )
    client = AzureCliEvidenceClient(
        subscription="subscription",
        resource_group="group",
        workspace="workspace",
        compute="compute",
    )
    assert client.list_jobs("parent") == []
    assert client.show_job("job") == {"name": "job"}
    assert client.show_compute()["name"] == "compute"
    client.download_job("job", Path("/tmp/download"))
    command = calls[0]
    assert command[command.index("--parent-job-name") + 1] == "parent"
    assert command[command.index("--all-results") + 1] == "true"


def test_azure_cli_rejects_wrong_response_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(("{}", "[]", "[]"))

    def fake_run(_command: list[str], **_kwargs: Any) -> Any:
        return SimpleNamespace(stdout=next(responses))

    monkeypatch.setattr(
        "chart_provenance_rl.azure_throughput.subprocess.run",
        fake_run,
    )
    client = AzureCliEvidenceClient(
        subscription="subscription",
        resource_group="group",
        workspace="workspace",
        compute="compute",
    )
    with pytest.raises(TypeError, match="job list"):
        client.list_jobs("parent")
    with pytest.raises(TypeError, match="job job"):
        client.show_job("job")
    with pytest.raises(TypeError, match="compute response"):
        client.show_compute()


def test_azure_metadata_parsers_fail_closed(tmp_path: Path) -> None:
    assert _pipeline_seed({"inputs": {"seed": {"value": "17"}}}) == 17
    with pytest.raises(ValueError, match="seed is invalid"):
        _pipeline_seed({"inputs": {"seed": {"value": "999"}}})
    with pytest.raises(TypeError, match="inputs"):
        _pipeline_seed({})

    job = {
        "status": "Completed",
        "compute": "azureml:gpu-t4-dedicated",
        "start_time": "2026-08-31T00:00:00Z",
        "end_time": "2026-08-31T01:00:00Z",
        "properties": {
            "_azureml.ClusterName": "gpu-t4-dedicated",
            "ComputeTargetType": "amlcompute",
            "azureml.pipeline": "pipeline",
        },
    }
    assert _job_time(job, "end") > _job_time(job, "start")
    _validate_child_job(
        job,
        expected_compute="gpu-t4-dedicated",
        node_name="sft",
    )
    with pytest.raises(ValueError, match="not completed"):
        _validate_child_job(
            {**job, "status": "Failed"},
            expected_compute="gpu-t4-dedicated",
            node_name="sft",
        )
    with pytest.raises(ValueError, match="unexpected compute"):
        _validate_child_job(
            {**job, "compute": "azureml:other"},
            expected_compute="gpu-t4-dedicated",
            node_name="sft",
        )
    with pytest.raises(ValueError, match="include a timezone"):
        _job_time({**job, "start_time": "2026-08-31T00:00:00"}, "start")
    with pytest.raises(ValueError, match="missing start time"):
        _job_time({"creation_context": {}}, "start")
    with pytest.raises(ValueError, match="exactly one adapter"):
        _find_adapter_artifact(tmp_path)
    with pytest.raises(ValueError, match="disagree"):
        _single_hash(
            [{"hash": "a" * 64}, {"hash": "b" * 64}],
            "hash",
        )


class _FakeAzureClient:
    def __init__(self, root: Path) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.children: dict[str, list[dict[str, Any]]] = {}
        self.artifacts: dict[str, Path] = {}
        self.compute = {
            "name": "gpu-t4-dedicated",
            "size": "Standard_NC4as_T4_v3",
        }
        for seed in SEEDS:
            parent_name = f"pipeline-{seed}"
            self.jobs[parent_name] = {
                "name": parent_name,
                "type": "pipeline",
                "status": "Completed",
                "inputs": {"seed": {"value": seed}},
            }
            self.children[parent_name] = []
            for node_name, (stage, reward_arm) in NODES.items():
                child_name = f"{node_name}-{seed}"
                summary = {
                    "name": child_name,
                    "display_name": node_name,
                }
                self.children[parent_name].append(summary)
                self.jobs[child_name] = {
                    **summary,
                    "status": "Completed",
                    "compute": "azureml:gpu-t4-dedicated",
                    "creation_context": {
                        "created_at": "2026-08-31T00:00:00+00:00",
                        "last_modified_at": "2026-08-31T01:00:00+00:00",
                    },
                    "properties": {
                        "_azureml.ClusterName": "gpu-t4-dedicated",
                        "ComputeTargetType": "amlcompute",
                        "azureml.pipeline": parent_name,
                        "mlflow.source.git.commit": "c" * 40,
                    },
                }
                self.artifacts[child_name] = _artifact(
                    root / "artifacts" / child_name,
                    stage=stage,
                    reward_arm=reward_arm,
                    seed=seed,
                )

    def list_jobs(self, parent_job_name: str) -> list[dict[str, Any]]:
        return self.children[parent_job_name]

    def show_job(self, name: str) -> dict[str, Any]:
        return self.jobs[name]

    def show_compute(self) -> dict[str, Any]:
        return self.compute

    def download_job(self, name: str, destination: Path) -> None:
        shutil.copytree(
            self.artifacts[name],
            destination,
            dirs_exist_ok=True,
        )


def _artifact(
    root: Path,
    *,
    stage: str,
    reward_arm: str | None,
    seed: int,
) -> Path:
    root.mkdir(parents=True)
    (root / "adapter_config.json").write_text("{}")
    (root / "adapter_model.safetensors").write_bytes(f"{stage}-{reward_arm}-{seed}".encode())
    write_json(
        root / "run_manifest.json",
        {
            "immutable": {
                "stage": stage,
                "seed": seed,
                "max_steps": 100,
                "reward_arm": reward_arm,
                "source_sha256": "a" * 64,
                "dataset_manifest_sha256": "b" * 64,
                "training_config": (
                    {"loss_type": "sft"} if stage == "sft" else {"loss_type": "dr_grpo"}
                ),
            },
            "trainable_parameters": {
                "count": 1,
                "names": ["adapter"],
            },
        },
    )
    write_adapter_artifact_manifest(root)
    return root
