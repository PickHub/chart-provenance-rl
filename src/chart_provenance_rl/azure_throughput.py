from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from chart_provenance_rl.evaluation.provenance import is_sha256
from chart_provenance_rl.serialization import (
    load_json,
    sha256_file,
    sha256_json,
    write_json,
)
from chart_provenance_rl.training.artifacts import (
    ADAPTER_ARTIFACT_MANIFEST,
    validate_adapter_artifact,
)
from chart_provenance_rl.training.common import RUN_MANIFEST

REQUIRED_SEEDS = frozenset((17, 29, 43))
REQUIRED_STEPS = {
    "sft": ("sft", None),
    "grpo_naive": ("grpo", "naive_recall"),
    "grpo_minimal": ("grpo", "minimal_provenance"),
}
EXPECTED_COMPUTE_SIZE = "Standard_NC4as_T4_v3"
EXPECTED_MAX_STEPS = 100


class AzureEvidenceClient(Protocol):
    def list_jobs(self, parent_job_name: str) -> list[dict[str, Any]]: ...

    def show_job(self, name: str) -> dict[str, Any]: ...

    def show_compute(self) -> dict[str, Any]: ...

    def download_job(self, name: str, destination: Path) -> None: ...


class AzureCliEvidenceClient:
    def __init__(
        self,
        *,
        subscription: str,
        resource_group: str,
        workspace: str,
        compute: str,
    ) -> None:
        self.scope = (
            "--subscription",
            subscription,
            "--resource-group",
            resource_group,
            "--workspace-name",
            workspace,
        )
        self.compute = compute

    def list_jobs(self, parent_job_name: str) -> list[dict[str, Any]]:
        value = self._json(
            "ml",
            "job",
            "list",
            "--parent-job-name",
            parent_job_name,
            "--all-results",
            "true",
        )
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise TypeError("Azure job list response must contain objects")
        return value

    def show_job(self, name: str) -> dict[str, Any]:
        value = self._json("ml", "job", "show", "--name", name)
        if not isinstance(value, dict):
            raise TypeError(f"Azure job {name} response must be an object")
        return value

    def show_compute(self) -> dict[str, Any]:
        value = self._json(
            "ml",
            "compute",
            "show",
            "--name",
            self.compute,
        )
        if not isinstance(value, dict):
            raise TypeError("Azure compute response must be an object")
        return value

    def download_job(self, name: str, destination: Path) -> None:
        subprocess.run(
            [
                "az",
                "ml",
                "job",
                "download",
                "--name",
                name,
                "--download-path",
                str(destination),
                "--all",
                *self.scope,
                "--output",
                "none",
            ],
            check=True,
        )

    def _json(self, *arguments: str) -> Any:
        completed = subprocess.run(
            ["az", *arguments, *self.scope, "--output", "json"],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)


def collect_azure_throughput(
    client: AzureEvidenceClient,
    pipeline_jobs: list[str],
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    if len(pipeline_jobs) != 3 or len(set(pipeline_jobs)) != 3:
        raise ValueError("Exactly three distinct seed pipeline jobs are required")
    compute = client.show_compute()
    compute_name = _required_string(compute, "name")
    compute_size = _required_string(compute, "size")
    if compute_size != EXPECTED_COMPUTE_SIZE:
        raise ValueError(f"Expected {EXPECTED_COMPUTE_SIZE}, received {compute_size}")
    pipeline_evidence: list[dict[str, Any]] = []
    run_evidence: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="chart-provenance-azure-evidence-") as temporary:
        download_root = Path(temporary)
        for parent_name in pipeline_jobs:
            parent = client.show_job(parent_name)
            if str(parent.get("type", "")).casefold() != "pipeline":
                raise ValueError(f"Azure job {parent_name} is not a pipeline")
            if _normalized_status(parent) != "completed":
                raise ValueError(f"Pipeline job {parent_name} is not completed")
            seed = _pipeline_seed(parent)
            children = client.list_jobs(parent_name)
            pipeline_evidence.append(
                {
                    "name": parent_name,
                    "status": "completed",
                    "seed": seed,
                }
            )
            for node_name, (stage, reward_arm) in REQUIRED_STEPS.items():
                child_summary = _find_child(children, node_name)
                child_name = _required_string(child_summary, "name")
                child = client.show_job(child_name)
                _validate_child_job(
                    child,
                    expected_compute=compute_name,
                    node_name=node_name,
                )
                destination = download_root / child_name
                destination.mkdir()
                client.download_job(child_name, destination)
                artifact_dir = _find_adapter_artifact(destination)
                run_evidence.append(
                    _run_evidence(
                        parent_name=parent_name,
                        child_name=child_name,
                        node_name=node_name,
                        child=child,
                        artifact_dir=artifact_dir,
                        seed=seed,
                        expected_stage=stage,
                        expected_reward_arm=reward_arm,
                        compute_name=compute_name,
                        compute_size=compute_size,
                    )
                )
    _validate_collected_runs(run_evidence)
    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "evidence_source": "azure_ml_v2_cli",
        "compute": {
            "name": compute_name,
            "size": compute_size,
        },
        "pipeline_jobs": sorted(
            pipeline_evidence,
            key=lambda item: int(item["seed"]),
        ),
        "source_sha256": _single_hash(run_evidence, "source_sha256"),
        "git_commit": _single_git_commit(run_evidence),
        "dataset_manifest_sha256": _single_hash(
            run_evidence,
            "dataset_manifest_sha256",
        ),
        "runs": sorted(
            run_evidence,
            key=lambda item: (int(item["seed"]), str(item["stage"])),
        ),
    }
    evidence["collection_sha256"] = sha256_json(evidence)
    if output is not None:
        write_json(output, evidence)
    return evidence


def _run_evidence(
    *,
    parent_name: str,
    child_name: str,
    node_name: str,
    child: dict[str, Any],
    artifact_dir: Path,
    seed: int,
    expected_stage: str,
    expected_reward_arm: str | None,
    compute_name: str,
    compute_size: str,
) -> dict[str, Any]:
    manifest_path = artifact_dir / RUN_MANIFEST
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError("Downloaded run manifest must be an object")
    immutable = manifest.get("immutable")
    if not isinstance(immutable, dict):
        raise TypeError("Downloaded run manifest is missing immutable metadata")
    if immutable.get("stage") != expected_stage:
        raise ValueError(f"{node_name} run manifest stage mismatch")
    if immutable.get("seed") != seed:
        raise ValueError(f"{node_name} run manifest seed mismatch")
    if immutable.get("max_steps") != EXPECTED_MAX_STEPS:
        raise ValueError(f"{node_name} run manifest max_steps mismatch")
    if immutable.get("reward_arm") != expected_reward_arm:
        raise ValueError(f"{node_name} run manifest reward arm mismatch")
    source_sha256 = immutable.get("source_sha256")
    dataset_sha256 = immutable.get("dataset_manifest_sha256")
    training_config = immutable.get("training_config")
    if not is_sha256(source_sha256) or not is_sha256(dataset_sha256):
        raise ValueError(f"{node_name} run manifest hashes are invalid")
    if not isinstance(training_config, dict):
        raise TypeError(f"{node_name} training config must be an object")
    start = _job_time(child, "start")
    end = _job_time(child, "end")
    wall_seconds = (end - start).total_seconds()
    if wall_seconds <= 0:
        raise ValueError(f"{node_name} job duration must be positive")
    stage = expected_reward_arm or expected_stage
    properties = child.get("properties")
    if not isinstance(properties, dict):
        raise TypeError(f"{node_name} child properties must be an object")
    git_commit = properties.get("mlflow.source.git.commit")
    if not isinstance(git_commit, str) or not _is_git_commit(git_commit):
        raise ValueError(f"{node_name} child job git commit is invalid")
    return {
        "seed": seed,
        "stage": stage,
        "status": "completed",
        "steps": EXPECTED_MAX_STEPS,
        "max_steps": EXPECTED_MAX_STEPS,
        "wall_seconds": wall_seconds,
        "parent_job_name": parent_name,
        "job_name": child_name,
        "component_name": node_name,
        "compute": compute_name,
        "compute_size": compute_size,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "source_sha256": source_sha256,
        "git_commit": git_commit,
        "dataset_manifest_sha256": dataset_sha256,
        "training_config_sha256": sha256_json(training_config),
        "run_manifest_sha256": sha256_file(manifest_path),
        "adapter_artifact_sha256": validate_adapter_artifact(artifact_dir),
    }


def _validate_child_job(
    child: dict[str, Any],
    *,
    expected_compute: str,
    node_name: str,
) -> None:
    if _normalized_status(child) != "completed":
        raise ValueError(f"Child job {node_name} is not completed")
    compute = child.get("compute")
    if isinstance(compute, dict):
        compute = compute.get("name") or compute.get("id")
    if not isinstance(compute, str) or not (
        compute == expected_compute
        or compute == f"azureml:{expected_compute}"
        or compute.rstrip("/").endswith(f"/computes/{expected_compute}")
    ):
        raise ValueError(f"Child job {node_name} used unexpected compute")
    properties = child.get("properties")
    if not isinstance(properties, dict):
        raise TypeError(f"Child job {node_name} properties must be an object")
    cluster_name = properties.get("_azureml.ClusterName")
    compute_type = properties.get("ComputeTargetType")
    pipeline_property = properties.get("azureml.pipeline")
    if cluster_name != expected_compute:
        raise ValueError(f"Child job {node_name} cluster property mismatch")
    if str(compute_type).casefold() != "amlcompute":
        raise ValueError(f"Child job {node_name} is not AmlCompute")
    if not isinstance(pipeline_property, str) or not pipeline_property:
        raise ValueError(f"Child job {node_name} lacks pipeline provenance")


def _validate_collected_runs(runs: list[dict[str, Any]]) -> None:
    expected_pairs = {
        (seed, stage)
        for seed in REQUIRED_SEEDS
        for stage in ("sft", "naive_recall", "minimal_provenance")
    }
    observed_pairs = {(int(run["seed"]), str(run["stage"])) for run in runs}
    if observed_pairs != expected_pairs:
        raise ValueError(f"Azure evidence does not cover required runs: {observed_pairs}")
    for stage in ("sft", "naive_recall", "minimal_provenance"):
        config_hashes = {run["training_config_sha256"] for run in runs if run["stage"] == stage}
        if len(config_hashes) != 1:
            raise ValueError(f"{stage} training config differs across seeds")
    for seed in REQUIRED_SEEDS:
        grpo_configs = {
            run["training_config_sha256"]
            for run in runs
            if run["seed"] == seed and run["stage"] in {"naive_recall", "minimal_provenance"}
        }
        if len(grpo_configs) != 1:
            raise ValueError(f"Naive/minimal training config differs for seed {seed}")


def _find_child(
    children: list[dict[str, Any]],
    node_name: str,
) -> dict[str, Any]:
    matches = [child for child in children if node_name in _job_labels(child)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {node_name} child job, received {len(matches)}")
    return matches[0]


def _job_labels(job: dict[str, Any]) -> set[str]:
    labels = {
        value
        for value in (
            job.get("name"),
            job.get("display_name"),
            job.get("displayName"),
        )
        if isinstance(value, str)
    }
    for container_name in ("properties", "tags"):
        container = job.get(container_name)
        if isinstance(container, dict):
            labels.update(
                value
                for key, value in container.items()
                if isinstance(value, str)
                and ("node" in str(key).casefold() or "display" in str(key).casefold())
            )
    return {_normalize_label(label) for label in labels}


def _normalize_label(value: str) -> str:
    return value.casefold().replace("-", "_").replace(" ", "_")


def _pipeline_seed(job: dict[str, Any]) -> int:
    inputs = job.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("Pipeline job inputs must be an object")
    seed = inputs.get("seed")
    if isinstance(seed, dict):
        seed = seed.get("value")
    if isinstance(seed, str) and seed.isdigit():
        seed = int(seed)
    if not isinstance(seed, int) or seed not in REQUIRED_SEEDS:
        raise ValueError(f"Pipeline job seed is invalid: {seed!r}")
    return seed


def _normalized_status(job: dict[str, Any]) -> str:
    status = job.get("status")
    if not isinstance(status, str):
        raise TypeError("Azure job status must be a string")
    return status.casefold()


def _job_time(job: dict[str, Any], boundary: str) -> datetime:
    keys = ("start_time", "startTime") if boundary == "start" else ("end_time", "endTime")
    value = next(
        (job.get(key) for key in keys if isinstance(job.get(key), str)),
        None,
    )
    if value is None:
        creation_context = job.get("creation_context")
        if isinstance(creation_context, dict):
            context_keys = (
                ("created_at", "createdAt")
                if boundary == "start"
                else ("last_modified_at", "lastModifiedAt")
            )
            value = next(
                (
                    creation_context.get(key)
                    for key in context_keys
                    if isinstance(creation_context.get(key), str)
                ),
                None,
            )
    if not isinstance(value, str):
        raise ValueError(f"Azure job is missing {boundary} time")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Azure job {boundary} time must include a timezone")
    return parsed


def _find_adapter_artifact(download_root: Path) -> Path:
    manifests = list(download_root.rglob(ADAPTER_ARTIFACT_MANIFEST))
    if len(manifests) != 1:
        raise ValueError("Downloaded job must contain exactly one adapter artifact manifest")
    artifact_dir = manifests[0].parent
    if not (artifact_dir / RUN_MANIFEST).is_file():
        raise ValueError("Downloaded adapter is missing its run manifest")
    return artifact_dir


def _single_hash(runs: list[dict[str, Any]], field: str) -> str:
    values = {run.get(field) for run in runs}
    if len(values) != 1:
        raise ValueError(f"Collected runs disagree on {field}")
    value = values.pop()
    if not isinstance(value, str) or not is_sha256(value):
        raise ValueError(f"Collected run {field} is invalid")
    return value


def _single_git_commit(runs: list[dict[str, Any]]) -> str:
    values = {run.get("git_commit") for run in runs}
    if len(values) != 1:
        raise ValueError("Collected runs disagree on git commit")
    value = values.pop()
    if not isinstance(value, str) or not _is_git_commit(value):
        raise ValueError("Collected run git commit is invalid")
    return value


def _is_git_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _required_string(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise TypeError(f"Azure evidence {field} must be a non-empty string")
    return result
