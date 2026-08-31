from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
from typing import Any

DEFAULT_LOCATION = "spaincentral"
T4_SIZE = "Standard_NC4as_T4_v3"
A10_SIZE = "Standard_NV36ads_A10_v5"


def main() -> None:
    args = _parser().parse_args()
    sizes = _az_json(
        "ml",
        "compute",
        "list-sizes",
        "--location",
        args.location,
        "--subscription",
        args.subscription,
        "--resource-group",
        args.resource_group,
        "--workspace-name",
        args.workspace,
    )
    if not isinstance(sizes, list):
        raise TypeError("Azure compute size response must be a list")
    by_name = {
        str(item["name"]): item for item in sizes if isinstance(item, dict) and "name" in item
    }
    t4_types = _supported_compute_types(by_name, T4_SIZE)
    a10_types = _supported_compute_types(by_name, A10_SIZE)
    if "AmlCompute" not in t4_types:
        raise RuntimeError(f"{T4_SIZE} is not available for AmlCompute: {t4_types}")
    if "AmlCompute" in a10_types:
        raise RuntimeError(
            "A10 availability changed; revalidate quota and update the preregistered infrastructure"
        )
    compute = _az_json(
        "ml",
        "compute",
        "show",
        "--name",
        args.compute,
        "--subscription",
        args.subscription,
        "--resource-group",
        args.resource_group,
        "--workspace-name",
        args.workspace,
    )
    if not isinstance(compute, dict):
        raise TypeError("Azure compute response must be an object")
    report = {
        "workspace": args.workspace,
        "location": args.location,
        "t4": {
            "size": T4_SIZE,
            "supported_compute_types": t4_types,
            "cluster": {
                "name": compute.get("name"),
                "size": compute.get("size"),
                "min_instances": compute.get("min_instances"),
                "max_instances": compute.get("max_instances"),
                "idle_time_before_scale_down": compute.get("idle_time_before_scale_down"),
            },
        },
        "a10_blocker": {
            "size": A10_SIZE,
            "supported_compute_types": a10_types,
            "amlcompute_available": False,
            "required_cores": 36,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _supported_compute_types(
    sizes: dict[str, dict[str, Any]],
    size: str,
) -> list[str]:
    try:
        value = sizes[size]
    except KeyError as error:
        raise RuntimeError(f"Azure ML does not expose requested size {size}") from error
    types = value.get("supportedComputeTypes", value.get("supported_compute_types"))
    if isinstance(types, str):
        types = ast.literal_eval(types)
    if not isinstance(types, list) or not all(isinstance(item, str) for item in types):
        raise TypeError(f"Invalid supportedComputeTypes for {size}: {types!r}")
    return types


def _az_json(*arguments: str) -> Any:
    completed = subprocess.run(
        ["az", *arguments, "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Azure ML compute before submission.")
    parser.add_argument(
        "--subscription",
        default=os.environ.get("AZURE_SUBSCRIPTION_ID"),
        required="AZURE_SUBSCRIPTION_ID" not in os.environ,
    )
    parser.add_argument(
        "--resource-group",
        default=os.environ.get("AZURE_RESOURCE_GROUP"),
        required="AZURE_RESOURCE_GROUP" not in os.environ,
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("AZURE_ML_WORKSPACE"),
        required="AZURE_ML_WORKSPACE" not in os.environ,
    )
    parser.add_argument(
        "--compute",
        default=os.environ.get("AZURE_ML_COMPUTE", "gpu-t4-dedicated"),
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("AZURE_LOCATION", DEFAULT_LOCATION),
    )
    return parser


if __name__ == "__main__":
    main()
