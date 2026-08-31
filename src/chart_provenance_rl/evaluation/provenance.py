from __future__ import annotations

from typing import Any

PROVENANCE_FIELDS = (
    "adapter_artifact_sha256",
    "run_manifest_sha256",
    "run_max_steps",
    "adapter_stage",
    "adapter_seed",
    "adapter_reward_arm",
    "sft_adapter_sha256",
    "dataset_manifest_sha256",
    "image_mode",
)


def expected_prediction_provenance(
    method: str,
) -> dict[str, str | None]:
    expected = {
        "prompted_frozen": ("prompted_frozen", None, "normal"),
        "sft": ("sft", None, "normal"),
        "sft_json_repair": ("sft", None, "normal"),
        "grpo_answer_only": ("grpo", "answer_only", "normal"),
        "grpo_naive_recall": ("grpo", "naive_recall", "normal"),
        "grpo_minimal_provenance": (
            "grpo",
            "minimal_provenance",
            "normal",
        ),
        "grpo_minimal_question_only": (
            "grpo",
            "minimal_provenance",
            "blank",
        ),
        "grpo_minimal_image_shuffle": (
            "grpo",
            "minimal_provenance",
            "shuffled",
        ),
    }
    try:
        adapter_stage, adapter_reward_arm, image_mode = expected[method]
    except KeyError as error:
        raise ValueError(f"Unsupported evaluation method: {method}") from error
    return {
        "adapter_stage": adapter_stage,
        "adapter_reward_arm": adapter_reward_arm,
        "image_mode": image_mode,
    }


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
