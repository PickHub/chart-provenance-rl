from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from chart_provenance_rl.evaluation.parsing import completion_to_text
from chart_provenance_rl.evaluation.rewards import RewardArm, score_reward
from chart_provenance_rl.serialization import canonical_json
from chart_provenance_rl.training.data import (
    parse_evidence_refs,
    parse_provenance_sets,
)


@dataclass
class TrainingReward:
    arm: RewardArm
    rollout_log: Path | None = None

    @property
    def __name__(self) -> str:
        return f"{self.arm.value}_reward"

    def __call__(
        self,
        *,
        prompts: list[Any],
        completions: list[Any],
        completion_ids: list[Any],
        expected_answer: list[str],
        minimal_provenance_json: list[str],
        valid_evidence_json: list[str],
        record_id: list[str],
        trainer_state: Any,
        **kwargs: Any,
    ) -> list[float]:
        del prompts, completion_ids, kwargs
        lengths = {
            len(completions),
            len(expected_answer),
            len(minimal_provenance_json),
            len(valid_evidence_json),
            len(record_id),
        }
        if len(lengths) != 1:
            raise ValueError("TRL reward columns must align with completions")
        selected_rewards: list[float] = []
        rollout_rows: list[dict[str, Any]] = []
        for completion, answer, provenance_json, evidence_json, question_id in zip(
            completions,
            expected_answer,
            minimal_provenance_json,
            valid_evidence_json,
            record_id,
            strict=True,
        ):
            raw_output = completion_to_text(completion)
            provenance_sets = parse_provenance_sets(provenance_json)
            visible_evidence = parse_evidence_refs(evidence_json)
            selected = score_reward(
                raw_output,
                answer,
                provenance_sets,
                self.arm,
                valid_evidence_refs=visible_evidence,
            )
            hidden = score_reward(
                raw_output,
                answer,
                provenance_sets,
                RewardArm.MINIMAL_PROVENANCE,
                valid_evidence_refs=visible_evidence,
            )
            naive = score_reward(
                raw_output,
                answer,
                provenance_sets,
                RewardArm.NAIVE_RECALL,
                valid_evidence_refs=visible_evidence,
            )
            selected_rewards.append(selected.reward)
            rollout_rows.append(
                {
                    "global_step": int(trainer_state.global_step),
                    "record_id": question_id,
                    "reward_arm": self.arm.value,
                    "selected_reward": selected.reward,
                    "naive_reward": naive.reward,
                    "hidden_reward": hidden.reward,
                    "answer_exact": hidden.answer_exact,
                    "evidence_precision": hidden.evidence.precision,
                    "evidence_recall": hidden.evidence.recall,
                    "evidence_f1": hidden.evidence.f1,
                    "joint_exact": hidden.answer_exact and hidden.evidence.exact,
                    "valid_schema": hidden.parsed.schema_valid,
                    "overcitation_count": hidden.overcitation_count,
                }
            )
        self._log_rollouts(rollout_rows)
        self._log_mlflow(trainer_state, rollout_rows)
        return selected_rewards

    def _log_rollouts(self, rows: list[dict[str, Any]]) -> None:
        if self.rollout_log is None:
            return
        self.rollout_log.parent.mkdir(parents=True, exist_ok=True)
        with self.rollout_log.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")

    def _log_mlflow(self, trainer_state: Any, rows: list[dict[str, Any]]) -> None:
        if "MLFLOW_TRACKING_URI" not in os.environ:
            return
        import mlflow

        if mlflow.active_run() is None:
            return
        mlflow.log_metrics(
            {
                "rollout/selected_reward": fmean(float(row["selected_reward"]) for row in rows),
                "rollout/hidden_joint_exact": fmean(float(row["joint_exact"]) for row in rows),
                "rollout/hidden_evidence_precision": fmean(
                    float(row["evidence_precision"]) for row in rows
                ),
                "rollout/overcitation_count": fmean(
                    float(row["overcitation_count"]) for row in rows
                ),
            },
            step=int(trainer_state.global_step),
        )
