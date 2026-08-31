from pathlib import Path

from chart_provenance_rl.smoke import run_cpu_smoke


def test_cpu_smoke_exercises_every_reward_metric_and_statistic(
    tmp_path: Path,
) -> None:
    report = run_cpu_smoke(tmp_path / "smoke")
    assert report["report_kind"] == "synthetic_cpu_smoke_not_model_results"
    assert report["audit"]["status"] == "pass"
    assert set(report["reward_arms"]) == {
        "answer_only",
        "naive_recall",
        "minimal_provenance",
    }
    assert report["reward_arms"]["naive_recall"]["cite_all"] == 1.0
    assert report["reward_arms"]["minimal_provenance"]["cite_all"] < 1.0
    assert report["reward_hacking_attack_success"] == 1.0
    assert report["oracle_metrics"]["joint_exact_pass"] == 1.0
    assert report["three_seed_aggregation"]["seed_count"] == 3
    assert (tmp_path / "smoke" / "smoke_report.json").is_file()
