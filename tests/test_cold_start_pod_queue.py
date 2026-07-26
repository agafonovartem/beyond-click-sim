from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_cold_start_pod_queue as pod_queue  # noqa: E402


def test_required_metric_files_are_method_aware() -> None:
    assert pod_queue.required_metric_files(
        "llm_preference_listwise_ranking_vllm_qwen36_27b_full"
    ) == ("metrics_ranking.json",)
    assert pod_queue.required_metric_files(
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_full"
    ) == ("metrics_ranking.json",)
    assert pod_queue.required_metric_files(
        "llm_yes_no_vllm_qwen36_27b_full"
    ) == ("metrics.json", "metrics_ranking.json")


def test_method_task_variant_rules() -> None:
    assert pod_queue.method_requires_item_stats_tasks(
        "agent4rec_yes_no_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_full"
    )
    assert pod_queue.method_requires_item_stats_tasks(
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_full"
    )
    assert not pod_queue.method_requires_item_stats_tasks(
        "llm_preference_listwise_ranking_vllm_qwen36_27b_full"
    )
    assert pod_queue.method_requires_item_stats_tasks(
        "llm_preference_listwise_ranking_vllm_qwen36_27b_item_stats_sumboth_full"
    )
    assert pod_queue.method_requires_taste_cache(
        "agent4rec_yes_no_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_full"
    )
    assert pod_queue.method_requires_taste_cache(
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_"
        "traits_taste_gpt4o_mini_candidate_summary_full"
    )
    assert not pod_queue.method_requires_taste_cache(
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_full"
    )
    assert not pod_queue.method_requires_taste_cache(
        "llm_preference_listwise_ranking_vllm_qwen36_27b_full"
    )


def test_session_names_are_unique_across_cold_start_grid() -> None:
    names = [pod_queue.session_name(task) for task in pod_queue.TASKS]
    assert len(names) == len(set(names))
    assert all(name.startswith("cs_") for name in names)


def test_task_from_result_dirname_parses_timestamped_dirs() -> None:
    task = pod_queue.TASKS[0]
    dirname = f"20260726T114014Z_{task}_{pod_queue.METHOD}"
    assert pod_queue.task_from_result_dirname(dirname) == task
    assert pod_queue.task_from_result_dirname("not_a_result") is None


def test_task_name_grid_switches_between_plain_and_item_stats() -> None:
    listwise_tasks = pod_queue.task_names_for_method(
        "llm_preference_listwise_ranking_vllm_qwen36_27b_full"
    )
    assert listwise_tasks
    assert all("item_stats" not in task_name for task_name in listwise_tasks)

    item_stats_llm_tasks = pod_queue.task_names_for_method(
        "llm_preference_listwise_ranking_vllm_qwen36_27b_item_stats_sumboth_full"
    )
    assert item_stats_llm_tasks
    assert all("item_stats" in task_name for task_name in item_stats_llm_tasks)

    agent_tasks = pod_queue.task_names_for_method(
        "agent4rec_yes_no_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_full"
    )
    assert agent_tasks
    assert all("item_stats" in task_name for task_name in agent_tasks)

    agent_listwise_tasks = pod_queue.task_names_for_method(
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_full"
    )
    assert agent_listwise_tasks
    assert all("item_stats" in task_name for task_name in agent_listwise_tasks)


def test_has_required_metric_files_uses_configured_requirements(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics_ranking.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pod_queue, "REQUIRED_METRIC_FILES", ("metrics_ranking.json",))
    assert pod_queue.has_required_metric_files(run_dir)

    monkeypatch.setattr(
        pod_queue,
        "REQUIRED_METRIC_FILES",
        ("metrics.json", "metrics_ranking.json"),
    )
    assert not pod_queue.has_required_metric_files(run_dir)


def test_queue_state_is_method_scoped_and_rechecks_done_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    task = "cold_task"
    method = "agent4rec_preference_listwise_ranking_test"
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(pod_queue, "TASKS", [task])
    monkeypatch.setattr(pod_queue, "METHOD", method)
    monkeypatch.setattr(pod_queue, "local_output_root", lambda: tmp_path / "outputs")
    monkeypatch.setattr(
        pod_queue, "REQUIRED_METRIC_FILES", ("metrics_ranking.json",)
    )

    state_path.write_text(
        json.dumps(
            {
                "method": "llm_preference_listwise_ranking_other",
                "tasks": {task: {"status": "done"}},
            }
        ),
        encoding="utf-8",
    )
    assert pod_queue.QueueState.load(state_path).tasks[task].status == "pending"

    state_path.write_text(
        json.dumps(
            {"method": method, "tasks": {task: {"status": "done"}}}
        ),
        encoding="utf-8",
    )
    assert pod_queue.QueueState.load(state_path).tasks[task].status == "pending"

    run_dir = tmp_path / "outputs" / f"run_{task}_{method}"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics_ranking.json").write_text("{}", encoding="utf-8")
    assert pod_queue.QueueState.load(state_path).tasks[task].status == "downloaded"
