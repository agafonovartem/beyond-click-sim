from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.cold_start import run as cold_run  # noqa: E402
from scripts import cold_start_pod_grid_lib as grid  # noqa: E402
from scripts import run_cold_start_pod_queue as pod_queue  # noqa: E402


def test_make_output_dir_inserts_run_token(tmp_path: Path) -> None:
    path = cold_run.make_output_dir(
        tmp_path,
        task_name="task_a",
        method_name="method_b",
        run_token="gdeadbeef_jabc_a1",
    )
    assert path.parent == tmp_path
    assert path.name.endswith("_gdeadbeef_jabc_a1_task_a_method_b")
    assert path.name.split("_", 1)[0].endswith("Z") or "T" in path.name.split("_", 1)[0]


def test_run_tokens_prevent_same_second_collision(tmp_path: Path, monkeypatch) -> None:
    class FixedDateTime:
        @staticmethod
        def now(tz=None):
            from datetime import datetime, timezone

            return datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(cold_run, "datetime", FixedDateTime)
    a = cold_run.make_output_dir(
        tmp_path, task_name="t", method_name="m", run_token="tokenA"
    )
    b = cold_run.make_output_dir(
        tmp_path, task_name="t", method_name="m", run_token="tokenB"
    )
    assert a != b
    assert a.name == "20260726T120000Z_tokenA_t_m"
    assert b.name == "20260726T120000Z_tokenB_t_m"


def test_validate_run_token_rejects_unsafe() -> None:
    with pytest.raises(ValueError):
        cold_run.validate_run_token("../evil")
    with pytest.raises(ValueError):
        cold_run.validate_run_token("has space")
    assert cold_run.validate_run_token("ok_token-1") == "ok_token-1"


def test_listwise_completion_requires_manifest_and_metrics(tmp_path: Path) -> None:
    method = "llm_preference_listwise_ranking_vllm_qwen36_27b_item_stats_sumboth_full"
    task = "ml-1m_item_stats_cold_like_k1_c5_eval1000_m1_negu_split0_sample0"
    run_dir = tmp_path / f"20260726T120000Z_{task}_{method}"
    run_dir.mkdir()
    assert not cold_run.is_completed(tmp_path, task, method)
    (run_dir / "metrics_ranking.json").write_text("{}", encoding="utf-8")
    assert not cold_run.is_completed(tmp_path, task, method)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    assert cold_run.is_completed(tmp_path, task, method)


def test_ablation_grid_is_4x45_item_stats_cartesian() -> None:
    built = grid.build_ablation_grid()
    assert built["job_count"] == 180
    assert len(built["methods"]) == 4
    assert len(built["tasks"]) == 45
    assert all("item_stats" in t for t in built["tasks"])
    pairs = [(j["method"], j["task"]) for j in built["jobs"]]
    assert len(pairs) == len(set(pairs))
    # every method gets every task
    for method in built["methods"]:
        method_tasks = {t for m, t in pairs if m == method}
        assert method_tasks == set(built["tasks"])


def test_item_stats_llm_methods_never_get_plain_tasks() -> None:
    method = (
        "llm_preference_listwise_ranking_vllm_qwen36_27b_item_stats_sumhistory_full"
    )
    tasks = grid.task_names_for_method(method)
    assert tasks
    assert all("item_stats" in t for t in tasks)
    # Regression against the old queue helper once fixed.
    assert pod_queue.method_requires_item_stats_tasks(method)
    assert all("item_stats" in t for t in pod_queue.task_names_for_method(method))


def test_pod_assignment_is_unique_and_balanced() -> None:
    built = grid.build_ablation_grid(grid_id="ablation_testgrid_deadbeef")
    sizes = [len(jobs) for jobs in built["pods_jobs"].values()]
    assert min(sizes) >= 22
    assert max(sizes) <= 23
    assert sum(sizes) == 180
    owned = {}
    for pod, jobs in built["pods_jobs"].items():
        for job in jobs:
            key = (job["method"], job["task"])
            assert key not in owned
            owned[key] = pod
        methods_on_pod = {job["method"] for job in jobs}
        assert len(methods_on_pod) == 4
    job_ids = [j["job_id"] for j in built["jobs"]]
    assert len(job_ids) == len(set(job_ids))
    tokens = [
        grid.make_run_token(built["grid_id"], j["job_id"], attempt)
        for j in built["jobs"]
        for attempt in (1, 2, 3)
    ]
    assert len(tokens) == len(set(tokens))


def test_worker_state_roundtrip(tmp_path: Path) -> None:
    st = grid.JobState(status="running", attempts=1, run_token="tok", result_dir=None)
    path = tmp_path / "state.json"
    grid.write_json_atomic(
        path,
        {
            "grid_id": "g",
            "pod": "p",
            "jobs": {"j1": st.to_dict()},
        },
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    restored = grid.JobState.from_dict(raw["jobs"]["j1"])
    assert restored.status == "running"
    assert restored.attempts == 1
    assert restored.run_token == "tok"


def test_find_result_dir_for_token(tmp_path: Path) -> None:
    method = "method_x"
    task = "task_y"
    token = "g123_jabc_a1"
    leaf = tmp_path / f"20260726T010203Z_{token}_{task}_{method}"
    leaf.mkdir()
    found = grid.find_result_dir_for_token(
        tmp_path, task=task, method=method, run_token=token
    )
    assert found == leaf
