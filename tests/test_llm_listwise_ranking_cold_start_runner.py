from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.tasks import TaskSchema
from beyond_click_sim.tasks.cold_start import ColdStartTask

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.cold_start.methods import METHOD_RUNNERS  # noqa: E402
from runners.in_distribution.cold_start.methods import llm_listwise_ranking  # noqa: E402
from runners.in_distribution.llm_error_budget import LLMErrorRateExceededError  # noqa: E402


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Fake LLM response queue is empty")
        response = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def _cold_task() -> ColdStartTask:
    history_df = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "ih1",
                "item_title": "History Movie",
                "item_genres": "Drama",
                "rating": 5,
                "target": 1,
            }
        ]
    )
    train_df = pd.DataFrame(
        [
            {
                "user_id": "u_warm",
                "item_id": "iw1",
                "item_title": "Warm Train Only",
                "item_genres": "Action",
                "rating": 4,
                "target": 1,
            }
        ]
    )
    val_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["v1", "v2"],
            "candidate_group": ["vg1", "vg1"],
            "item_title": ["Val Positive", "Val Negative"],
            "item_genres": ["Drama", "Comedy"],
            "sampled": [False, True],
            "target": [1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "item_id": ["t1", "t2", "t3", "t4"],
            "candidate_group": ["tg1", "tg1", "tg2", "tg2"],
            "item_title": ["Test A", "Test B", "Test C", "Test D"],
            "item_genres": ["Drama", "Comedy", "Action", "Crime"],
            "sampled": [False, True, False, True],
            "target": [1, 0, 1, 0],
        }
    )
    return ColdStartTask(
        name="cold_listwise_toy",
        train=train_df,
        val=val_df,
        test=test_df,
        online_session_history=history_df,
        k=3,
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            sampled_column="sampled",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={
            "dataset": "ml-1m",
            "k": 3,
            "sampling_seed": 2,
            "negative_strategy": "uniform",
            "sampler": {"class": "PerTargetCandidateSampler", "negative_ratio": 9},
        },
    )


def _summary_cold_task() -> ColdStartTask:
    task = _cold_task()
    task.online_session_history = task.online_session_history.assign(
        item_summary="PROFILE SUMMARY MARKER"
    )
    task.train = task.train.assign(item_summary="WARM TRAIN SUMMARY")
    task.val = task.val.assign(
        item_summary=["CANDIDATE SUMMARY MARKER VAL ONE", "CANDIDATE SUMMARY MARKER VAL TWO"]
    )
    task.test = task.test.assign(
        item_summary=[
            "CANDIDATE SUMMARY MARKER TEST ONE",
            "CANDIDATE SUMMARY MARKER TEST TWO",
            "CANDIDATE SUMMARY MARKER TEST THREE",
            "CANDIDATE SUMMARY MARKER TEST FOUR",
        ]
    )
    task.manifest["item_enrichment"] = {
        "movie_summaries": {
            "enabled": True,
            "canonical_column": "summary",
            "task_column": "item_summary",
            "source_sha256": "test-sha",
        }
    }
    return task


def test_cold_start_listwise_runner_writes_ranking_only_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(["1. C2\n2. C1", "1. C1\n2. C2", "1. C2\n2. C1"])
    monkeypatch.setattr(llm_listwise_ranking, "make_llm_client", lambda _: client)
    task = _cold_task()

    result = llm_listwise_ranking.run_method(
        task,
        tmp_path,
        method_name="llm_preference_listwise_ranking_vllm_qwen36_27b_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert len(client.completions.calls) == 3
    first_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "History Movie" in first_prompt
    assert "Warm Train Only" not in first_prompt
    assert "The user would rate the candidate movie at least 4 out of 5." in first_prompt

    assert not (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics_ranking.json").exists()
    assert result["main_metric"] == (
        "test_failure_as_zero_group.macro_by_user_group_mean.ndcg@1"
    )
    assert result["ranking_evaluation"]["ks"] == [1, 3, 5, 10]
    assert result["test"]["macro_by_group"]["groups_with_score_ties"] == 0
    assert result["test"]["macro_by_group"]["ndcg@10"] >= 0.0

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["split"].tolist() == ["val", "val", "test", "test", "test", "test"]
    assert predictions["score"].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert "prediction" not in predictions.columns
    assert predictions["k"].unique().tolist() == [3]
    assert predictions["negative_ratio"].unique().tolist() == [9]
    assert predictions["sampling_seed"].unique().tolist() == [2]
    assert predictions["negative_strategy"].unique().tolist() == ["uniform"]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "direct_listwise_preference_ranking"
    assert manifest["scorer"]["fit_on"] == "online_session_history"
    assert manifest["scorer"]["k"] == 3
    assert manifest["scorer"]["target_description"] == (
        "The user would rate the candidate movie at least 4 out of 5."
    )
    assert manifest["decision_rule"]["ranking"]["kind"] == "complete_listwise_rank_parser"
    assert (
        manifest["decision_rule"]["ranking"]["score_mapping"]
        == "score = n_candidates - rank"
    )


def test_cold_start_listwise_runner_sumboth_shows_profile_and_candidate_summaries(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(["1. C2\n2. C1", "1. C1\n2. C2", "1. C2\n2. C1"])
    monkeypatch.setattr(llm_listwise_ranking, "make_llm_client", lambda _: client)
    task = _summary_cold_task()

    llm_listwise_ranking.run_method(
        task,
        tmp_path,
        method_name="llm_preference_listwise_ranking_sumboth_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        summary_visibility="both",
    )

    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "PROFILE SUMMARY MARKER" in prompt
    assert "CANDIDATE SUMMARY MARKER VAL ONE" in prompt
    assert "CANDIDATE SUMMARY MARKER VAL TWO" in prompt

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["summary_visibility"] == "both"
    assert "item_summary" in (
        manifest["scorer"]["prompt_columns"]["history_description_columns"]
    )
    assert "item_summary" in (
        manifest["scorer"]["prompt_columns"]["candidate_description_columns"]
    )
    assert manifest["scorer"]["item_summaries"]["history_item_summaries"] is True
    assert manifest["scorer"]["item_summaries"]["candidate_item_summaries"] is True


def test_cold_start_listwise_runner_logs_split_errors_and_failure_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    # val parses; test has one valid and one unparseable group.
    client = FakeClient(["1. C1\n2. C2", "1. C1\n2. C2", "not a ranking"])
    monkeypatch.setattr(llm_listwise_ranking, "make_llm_client", lambda _: client)
    task = _cold_task()

    result = llm_listwise_ranking.run_method(
        task,
        tmp_path,
        method_name="llm_preference_listwise_ranking_vllm_qwen36_27b_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        max_error_rate=1.0,
        min_groups_before_check=1,
    )

    assert result["llm_errors"] == {"val": 0, "test": 1}
    assert result["coverage"]["test"]["failed_rows"] == 2
    assert result["test"]["macro_by_group"]["n_groups"] == 1
    assert result["test_failure_as_zero_group"]["macro_by_group"]["n_groups"] == 2
    assert result["test_failure_as_zero_group"]["macro_by_group"]["failed_groups"] == 1

    errors = [
        json.loads(line)
        for line in (tmp_path / "llm_errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(errors) == 1
    assert errors[0]["split"] == "test"
    assert errors[0]["candidate_group"] == "tg2"


def test_cold_start_listwise_runner_aborts_on_error_budget_and_persists_errors(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(["not a ranking"])
    monkeypatch.setattr(llm_listwise_ranking, "make_llm_client", lambda _: client)
    task = _cold_task()

    with pytest.raises(LLMErrorRateExceededError):
        llm_listwise_ranking.run_method(
            task,
            tmp_path,
            method_name="llm_preference_listwise_ranking_vllm_qwen36_27b_test",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            max_llm_attempts=1,
            max_workers=1,
            max_error_rate=0.10,
            min_groups_before_check=1,
        )

    errors = [
        json.loads(line)
        for line in (tmp_path / "llm_errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(errors) == 1
    assert errors[0]["split"] == "val"
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "predictions.parquet").exists()
    assert not (tmp_path / "metrics_ranking.json").exists()


def test_cold_start_listwise_wrappers_and_registry(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_method(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(llm_listwise_ranking, "run_method", _fake_run_method)
    task = SimpleNamespace()

    llm_listwise_ranking.run_qwen36_27b_smoke(task, tmp_path)
    llm_listwise_ranking.run_qwen36_27b_full(task, tmp_path)
    llm_listwise_ranking.run_qwen36_27b_sumboth_smoke(task, tmp_path)
    llm_listwise_ranking.run_qwen36_27b_sumboth_full(task, tmp_path)
    llm_listwise_ranking.run_qwen36_35b_a3b_smoke(task, tmp_path)
    llm_listwise_ranking.run_qwen36_35b_a3b_full(task, tmp_path)
    llm_listwise_ranking.run_qwen36_35b_a3b_sumboth_smoke(task, tmp_path)
    llm_listwise_ranking.run_qwen36_35b_a3b_sumboth_full(task, tmp_path)

    assert calls == [
        {
            "method_name": "llm_preference_listwise_ranking_vllm_qwen36_27b_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": llm_listwise_ranking.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": "llm_preference_listwise_ranking_vllm_qwen36_27b_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": llm_listwise_ranking.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": "llm_preference_listwise_ranking_vllm_qwen36_27b_sumboth_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": llm_listwise_ranking.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_visibility": "both",
        },
        {
            "method_name": "llm_preference_listwise_ranking_vllm_qwen36_27b_sumboth_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": llm_listwise_ranking.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_visibility": "both",
        },
        {
            "method_name": "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "max_candidate_groups": 25,
            "max_workers": llm_listwise_ranking.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "max_candidate_groups": None,
            "max_workers": llm_listwise_ranking.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b_sumboth_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "max_candidate_groups": 25,
            "max_workers": llm_listwise_ranking.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_visibility": "both",
        },
        {
            "method_name": "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b_sumboth_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "max_candidate_groups": None,
            "max_workers": llm_listwise_ranking.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_visibility": "both",
        },
    ]

    assert (
        "llm_preference_listwise_ranking_vllm_qwen36_27b_smoke" in METHOD_RUNNERS
    )
    assert "llm_preference_listwise_ranking_vllm_qwen36_27b_full" in METHOD_RUNNERS
    assert (
        "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b_smoke" in METHOD_RUNNERS
    )
    assert "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b_full" in METHOD_RUNNERS
    assert (
        "llm_preference_listwise_ranking_vllm_qwen36_27b_sumboth_smoke"
        in METHOD_RUNNERS
    )
    assert (
        "llm_preference_listwise_ranking_vllm_qwen36_27b_sumboth_full"
        in METHOD_RUNNERS
    )
    assert (
        "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b_sumboth_smoke"
        in METHOD_RUNNERS
    )
    assert (
        "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b_sumboth_full"
        in METHOD_RUNNERS
    )
