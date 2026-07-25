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
from runners.in_distribution.cold_start.methods import (  # noqa: E402
    agent4rec_listwise_ranking,
)
from runners.in_distribution.llm_error_budget import (  # noqa: E402
    LLMErrorRateExceededError,
)


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Fake LLM response queue is empty")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.responses.pop(0))
                )
            ]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def _cold_task(*, val_groups: int = 1, test_groups: int = 2) -> ColdStartTask:
    history = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "h1",
                "item_title": "History Positive",
                "item_rating_mean": 3.9,
                "item_genres": "Drama",
                "item_summary": "A remembered dramatic film.",
                "rating": 5,
                "target": 1,
            },
            {
                "user_id": "u1",
                "item_id": "h2",
                "item_title": "History Negative",
                "item_rating_mean": 3.4,
                "item_genres": "Comedy",
                "item_summary": "A remembered comedy.",
                "rating": 2,
                "target": 1,
            },
        ]
    )
    train = pd.DataFrame(
        [
            {
                "user_id": "warm",
                "item_id": "w1",
                "item_title": "Warm Train Only",
                "item_rating_mean": 4.5,
                "item_genres": "Action",
                "item_summary": "A warm-user film.",
                "rating": 5,
                "target": 1,
            }
        ]
    )

    def candidates(split: str, groups: int) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for group_index in range(groups):
            for candidate_index in range(2):
                rows.append(
                    {
                        "user_id": "u1",
                        "item_id": f"{split}{group_index}_{candidate_index}",
                        "candidate_group": f"{split}g{group_index}",
                        "item_title": f"{split} candidate {candidate_index}",
                        "item_rating_mean": 4.0 - candidate_index,
                        "item_genres": "Drama" if candidate_index == 0 else "Comedy",
                        "item_summary": (
                            f"Canonical summary for {split} candidate {candidate_index}."
                        ),
                        "sampled": candidate_index == 1,
                        "target": 1 - candidate_index,
                    }
                )
        return pd.DataFrame(rows)

    return ColdStartTask(
        name="agent4rec_listwise_toy",
        train=train,
        val=candidates("v", val_groups),
        test=candidates("t", test_groups),
        online_session_history=history,
        k=2,
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            sampled_column="sampled",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={
            "dataset": "ml-1m",
            "k": 2,
            "sampling_seed": 7,
            "negative_strategy": "uniform",
            "sampler": {"class": "PerTargetCandidateSampler", "negative_ratio": 1},
            "item_enrichment": {
                "movie_summaries": {
                    "enabled": True,
                    "canonical_column": "summary",
                    "task_column": "item_summary",
                    "source_sha256": "fixture-sha256",
                }
            },
        },
    )


def test_traits_runner_uses_history_profiles_and_ranking_only_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(["1. C1\n2. C2"] * 3)
    monkeypatch.setattr(
        agent4rec_listwise_ranking, "make_llm_client", lambda _: client
    )

    result = agent4rec_listwise_ranking.run_method(
        _cold_task(),
        tmp_path,
        method_name="agent4rec_listwise_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_workers=1,
        max_llm_attempts=1,
        profile_components=("traits",),
        summary_usage="candidate",
    )

    assert len(client.completions.calls) == 3
    messages = client.completions.calls[0]["messages"]
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "activity" in system_prompt.lower()
    assert "History Positive" not in user_prompt
    assert "Warm Train Only" not in user_prompt
    assert "History ratings" in user_prompt
    assert "Canonical summary" in user_prompt
    assert "rate the candidate movie at least 4 out of 5" in user_prompt

    assert not (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics_ranking.json").exists()
    assert result["main_metric"].endswith("macro_by_user_group_mean.ndcg@1")
    assert result["test"]["macro_by_group"]["groups_with_score_ties"] == 0

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["score"].tolist() == [1.0, 0.0] * 3
    assert "prediction" not in predictions
    assert predictions["split"].tolist() == ["val", "val"] + ["test"] * 4

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    scorer = manifest["scorer"]
    assert scorer["fit_on"] == "online_session_history"
    assert scorer["profile_generator"]["activity_source"] == "fit_frame_row_counts"
    assert scorer["profile_generator"]["profile_components"] == ["traits"]
    assert scorer["summary_usage"] == "candidate"
    assert scorer["item_summaries"]["candidate_item_summaries"] is True
    assert (
        scorer["item_summaries"]["canonical_enrichment"]["movie_summaries"][
            "source_sha256"
        ]
        == "fixture-sha256"
    )
    assert "threshold" not in manifest["decision_rule"]


def test_traits_runner_logs_split_failures_and_enforces_error_budget(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(["1. C1\n2. C2", "invalid ranking"])
    monkeypatch.setattr(
        agent4rec_listwise_ranking, "make_llm_client", lambda _: client
    )

    with pytest.raises(LLMErrorRateExceededError):
        agent4rec_listwise_ranking.run_method(
            _cold_task(),
            tmp_path,
            method_name="agent4rec_listwise_test",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            max_workers=1,
            max_llm_attempts=1,
            profile_components=("traits",),
            summary_usage="candidate",
            max_error_rate=0.0,
            min_groups_before_check=1,
        )

    errors = [
        json.loads(line)
        for line in (tmp_path / "llm_errors.jsonl").read_text().splitlines()
    ]
    assert errors[0]["split"] == "test"
    assert not (tmp_path / "metrics_ranking.json").exists()


def test_smoke_limit_applies_independently_to_val_and_test(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(["1. C1\n2. C2"] * 50)
    monkeypatch.setattr(
        agent4rec_listwise_ranking, "make_llm_client", lambda _: client
    )

    agent4rec_listwise_ranking.run_method(
        _cold_task(val_groups=26, test_groups=27),
        tmp_path,
        method_name="agent4rec_listwise_smoke_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=25,
        max_workers=1,
        max_llm_attempts=1,
        profile_components=("traits",),
        summary_usage="candidate",
    )

    assert len(client.completions.calls) == 50
    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions.groupby("split")["candidate_group"].nunique().to_dict() == {
        "test": 25,
        "val": 25,
    }


def test_taste_uses_history_only_and_non_summary_cache(
    tmp_path: Path, monkeypatch
) -> None:
    taste_client = FakeClient(
        [
            "TASTE: Likes dramas.\n"
            "REASON: Rated a drama highly.\n"
            "HIGH RATINGS: dramas\n"
            "LOW RATINGS: comedies"
        ]
    )
    ranking_client = FakeClient(["1. C1\n2. C2"] * 3)
    monkeypatch.setattr(
        agent4rec_listwise_ranking,
        "make_llm_client",
        lambda name: taste_client if name == "taste" else ranking_client,
    )
    observed_cache_args: dict[str, object] = {}

    def fake_cache_path(_task: ColdStartTask, **kwargs: object) -> Path:
        observed_cache_args.update(kwargs)
        return tmp_path / "taste_cache.json"

    monkeypatch.setattr(
        agent4rec_listwise_ranking,
        "agent4rec_taste_cache_path",
        fake_cache_path,
    )

    agent4rec_listwise_ranking.run_method(
        _cold_task(),
        tmp_path / "run",
        method_name="agent4rec_listwise_taste_test",
        client_name="ranking",
        model="fake-model",
        max_candidate_groups=None,
        max_workers=1,
        max_llm_attempts=1,
        profile_components=("traits", "taste"),
        taste_client_name="taste",
        taste_model="fake-taste-model",
        taste_max_attempts=1,
        summary_usage="candidate",
    )

    assert observed_cache_args["use_history_item_summaries"] is False
    taste_prompt = taste_client.completions.calls[0]["messages"][1]["content"]
    assert "History Positive" in taste_prompt
    assert "Canonical summary for v" not in taste_prompt
    system_prompt = ranking_client.completions.calls[0]["messages"][0]["content"]
    assert "Likes dramas" in system_prompt


def test_taste_sumboth_uses_history_summaries_in_cache_key(
    tmp_path: Path, monkeypatch
) -> None:
    taste_client = FakeClient(
        [
            "TASTE: Likes dramas.\n"
            "REASON: Rated a drama highly.\n"
            "HIGH RATINGS: dramas\n"
            "LOW RATINGS: comedies"
        ]
    )
    ranking_client = FakeClient(["1. C1\n2. C2"] * 3)
    monkeypatch.setattr(
        agent4rec_listwise_ranking,
        "make_llm_client",
        lambda name: taste_client if name == "taste" else ranking_client,
    )
    observed_cache_args: dict[str, object] = {}

    def fake_cache_path(_task: ColdStartTask, **kwargs: object) -> Path:
        observed_cache_args.update(kwargs)
        return tmp_path / "taste_cache_summary.json"

    monkeypatch.setattr(
        agent4rec_listwise_ranking,
        "agent4rec_taste_cache_path",
        fake_cache_path,
    )

    agent4rec_listwise_ranking.run_method(
        _cold_task(),
        tmp_path / "run",
        method_name="agent4rec_listwise_taste_sumboth_test",
        client_name="ranking",
        model="fake-model",
        max_candidate_groups=None,
        max_workers=1,
        max_llm_attempts=1,
        profile_components=("traits", "taste"),
        taste_client_name="taste",
        taste_model="fake-taste-model",
        taste_max_attempts=1,
        summary_usage="both",
    )

    assert observed_cache_args["use_history_item_summaries"] is True
    taste_prompt = taste_client.completions.calls[0]["messages"][1]["content"]
    assert "History Positive" in taste_prompt
    assert "A remembered dramatic film." in taste_prompt


def test_taste_failure_aborts_before_ranking_and_is_not_group_error(
    tmp_path: Path, monkeypatch
) -> None:
    taste_client = FakeClient(["invalid taste response"])
    ranking_client = FakeClient(["1. C1\n2. C2"] * 3)
    monkeypatch.setattr(
        agent4rec_listwise_ranking,
        "make_llm_client",
        lambda name: taste_client if name == "taste" else ranking_client,
    )
    monkeypatch.setattr(
        agent4rec_listwise_ranking,
        "agent4rec_taste_cache_path",
        lambda *args, **kwargs: tmp_path / "taste_cache.json",
    )

    with pytest.raises(RuntimeError, match="taste profile"):
        agent4rec_listwise_ranking.run_method(
            _cold_task(),
            tmp_path / "run",
            method_name="agent4rec_listwise_taste_test",
            client_name="ranking",
            model="fake-model",
            max_candidate_groups=None,
            max_workers=1,
            max_llm_attempts=1,
            profile_components=("traits", "taste"),
            taste_client_name="taste",
            taste_model="fake-taste-model",
            taste_max_attempts=1,
            summary_usage="candidate",
        )

    assert not ranking_client.completions.calls
    assert not (tmp_path / "run" / "llm_errors.jsonl").exists()
    assert not (tmp_path / "run" / "metrics_ranking.json").exists()


def test_wrappers_and_registry_cover_methods(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(
        _task: ColdStartTask, _output_dir: Path, **kwargs: object
    ) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_listwise_ranking, "run_method", fake_run_method)
    task = _cold_task()
    agent4rec_listwise_ranking.run_qwen36_27b_traits_smoke(task, tmp_path)
    agent4rec_listwise_ranking.run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_full(
        task, tmp_path
    )
    agent4rec_listwise_ranking.run_qwen36_27b_traits_taste_gpt4o_mini_sumboth_full(
        task, tmp_path
    )
    agent4rec_listwise_ranking.run_qwen36_35b_a3b_traits_smoke(task, tmp_path)
    agent4rec_listwise_ranking.run_qwen36_35b_a3b_traits_taste_gpt4o_mini_candidate_summary_full(
        task, tmp_path
    )
    agent4rec_listwise_ranking.run_qwen36_35b_a3b_traits_taste_gpt4o_mini_sumboth_full(
        task, tmp_path
    )
    assert calls[0]["max_candidate_groups"] == 25
    assert calls[0]["profile_components"] == ("traits",)
    assert calls[0]["model"] == "Qwen/Qwen3.6-27B"
    assert calls[1]["max_candidate_groups"] is None
    assert calls[1]["profile_components"] == ("traits", "taste")
    assert calls[1]["summary_usage"] == "candidate"
    assert calls[2]["summary_usage"] == "both"
    assert calls[3]["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert calls[3]["profile_components"] == ("traits",)
    assert calls[4]["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert calls[4]["profile_components"] == ("traits", "taste")
    assert calls[4]["summary_usage"] == "candidate"
    assert calls[5]["summary_usage"] == "both"

    expected = {
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_smoke",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_full",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_smoke",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_full",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_taste_gpt4o_mini_sumboth_smoke",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_taste_gpt4o_mini_sumboth_full",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_traits_smoke",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_traits_full",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_candidate_summary_smoke",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_candidate_summary_full",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_sumboth_smoke",
        "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_sumboth_full",
    }
    assert expected <= METHOD_RUNNERS.keys()
