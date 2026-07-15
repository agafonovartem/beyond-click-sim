from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.tasks import Task, TaskSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.policy_ranking_agreement.methods import agent4rec_yes_no  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


# ---------------------------------------------------------------------------
# LLM response helpers
# ---------------------------------------------------------------------------

_YES_NO_RESPONSE = (
    "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated fits\n"
    "ID: C2; MOVIE: Godfather; WATCH: no; REASON: crime not fit"
)


# ---------------------------------------------------------------------------
# Task fixture helpers
# ---------------------------------------------------------------------------


def _ml1m_policy_task(
    *,
    dataset: str = "ml-1m",
    policies: list[str] | None = None,
    users: list[str] | None = None,
) -> Task:
    """Minimal policy ranking Task with 2 policies, 2 users, 2 items per group.

    Train contains only warm users (not in test). Test has 'policy' and 'rank'
    columns as required by the policy ranking protocol.
    """
    if policies is None:
        policies = ["popularity", "random"]
    if users is None:
        users = ["u1", "u2"]

    # In policy ranking, train and test users overlap (random fraction split).
    # Include a training row for each test user so the profile generator can build traits.
    train_rows = [
        {
            "user_id": user,
            "item_id": f"iw_{user}",
            "item_title": "Forrest Gump",
            "item_genres": "Drama",
            "item_summary": "A kind man witnesses decades of American life.",
            "rating": 5,
            "target": 1,
        }
        for user in users
    ]
    train_df = pd.DataFrame(train_rows)

    test_rows = []
    rank = 1
    for policy in policies:
        for user in users:
            for j, (title, genres, target) in enumerate([
                ("Lion King", "Animation", 1),
                ("Godfather", "Crime", 0),
            ]):
                test_rows.append({
                    "user_id": user,
                    "item_id": f"it_{policy}_{user}_{j}",
                    "item_title": title,
                    "item_genres": genres,
                    "item_summary": (
                        "A lion prince returns to reclaim his kingdom."
                        if title == "Lion King"
                        else "A crime family transfers power to a reluctant son."
                    ),
                    "policy": policy,
                    "rank": rank,
                    "target": target,
                })
            rank += 1

    test_df = pd.DataFrame(test_rows)

    return Task(
        name="test_policy_ranking_task",
        train=train_df,
        val=pd.DataFrame(
            columns=["user_id", "item_id", "item_summary", "target"]
        ),
        test=test_df,
        schema=TaskSchema(
            target_column="target",
            candidate_group_column=None,
            sampled_column=None,
            feature_columns=("user_id", "item_id"),
        ),
        manifest={
            "dataset": dataset,
            "dataset_version": "v1",
            "splitter": {"seed": 0},
            "item_enrichment": {
                "movie_summaries": {
                    "enabled": True,
                    "canonical_column": "summary",
                    "task_column": "item_summary",
                    "source_sha256": "fake-sha256",
                }
            },
        },
    )


def _n_groups(policies: list[str], users: list[str]) -> int:
    return len(policies) * len(users)


# ---------------------------------------------------------------------------
# Happy path — traits-only, ml-1m
# ---------------------------------------------------------------------------


def test_policy_ranking_agent4rec_runner_writes_all_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    # 4 groups (2 policies × 2 users) → 4 LLM calls
    client = FakeClient([_YES_NO_RESPONSE] * 4)
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_policy_task()
    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="agent4rec_yes_no_policy_ranking_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert result["scored_rows"] > 0
    assert result["requested_rows"] == 8  # 2 policies × 2 users × 2 items
    assert result["llm_errors"] == 0

    assert (tmp_path / "predictions.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "llm_errors.jsonl").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "policy_metrics.json").exists()


def test_policy_ranking_agent4rec_runner_metrics_has_kendall_tau(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient([_YES_NO_RESPONSE] * 4)
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_policy_task()
    agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="test_metrics",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "test" in metrics
    assert "kendall_tau" in metrics["test"], (
        "metrics.json must contain test.kendall_tau for the main policy ranking metric"
    )
    assert metrics["main_metric"] == "test.kendall_tau"


# ---------------------------------------------------------------------------
# Manifest correctness
# ---------------------------------------------------------------------------


def test_policy_ranking_agent4rec_runner_manifest_structure(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient([_YES_NO_RESPONSE] * 4)
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_policy_task()
    agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="test_manifest",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "Agent4RecYesNoScorer"
    assert manifest["scorer"]["fit_on"] == "task.train"
    assert manifest["scorer"]["candidate_group"] == "user_id::policy"
    assert manifest["protocol"] == "policy_ranking"
    assert manifest["scorer"]["candidate_description_columns"] == [
        "item_title",
        "item_genres",
        "item_summary",
    ]
    assert manifest["scorer"]["summary_usage"] == "candidate"
    assert manifest["scorer"]["item_summaries"]["candidate_item_summaries"] is True


# ---------------------------------------------------------------------------
# Dataset configuration: ml-1m includes item_rating_mean; steam uses json columns
# ---------------------------------------------------------------------------


def test_dataset_candidate_columns_ml1m_uses_title_and_genres_only() -> None:
    # Policy ranking tasks do not add item_rating_mean (no item-stats enrichment step).
    assert agent4rec_yes_no.DATASET_CANDIDATE_COLUMNS["ml-1m"] == (
        "item_title",
        "item_genres",
    )


def test_dataset_candidate_columns_steam_includes_genres_and_tags() -> None:
    steam_cols = agent4rec_yes_no.DATASET_CANDIDATE_COLUMNS["steam"]
    assert "item_genres_json" in steam_cols
    assert "item_tags_json" in steam_cols


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_policy_ranking_agent4rec_runner_raises_for_unknown_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: FakeClient([]))
    task = _ml1m_policy_task(dataset="nonexistent_dataset")

    with pytest.raises(ValueError, match="no dataset config"):
        agent4rec_yes_no.run_method(
            task,
            tmp_path,
            method_name="test_bad_dataset",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
        )


def test_policy_ranking_agent4rec_taste_requires_client_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: FakeClient([]))
    task = _ml1m_policy_task()

    with pytest.raises(ValueError, match="taste_client_name"):
        agent4rec_yes_no.run_method(
            task,
            tmp_path,
            method_name="test",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            profile_components=("traits", "taste"),
            taste_model="gpt-4o-mini",
        )


# ---------------------------------------------------------------------------
# Smoke capping: proportional per policy
# ---------------------------------------------------------------------------


def test_policy_ranking_agent4rec_runner_smoke_cap_limits_groups(
    tmp_path: Path, monkeypatch
) -> None:
    # 2 policies × 2 users = 4 groups. Cap at max_candidate_groups=2 → 1 per policy.
    client = FakeClient([_YES_NO_RESPONSE] * 2)
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_policy_task()
    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="test_smoke_cap",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=2,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert result["requested_groups"] == 2
    assert result["max_candidate_groups"] == 2
    assert result["scored_rows"] < 8  # fewer rows than the full test set


# ---------------------------------------------------------------------------
# Taste profile path
# ---------------------------------------------------------------------------


def test_policy_ranking_agent4rec_taste_profile_writes_cache(
    tmp_path: Path, monkeypatch
) -> None:
    scoring_client = FakeClient([_YES_NO_RESPONSE] * 4)
    taste_client = FakeClient(
        [
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I rated animated movies highly.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies"
        ]
        * 2  # 2 test users (u1, u2)
    )

    def _fake_make_llm_client(client_name: str) -> FakeClient:
        if client_name == "openai":
            return taste_client
        if client_name == "fake":
            return scoring_client
        raise AssertionError(f"unexpected client: {client_name!r}")

    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", _fake_make_llm_client)
    cache_path = tmp_path / "taste-cache.jsonl"
    monkeypatch.setattr(
        agent4rec_yes_no,
        "agent4rec_taste_cache_path",
        lambda *_a, **_kw: cache_path,
    )

    task = _ml1m_policy_task()
    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="test_taste",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        profile_components=("traits", "taste"),
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
    )

    assert result["scored_rows"] > 0
    assert cache_path.exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    profile_manifest = manifest["scorer"]["profile_generator"]
    assert profile_manifest["profile_components"] == ["traits", "taste"]
    assert profile_manifest["taste"]["client_name"] == "openai"
    assert profile_manifest["taste"]["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# agent4rec_taste_cache_path — path construction
# ---------------------------------------------------------------------------


def test_agent4rec_taste_cache_path_builds_expected_filename(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(agent4rec_yes_no, "repo_root", lambda: tmp_path)
    task = _ml1m_policy_task()

    path = agent4rec_yes_no.agent4rec_taste_cache_path(
        task,
        taste_model="gpt-4o-mini",
        taste_prompt_version="agent4rec_taste_v1",
    )

    assert path.parent == tmp_path / "outputs" / "agent4rec_taste_cache"
    assert path.name == "ml-1m_v1_seed0_gpt-4o-mini_agent4rec_taste_v1.jsonl"


def test_agent4rec_taste_cache_path_slugifies_model_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(agent4rec_yes_no, "repo_root", lambda: tmp_path)
    task = _ml1m_policy_task()

    path = agent4rec_yes_no.agent4rec_taste_cache_path(
        task,
        taste_model="Qwen/Qwen3.6-27B",
        taste_prompt_version="v1",
    )

    assert "Qwen-Qwen3.6-27B" in path.name
    assert "/" not in path.name


# ---------------------------------------------------------------------------
# _cache_slug — unit tests
# ---------------------------------------------------------------------------


def test_cache_slug_leaves_safe_characters_unchanged() -> None:
    assert agent4rec_yes_no._cache_slug("gpt-4o-mini") == "gpt-4o-mini"
    assert agent4rec_yes_no._cache_slug("llama3.1:8b") == "llama3.1-8b"
    assert agent4rec_yes_no._cache_slug("Qwen/Qwen3.6-27B") == "Qwen-Qwen3.6-27B"


def test_cache_slug_raises_on_empty_result() -> None:
    with pytest.raises(ValueError, match="Cannot build cache slug"):
        agent4rec_yes_no._cache_slug("///")


# ---------------------------------------------------------------------------
# Wrapper functions — smoke & full kwargs
# ---------------------------------------------------------------------------


def test_policy_ranking_agent4rec_qwen_smoke_and_full_wrappers(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_yes_no, "run_method", _fake_run_method)
    task = SimpleNamespace(manifest=_ml1m_policy_task().manifest)

    agent4rec_yes_no.run_qwen36_27b_smoke(task, tmp_path)
    agent4rec_yes_no.run_qwen36_27b_full(task, tmp_path)

    assert calls == [
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_usage": "candidate",
        },
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_usage": "candidate",
        },
    ]


def test_policy_ranking_agent4rec_qwen_traits_taste_wrappers(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_yes_no, "run_method", _fake_run_method)
    task = SimpleNamespace(manifest=_ml1m_policy_task().manifest)

    agent4rec_yes_no.run_qwen36_27b_traits_taste_gpt4o_mini_smoke(task, tmp_path)
    agent4rec_yes_no.run_qwen36_27b_traits_taste_gpt4o_mini_full(task, tmp_path)

    assert calls == [
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "profile_components": ("traits", "taste"),
            "taste_client_name": "openai",
            "taste_model": "gpt-4o-mini",
            "summary_usage": "candidate",
        },
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "profile_components": ("traits", "taste"),
            "taste_client_name": "openai",
            "taste_model": "gpt-4o-mini",
            "summary_usage": "candidate",
        },
    ]
