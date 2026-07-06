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

from runners.in_distribution.regression_prediction.methods import agent4rec_regressor  # noqa: E402


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response),
                )
            ]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def test_agent4rec_regression_runner_writes_profile_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(["RATING: 5", "RATING: 2"])
    monkeypatch.setattr(agent4rec_regressor, "make_llm_client", lambda _: client)

    task = _toy_task()
    result = agent4rec_regressor.run_method(
        task,
        tmp_path,
        method_name="agent4rec_regressor_test",
        client_name="fake",
        model="fake-model",
        max_rows=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert result["scored_rows"] == 2
    assert result["requested_rows"] == 2
    assert result["llm_errors"] == 0
    assert result["test"]["micro"]["mae"] == 0.0

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["score"].tolist() == [5.0, 2.0]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "Agent4RecRegressor"
    assert manifest["scorer"]["profile_generator"]["profile_components"] == ["traits"]
    assert manifest["scorer"]["profile_generator"]["trait_thresholds"] is not None
    assert manifest["scorer"]["candidate_description_columns"] == [
        "item_title",
        "item_rating_mean",
        "item_genres",
    ]
    assert manifest["scorer"]["uses_item_stats"] is True
    assert (
        manifest["decision_rule"]["parser_contract"]
        == "agent4rec_rating_line"
    )

    first_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "##movie##" in first_prompt
    assert "<- Lion King -> <- History ratings:4.15 -> <- genres:Animation ->" in first_prompt
    assert "Use this format: RATING: [integer from 1 to 5]" in first_prompt


def test_agent4rec_regression_runner_can_add_candidate_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(["RATING: 5"])
    monkeypatch.setattr(agent4rec_regressor, "make_llm_client", lambda _: client)

    def fake_add_item_summaries(**kwargs: object):
        X_train = kwargs["X_train"].copy()
        X_test = kwargs["X_test"].copy()
        summaries = {
            "i-train-1": "Toys plan a rescue.",
            "i-train-2": "A street thief finds a lamp.",
            "i-train-3": "A detective hunts criminals.",
            "i1": "A young lion reclaims his kingdom.",
            "i2": "A crime family saga.",
        }
        X_train["item_summary"] = X_train["item_id"].map(summaries)
        X_test["item_summary"] = X_test["item_id"].map(summaries)
        return X_train, X_test, {
            "uses_item_summaries": True,
            "history_item_summaries": kwargs["summary_visibility"]["history"],
            "candidate_item_summaries": kwargs["summary_visibility"]["candidate"],
            "source_path": "fake.csv",
        }

    monkeypatch.setattr(
        agent4rec_regressor,
        "add_ml1m_item_summaries",
        fake_add_item_summaries,
    )

    task = _toy_task(test=_toy_task().test.head(1).copy())
    agent4rec_regressor.run_method(
        task,
        tmp_path,
        method_name="agent4rec_regressor_test_summary",
        client_name="fake",
        model="fake-model",
        max_rows=None,
        max_llm_attempts=1,
        max_workers=1,
        use_item_summaries=True,
    )

    first_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "<- summary:A young lion reclaims his kingdom. ->" in first_prompt

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["candidate_description_columns"] == [
        "item_title",
        "item_rating_mean",
        "item_genres",
        "item_summary",
    ]
    assert manifest["scorer"]["profile_generator"]["summary_column"] is None
    assert manifest["scorer"]["item_summaries"] == {
        "uses_item_summaries": True,
        "history_item_summaries": False,
        "candidate_item_summaries": True,
        "source_path": "fake.csv",
    }


def test_agent4rec_regression_runner_writes_taste_manifest_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoring_client = FakeClient(["RATING: 5"])
    taste_client = FakeClient(
        [
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I rated animated movies highly.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies"
        ]
    )

    def fake_make_llm_client(client_name: str) -> FakeClient:
        if client_name == "openai":
            return taste_client
        if client_name == "fake":
            return scoring_client
        raise AssertionError(client_name)

    monkeypatch.setattr(agent4rec_regressor, "make_llm_client", fake_make_llm_client)
    base_task = _toy_task()
    cache_path = tmp_path / "taste-cache.jsonl"
    task = _toy_task(test=base_task.test.head(1).copy())

    result = agent4rec_regressor.run_method(
        task,
        tmp_path,
        method_name="agent4rec_regressor_traits_taste_test",
        client_name="fake",
        model="fake-model",
        max_rows=None,
        max_llm_attempts=1,
        max_workers=1,
        profile_components=("traits", "taste"),
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=cache_path,
    )

    assert result["scored_rows"] == 1
    assert len(taste_client.completions.calls) == 1
    assert cache_path.exists()
    cached_rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
    ]
    assert cached_rows[0]["history_item_ids"] == [
        "i-train-1",
        "i-train-2",
        "i-train-3",
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    profile_manifest = manifest["scorer"]["profile_generator"]
    assert profile_manifest["profile_components"] == ["traits", "taste"]
    assert profile_manifest["taste"]["client_name"] == "openai"
    assert profile_manifest["taste"]["model"] == "gpt-4o-mini"
    assert profile_manifest["taste"]["cache_path"] == str(cache_path)
    assert profile_manifest["taste"]["cache_stats"] == {
        "requested_users": 1,
        "hits": 0,
        "misses": 1,
        "generated": 1,
        "max_workers": 1,
    }


def test_agent4rec_regression_runner_requires_item_rating_mean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent4rec_regressor,
        "make_llm_client",
        lambda _: FakeClient([]),
    )
    base_task = _toy_task()
    task = _toy_task(
        test=base_task.test.drop(columns=["item_rating_mean"]),
    )

    with pytest.raises(ValueError, match="requires item-stats task columns"):
        agent4rec_regressor.run_method(
            task,
            tmp_path,
            method_name="agent4rec_regressor_test",
            client_name="fake",
            model="fake-model",
            max_rows=None,
            max_llm_attempts=1,
            max_workers=1,
        )


def test_agent4rec_regression_qwen_traits_taste_wrappers_use_openai_taste(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_regressor, "run_method", fake_run_method)
    task = SimpleNamespace()

    agent4rec_regressor.run_qwen36_27b_traits_taste_gpt4o_mini_smoke(task, tmp_path)
    agent4rec_regressor.run_qwen36_27b_traits_taste_gpt4o_mini_full(task, tmp_path)

    assert calls == [
        {
            "method_name": "agent4rec_regressor_vllm_qwen36_27b_traits_taste_gpt4o_mini_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_rows": agent4rec_regressor.SMOKE_ROWS,
            "max_workers": agent4rec_regressor.QWEN36_27B_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "profile_components": ("traits", "taste"),
            "taste_client_name": "openai",
            "taste_model": "gpt-4o-mini",
            "taste_temperature": 0.0,
            "taste_max_tokens": None,
        },
        {
            "method_name": "agent4rec_regressor_vllm_qwen36_27b_traits_taste_gpt4o_mini_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_rows": None,
            "max_workers": agent4rec_regressor.QWEN36_27B_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "profile_components": ("traits", "taste"),
            "taste_client_name": "openai",
            "taste_model": "gpt-4o-mini",
            "taste_temperature": 0.0,
            "taste_max_tokens": None,
        },
    ]


def test_agent4rec_regression_qwen36_27b_summary_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_regressor, "run_method", fake_run_method)
    task = SimpleNamespace()

    agent4rec_regressor.run_qwen36_27b_traits_full(task, tmp_path)
    agent4rec_regressor.run_qwen36_27b_traits_summary_full(task, tmp_path)
    agent4rec_regressor.run_qwen36_27b_taste_gpt4o_mini_full(task, tmp_path)
    agent4rec_regressor.run_qwen36_27b_taste_gpt4o_mini_summary_full(
        task,
        tmp_path,
    )
    agent4rec_regressor.run_qwen36_27b_traits_taste_gpt4o_mini_full(
        task,
        tmp_path,
    )
    agent4rec_regressor.run_qwen36_27b_traits_taste_gpt4o_mini_summary_full(
        task,
        tmp_path,
    )

    common = {
        "client_name": "vllm_local",
        "model": "Qwen/Qwen3.6-27B",
        "max_rows": None,
        "max_workers": agent4rec_regressor.QWEN36_27B_MAX_WORKERS,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    taste = {
        "taste_client_name": "openai",
        "taste_model": "gpt-4o-mini",
        "taste_temperature": 0.0,
        "taste_max_tokens": None,
    }
    assert calls == [
        {
            "method_name": "agent4rec_regressor_vllm_qwen36_27b_traits_full",
            "profile_components": ("traits",),
            **common,
        },
        {
            "method_name": "agent4rec_regressor_vllm_qwen36_27b_traits_summary_full",
            "profile_components": ("traits",),
            "use_item_summaries": True,
            **common,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen36_27b_taste_gpt4o_mini_full"
            ),
            "profile_components": ("taste",),
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen36_27b_taste_gpt4o_mini_summary_full"
            ),
            "profile_components": ("taste",),
            "use_item_summaries": True,
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen36_27b_traits_taste_gpt4o_mini_full"
            ),
            "profile_components": ("traits", "taste"),
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen36_27b_traits_taste_gpt4o_mini_summary_full"
            ),
            "profile_components": ("traits", "taste"),
            "use_item_summaries": True,
            **common,
            **taste,
        },
    ]


def test_agent4rec_regression_qwen3_8b_wrappers_use_expected_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_regressor, "run_method", fake_run_method)
    task = SimpleNamespace()

    agent4rec_regressor.run_qwen3_8b_traits_full(task, tmp_path)
    agent4rec_regressor.run_qwen3_8b_traits_summary_full(task, tmp_path)
    agent4rec_regressor.run_qwen3_8b_taste_gpt4o_mini_full(task, tmp_path)
    agent4rec_regressor.run_qwen3_8b_taste_gpt4o_mini_history_summary_full(
        task,
        tmp_path,
    )
    agent4rec_regressor.run_qwen3_8b_taste_gpt4o_mini_candidate_summary_full(
        task,
        tmp_path,
    )
    agent4rec_regressor.run_qwen3_8b_taste_gpt4o_mini_summary_full(task, tmp_path)
    agent4rec_regressor.run_qwen3_8b_traits_taste_gpt4o_mini_full(task, tmp_path)
    agent4rec_regressor.run_qwen3_8b_traits_taste_gpt4o_mini_history_summary_full(
        task,
        tmp_path,
    )
    agent4rec_regressor.run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full(
        task,
        tmp_path,
    )
    agent4rec_regressor.run_qwen3_8b_traits_taste_gpt4o_mini_summary_full(
        task,
        tmp_path,
    )
    agent4rec_regressor.run_qwen3_8b_traits_smoke(task, tmp_path)

    common = {
        "client_name": "vllm_local",
        "model": "Qwen/Qwen3-8B",
        "max_workers": 128,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    taste = {
        "taste_client_name": "openai",
        "taste_model": "gpt-4o-mini",
        "taste_temperature": 0.0,
        "taste_max_tokens": None,
    }
    assert calls == [
        {
            "method_name": "agent4rec_regressor_vllm_qwen3_8b_traits_full",
            "max_rows": None,
            "profile_components": ("traits",),
            **common,
        },
        {
            "method_name": "agent4rec_regressor_vllm_qwen3_8b_traits_summary_full",
            "max_rows": None,
            "profile_components": ("traits",),
            "use_item_summaries": True,
            **common,
        },
        {
            "method_name": "agent4rec_regressor_vllm_qwen3_8b_taste_gpt4o_mini_full",
            "max_rows": None,
            "profile_components": ("taste",),
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen3_8b_taste_gpt4o_mini_history_summary_full"
            ),
            "max_rows": None,
            "profile_components": ("taste",),
            "history_item_summaries": True,
            "candidate_item_summaries": False,
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen3_8b_taste_gpt4o_mini_candidate_summary_full"
            ),
            "max_rows": None,
            "profile_components": ("taste",),
            "history_item_summaries": False,
            "candidate_item_summaries": True,
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen3_8b_taste_gpt4o_mini_summary_full"
            ),
            "max_rows": None,
            "profile_components": ("taste",),
            "use_item_summaries": True,
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen3_8b_traits_taste_gpt4o_mini_full"
            ),
            "max_rows": None,
            "profile_components": ("traits", "taste"),
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen3_8b_traits_taste_gpt4o_mini_history_summary_full"
            ),
            "max_rows": None,
            "profile_components": ("traits", "taste"),
            "history_item_summaries": True,
            "candidate_item_summaries": False,
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full"
            ),
            "max_rows": None,
            "profile_components": ("traits", "taste"),
            "history_item_summaries": False,
            "candidate_item_summaries": True,
            **common,
            **taste,
        },
        {
            "method_name": (
                "agent4rec_regressor_vllm_qwen3_8b_traits_taste_gpt4o_mini_summary_full"
            ),
            "max_rows": None,
            "profile_components": ("traits", "taste"),
            "use_item_summaries": True,
            **common,
            **taste,
        },
        {
            "method_name": "agent4rec_regressor_vllm_qwen3_8b_traits_smoke",
            "max_rows": agent4rec_regressor.SMOKE_ROWS,
            "profile_components": ("traits",),
            **common,
        },
    ]


def _toy_task(*, test: pd.DataFrame | None = None) -> Task:
    return Task(
        name="ml-1m_rating_item_stats_eval_users1_seed0",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["i-train-1", "i-train-2", "i-train-3"],
                "item_title": ["Toy Story", "Aladdin", "Heat"],
                "item_genres": ["Animation|Comedy", "Animation", "Crime"],
                "item_rating_mean": [4.15, 3.95, 3.60],
                "rating": [5, 4, 2],
                "target": [5, 4, 2],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=(
            test
            if test is not None
            else pd.DataFrame(
                {
                    "user_id": ["u1", "u1"],
                    "item_id": ["i1", "i2"],
                    "item_title": ["Lion King", "Godfather"],
                    "item_genres": ["Animation", "Crime"],
                    "item_rating_mean": [4.153, 4.567],
                    "rating": [pd.NA, pd.NA],
                    "target": [5, 2],
                },
                index=["a", "b"],
            )
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column=None,
            sampled_column=None,
            feature_columns=(
                "item_title",
                "item_genres",
                "item_rating_mean",
            ),
            history_context_columns=("rating",),
        ),
        manifest={
            "protocol": "regression",
            "dataset": "ml-1m",
            "dataset_version": "v1",
            "target_source_column": "target_rating",
            "splitter": {"seed": 0},
        },
    )
