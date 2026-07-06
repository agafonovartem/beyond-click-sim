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

from runners.in_distribution.regression_prediction.methods import llm_regressor  # noqa: E402


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


def test_llm_regression_runner_writes_valid_coverage_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Task(
        name="ml-1m_rating_eval_users2_seed0",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "item_title": ["Toy Story", "Heat"],
                "item_genres": ["Animation", "Crime"],
                "rating": [5, 2],
                "target": [5, 2],
            }
        ),
        val=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i3"],
                "item_title": ["Aladdin"],
                "item_genres": ["Animation"],
                "rating": [pd.NA],
                "target": [4],
            }
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "item_id": ["i4", "i5"],
                "item_title": ["Lion King", "Portal 2"],
                "item_genres": ["Animation", "Puzzle"],
                "rating": [pd.NA, pd.NA],
                "target": [4, 5],
            },
            index=["a", "b"],
        ),
        schema=TaskSchema(
            target_column="target",
            feature_columns=("item_title", "item_genres"),
            history_context_columns=("rating",),
        ),
        manifest={
            "protocol": "regression",
            "dataset": "ml-1m",
            "target_source_column": "target_rating",
        },
    )
    client = FakeClient(["4", "6", "bad"])
    monkeypatch.setattr(
        llm_regressor,
        "make_llm_client",
        lambda _client_name: client,
    )

    metrics = llm_regressor.run_method(
        task,
        tmp_path,
        method_name="llm_regressor_fake_smoke",
        client_name="fake",
        model="fake-model",
        max_rows=2,
        max_llm_attempts=2,
        max_workers=1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert metrics["method"] == "llm_regressor_fake_smoke"
    assert metrics["evaluated_splits"] == ["test"]
    assert metrics["requested_rows"] == 2
    assert metrics["scored_rows"] == 1
    assert metrics["coverage"] == 0.5
    assert metrics["llm_errors"] == 1
    assert metrics["test"]["micro"]["mae"] == 0.0
    assert "val" not in metrics

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "LLMRegressor"
    assert manifest["scorer"]["client_name"] == "fake"
    assert manifest["scorer"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert manifest["scorer"]["target"]["valid_values"] == [1, 2, 3, 4, 5]
    assert manifest["limits"] == {
        "max_rows": 2,
        "max_llm_attempts": 2,
        "max_workers": 1,
    }
    assert manifest["evaluated_splits"] == ["test"]

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["split"].tolist() == ["test", "test"]
    assert predictions["score"].iloc[0] == 4.0
    assert pd.isna(predictions["score"].iloc[1])
    assert "prediction" not in predictions.columns

    error_lines = (tmp_path / "llm_errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(error_lines) == 1
    error = json.loads(error_lines[0])
    assert error["row_index"] == "b"
    assert error["user_id"] == "u2"
    assert error["item_id"] == "i5"
    assert error["attempts"] == 2
    assert "target" not in error

    first_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "H1. item_title: Toy Story; item_genres: Animation; rating: 5" in first_prompt
    assert "Candidate. item_title: Lion King; item_genres: Animation" in first_prompt
    assert "Return exactly one integer" in first_prompt


def test_llm_regression_runner_requires_item_stats_columns_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Task(
        name="ml-1m_rating_eval_users1_seed0",
        train=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i1"],
                "item_title": ["Toy Story"],
                "item_genres": ["Animation"],
                "rating": [5],
                "target": [5],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i2"],
                "item_title": ["Lion King"],
                "item_genres": ["Animation"],
                "rating": [pd.NA],
                "target": [4],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            feature_columns=("item_title", "item_genres"),
            history_context_columns=("rating",),
        ),
        manifest={
            "protocol": "regression",
            "dataset": "ml-1m",
            "target_source_column": "target_rating",
        },
    )
    monkeypatch.setattr(
        llm_regressor,
        "make_llm_client",
        lambda _client_name: FakeClient([]),
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        llm_regressor.run_method(
            task,
            tmp_path,
            method_name="llm_regressor_fake_with_item_stats_smoke",
            client_name="fake",
            model="fake-model",
            max_rows=1,
            max_llm_attempts=1,
            max_workers=1,
            use_item_stats=True,
        )


def test_llm_regression_runner_labels_rating_only_with_item_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Task(
        name="ml-1m_rating_item_stats_eval_users1_seed0",
        train=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i1"],
                "item_title": ["Toy Story"],
                "item_genres": ["Animation"],
                "item_rating_mean": [4.153],
                "item_rating_count": [2077],
                "rating": [5],
                "target": [5],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i2"],
                "item_title": ["Lion King"],
                "item_genres": ["Animation"],
                "item_rating_mean": [3.333],
                "item_rating_count": [0],
                "rating": [pd.NA],
                "target": [4],
            },
            index=["a"],
        ),
        schema=TaskSchema(
            target_column="target",
            feature_columns=(
                "item_title",
                "item_genres",
                "item_rating_mean",
                "item_rating_count",
            ),
            history_context_columns=("rating",),
        ),
        manifest={
            "protocol": "regression",
            "dataset": "ml-1m",
            "target_source_column": "target_rating",
        },
    )
    client = FakeClient(["4"])
    monkeypatch.setattr(
        llm_regressor,
        "make_llm_client",
        lambda _client_name: client,
    )

    llm_regressor.run_method(
        task,
        tmp_path,
        method_name="llm_regressor_fake_with_item_stats_smoke",
        client_name="fake",
        model="fake-model",
        max_rows=1,
        max_llm_attempts=1,
        max_workers=1,
        use_item_stats=True,
    )

    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert (
        "H1. item_title: Toy Story; item_genres: Animation; user rating: 5; "
        "average rating: 4.15; number of prior reviews: 2077"
    ) in prompt
    assert (
        "Candidate. item_title: Lion King; item_genres: Animation; "
        "average rating: 3.33; number of prior reviews: 0"
    ) in prompt

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["uses_item_stats"] is True
    assert manifest["scorer"]["column_labels"]["rating"] == "user rating"


def test_llm_regression_runner_can_add_item_summaries_to_history_and_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Task(
        name="ml-1m_rating_item_stats_eval_users1_seed0",
        train=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["1"],
                "item_title": ["Toy Story"],
                "item_genres": ["Animation"],
                "item_rating_mean": [4.153],
                "item_rating_count": [2077],
                "rating": [5],
                "target": [5],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["364"],
                "item_title": ["Lion King"],
                "item_genres": ["Animation"],
                "item_rating_mean": [3.333],
                "item_rating_count": [0],
                "rating": [pd.NA],
                "target": [4],
            },
            index=["a"],
        ),
        schema=TaskSchema(
            target_column="target",
            feature_columns=(
                "item_title",
                "item_genres",
                "item_rating_mean",
                "item_rating_count",
            ),
            history_context_columns=("rating",),
        ),
        manifest={
            "protocol": "regression",
            "dataset": "ml-1m",
            "target_source_column": "target_rating",
        },
    )
    client = FakeClient(["4"])
    monkeypatch.setattr(
        llm_regressor,
        "make_llm_client",
        lambda _client_name: client,
    )

    def fake_add_item_summaries(**kwargs: object):
        X_train = kwargs["X_train"].copy()
        X_test = kwargs["X_test"].copy()
        X_train["item_summary"] = ["Toys plan a rescue."]
        X_test["item_summary"] = ["A young lion reclaims his kingdom."]
        return X_train, X_test, {"uses_item_summaries": True, "source_path": "fake.csv"}

    monkeypatch.setattr(
        llm_regressor,
        "add_ml1m_item_summaries",
        fake_add_item_summaries,
    )

    llm_regressor.run_method(
        task,
        tmp_path,
        method_name="llm_regressor_fake_with_item_stats_summary_full",
        client_name="fake",
        model="fake-model",
        max_rows=1,
        max_llm_attempts=1,
        max_workers=1,
        use_item_stats=True,
        use_item_summaries=True,
    )

    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "summary: Toys plan a rescue." in prompt
    assert "summary: A young lion reclaims his kingdom." in prompt

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["item_summaries"] == {
        "uses_item_summaries": True,
        "source_path": "fake.csv",
    }


def test_llm_regressor_openai_vk_gpt54_mini_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(llm_regressor, "run_method", fake_run_method)
    task = SimpleNamespace()

    llm_regressor.run_gpt54_mini_full(task, tmp_path)
    llm_regressor.run_gpt54_mini_with_item_stats_full(task, tmp_path)
    llm_regressor.run_gpt54_mini_smoke(task, tmp_path)
    llm_regressor.run_gpt54_mini_with_item_stats_smoke(task, tmp_path)

    assert calls == [
        {
            "method_name": "llm_regressor_openai_vk_gpt54_mini_full",
            "client_name": "openai_vk_proxy",
            "model": "gpt-5.4-mini",
            "max_rows": None,
            "max_workers": llm_regressor.OPENAI_VK_MAX_WORKERS,
        },
        {
            "method_name": "llm_regressor_openai_vk_gpt54_mini_with_item_stats_full",
            "client_name": "openai_vk_proxy",
            "model": "gpt-5.4-mini",
            "max_rows": None,
            "max_workers": llm_regressor.OPENAI_VK_MAX_WORKERS,
            "use_item_stats": True,
        },
        {
            "method_name": "llm_regressor_openai_vk_gpt54_mini_smoke",
            "client_name": "openai_vk_proxy",
            "model": "gpt-5.4-mini",
            "max_rows": llm_regressor.SMOKE_ROWS,
            "max_workers": llm_regressor.OPENAI_VK_MAX_WORKERS,
        },
        {
            "method_name": (
                "llm_regressor_openai_vk_gpt54_mini_with_item_stats_smoke"
            ),
            "client_name": "openai_vk_proxy",
            "model": "gpt-5.4-mini",
            "max_rows": llm_regressor.SMOKE_ROWS,
            "max_workers": llm_regressor.OPENAI_VK_MAX_WORKERS,
            "use_item_stats": True,
        },
    ]


def test_llm_regressor_openai_vk_gpt55_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(llm_regressor, "run_method", fake_run_method)
    task = SimpleNamespace()

    llm_regressor.run_gpt55_full(task, tmp_path)
    llm_regressor.run_gpt55_with_item_stats_full(task, tmp_path)
    llm_regressor.run_gpt55_smoke(task, tmp_path)
    llm_regressor.run_gpt55_with_item_stats_smoke(task, tmp_path)

    assert calls == [
        {
            "method_name": "llm_regressor_openai_vk_gpt55_full",
            "client_name": "openai_vk_proxy",
            "model": "gpt-5.5",
            "max_rows": None,
            "max_workers": llm_regressor.OPENAI_VK_MAX_WORKERS,
        },
        {
            "method_name": "llm_regressor_openai_vk_gpt55_with_item_stats_full",
            "client_name": "openai_vk_proxy",
            "model": "gpt-5.5",
            "max_rows": None,
            "max_workers": llm_regressor.OPENAI_VK_MAX_WORKERS,
            "use_item_stats": True,
        },
        {
            "method_name": "llm_regressor_openai_vk_gpt55_smoke",
            "client_name": "openai_vk_proxy",
            "model": "gpt-5.5",
            "max_rows": llm_regressor.SMOKE_ROWS,
            "max_workers": llm_regressor.OPENAI_VK_MAX_WORKERS,
        },
        {
            "method_name": "llm_regressor_openai_vk_gpt55_with_item_stats_smoke",
            "client_name": "openai_vk_proxy",
            "model": "gpt-5.5",
            "max_rows": llm_regressor.SMOKE_ROWS,
            "max_workers": llm_regressor.OPENAI_VK_MAX_WORKERS,
            "use_item_stats": True,
        },
    ]


def test_llm_regressor_qwen3_8b_with_item_stats_wrappers_disable_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(llm_regressor, "run_method", fake_run_method)
    task = SimpleNamespace()

    llm_regressor.run_qwen3_8b_with_item_stats_full(task, tmp_path)
    llm_regressor.run_qwen3_8b_with_item_stats_summary_full(task, tmp_path)
    llm_regressor.run_qwen3_8b_with_item_stats_smoke(task, tmp_path)

    assert calls == [
        {
            "method_name": "llm_regressor_vllm_qwen3_8b_with_item_stats_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3-8B",
            "max_rows": None,
            "max_workers": 128,
            "use_item_stats": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": (
                "llm_regressor_vllm_qwen3_8b_with_item_stats_summary_full"
            ),
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3-8B",
            "max_rows": None,
            "max_workers": 128,
            "use_item_stats": True,
            "use_item_summaries": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": "llm_regressor_vllm_qwen3_8b_with_item_stats_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3-8B",
            "max_rows": llm_regressor.SMOKE_ROWS,
            "max_workers": 128,
            "use_item_stats": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ]
