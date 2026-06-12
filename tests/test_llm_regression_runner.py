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
