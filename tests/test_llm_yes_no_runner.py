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

from runners.in_distribution.interaction_prediction.methods import llm_yes_no  # noqa: E402


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


def test_llm_yes_no_runner_retries_and_keeps_failed_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(
        [
            "C1: yes",
            "C1: yes\nC2: no",
            "C1: maybe\nC2: no",
            "C1: maybe\nC2: no",
        ]
    )
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: client)

    task = Task(
        name="toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "item_id": ["i-train-1", "i-train-2"],
                "item_title": ["Toy Story", "Portal 2"],
                "item_genres": ["Animation", "Puzzle"],
                "rating": [5, 4],
                "target": [1, 1],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "item_id": ["i1", "i2", "i3", "i4"],
                "candidate_group": ["g1", "g1", "g2", "g2"],
                "item_title": ["Lion King", "Heat", "Half-Life", "Doom"],
                "item_genres": ["Animation", "Crime", "Action", "Action"],
                "target": [1, 0, 1, 0],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={"dataset": "ml-1m", "seed": 0},
    )

    result = llm_yes_no.run_method(
        task,
        tmp_path,
        method_name="llm_yes_no_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=2,
        max_workers=1,
    )

    assert len(client.completions.calls) == 4
    assert result["llm_errors"] == 1
    assert result["scored_rows"] == 2
    assert result["requested_rows"] == 4
    assert result["candidate_groups"]["requested"]["groups"] == 2
    assert result["candidate_groups"]["scored"]["groups"] == 1
    assert result["test"]["micro"]["n"] == 2

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(predictions) == 4
    assert predictions.loc[0, "score"] == 1.0
    assert predictions.loc[1, "score"] == 0.0
    assert predictions.loc[2:, "score"].isna().all()
    assert predictions.loc[:1, "prediction"].tolist() == [True, False]
    assert predictions.loc[2:, "prediction"].isna().all()

    errors = [
        json.loads(line)
        for line in (tmp_path / "llm_errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert errors == [
        {
            "attempts": 2,
            "candidate_group": "g2",
            "errors": [
                "ValueError(\"Invalid yes/no answer for 'C1': 'maybe'\")",
                "ValueError(\"Invalid yes/no answer for 'C1': 'maybe'\")",
            ],
        }
    ]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["limits"]["max_llm_attempts"] == 2
    assert "max_llm_errors" not in manifest["limits"]

    first_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "H1. item_title: Toy Story; item_genres: Animation; rating: 5" in first_prompt
    assert "user rating" not in first_prompt


def test_llm_yes_no_runner_requires_item_stats_columns_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: FakeClient([]))

    task = Task(
        name="toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i-train-1"],
                "item_title": ["Toy Story"],
                "item_genres": ["Animation"],
                "rating": [5],
                "target": [1],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i1"],
                "candidate_group": ["g1"],
                "item_title": ["Lion King"],
                "item_genres": ["Animation"],
                "target": [1],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={"dataset": "ml-1m", "seed": 0},
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        llm_yes_no.run_method(
            task,
            tmp_path,
            method_name="llm_yes_no_test_with_item_stats",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            max_llm_attempts=1,
            max_workers=1,
            use_item_stats=True,
        )


def test_llm_yes_no_qwen_wrappers_disable_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(llm_yes_no, "run_method", fake_run_method)
    task = SimpleNamespace()

    llm_yes_no.run_qwen36_27b_full(task, tmp_path)
    llm_yes_no.run_qwen36_27b_with_item_stats_full(task, tmp_path)
    llm_yes_no.run_qwen36_27b_smoke(task, tmp_path)
    llm_yes_no.run_qwen36_27b_with_item_stats_smoke(task, tmp_path)

    assert calls == [
        {
            "method_name": "llm_yes_no_vllm_qwen36_27b_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_27b_with_item_stats_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "use_item_stats": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_27b_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_27b_with_item_stats_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "use_item_stats": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ]
