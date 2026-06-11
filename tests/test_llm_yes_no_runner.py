from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

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
