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

from runners.in_distribution.interaction_prediction.methods import agent4rec_yes_no  # noqa: E402


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


def test_agent4rec_yes_no_runner_writes_profile_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(
        [
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: crime movies do not fit the user's taste\n"
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated movies fit the user's taste"
        ]
    )
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    task = Task(
        name="toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["i-train-1", "i-train-2", "i-train-3"],
                "item_title": ["Toy Story", "Aladdin", "Heat"],
                "item_genres": ["Animation|Comedy", "Animation", "Crime"],
                "item_rating_mean": [4.15, 3.95, 3.60],
                "rating": [5, 4, 2],
                "target": [1, 1, 1],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "candidate_group": ["g1", "g1"],
                "item_title": ["Lion King", "Godfather"],
                "item_genres": ["Animation", "Crime"],
                "item_rating_mean": [4.153, 4.567],
                "sampled": [False, True],
                "target": [1, 0],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            sampled_column="sampled",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={"dataset": "ml-1m", "seed": 0},
    )

    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="agent4rec_yes_no_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert result["scored_rows"] == 2
    assert result["requested_rows"] == 2
    assert result["llm_errors"] == 0

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["score"].tolist() == [1.0, 0.0]
    assert predictions["prediction"].tolist() == [True, False]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "Agent4RecYesNoScorer"
    assert manifest["scorer"]["profile_generator"]["profile_components"] == ["traits"]
    assert manifest["scorer"]["profile_generator"]["trait_thresholds"] is not None
    assert manifest["scorer"]["profile_generator"]["diversity_top_mass"] == 0.8
    assert manifest["scorer"]["candidate_description_columns"] == [
        "item_title",
        "item_rating_mean",
        "item_genres",
    ]
    assert (
        manifest["decision_rule"]["parser_contract"]
        == "agent4rec_labeled_id_movie_watch_reason"
    )

    system_prompt = client.completions.calls[0]["messages"][0]["content"]
    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "Your activity trait is described as:" in system_prompt
    assert "##recommended list##" in user_prompt
    assert "C1. <- Lion King -> <- History ratings:4.15 -> <- genres:Animation ->" in user_prompt
    assert "Use this format: ID: [candidate id]; MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]" in user_prompt


def test_agent4rec_yes_no_runner_requires_item_rating_mean(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: FakeClient([]))
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
                "sampled": [False],
                "target": [1],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            sampled_column="sampled",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={"dataset": "ml-1m", "seed": 0},
    )

    with pytest.raises(ValueError, match="requires item-stats task columns"):
        agent4rec_yes_no.run_method(
            task,
            tmp_path,
            method_name="agent4rec_yes_no_test",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            max_llm_attempts=1,
            max_workers=1,
        )
