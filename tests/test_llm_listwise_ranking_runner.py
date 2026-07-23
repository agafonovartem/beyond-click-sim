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

from runners.in_distribution.interaction_prediction.methods import (  # noqa: E402
    llm_listwise_ranking,
)


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


def test_llm_listwise_ranking_runner_writes_ranking_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(["1. C1\n2. C2", "1. C1\n2. C2"])
    monkeypatch.setattr(llm_listwise_ranking, "make_llm_client", lambda _: client)
    monkeypatch.setattr(llm_listwise_ranking, "repo_root", lambda: REPO_ROOT)

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
        val=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["v1", "v2"],
                "candidate_group": ["vg1", "vg1"],
                "item_title": ["Aladdin", "Casino"],
                "item_genres": ["Animation", "Crime"],
                "target": [1, 0],
            }
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "candidate_group": ["g1", "g1"],
                "item_title": ["Lion King", "Heat"],
                "item_genres": ["Animation", "Crime"],
                "target": [1, 0],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={"dataset": "ml-1m", "seed": 0},
    )

    result = llm_listwise_ranking.run_method(
        task,
        tmp_path,
        method_name="llm_listwise_ranking_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert result["protocol"] == "direct_listwise_ranking"
    assert result["llm_errors"] == {"val": 0, "test": 0}
    assert result["pointwise_threshold"] == 1.0
    assert (tmp_path / "metrics.json").exists()

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["split"].tolist() == ["val", "val", "test", "test"]
    assert predictions["score"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert predictions["prediction"].tolist() == [True, False, True, False]

    pointwise_metrics = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )
    assert pointwise_metrics["threshold"] == 1.0
    assert pointwise_metrics["test"]["micro"]["f1"] == 1.0

    ranking_metrics = json.loads(
        (tmp_path / "metrics_ranking.json").read_text(encoding="utf-8")
    )
    assert ranking_metrics["protocol"] == "direct_listwise_ranking"
    assert ranking_metrics["val"]["macro_by_group"]["ndcg@1"] == 1.0
    assert ranking_metrics["test"]["macro_by_group"]["ndcg@1"] == 1.0
    assert ranking_metrics["test"]["macro_by_group"]["groups_with_score_ties"] == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "listwise_ranking_with_validation_threshold"
    assert (
        manifest["decision_rule"]["ranking"]["kind"]
        == "complete_listwise_rank_parser"
    )
    assert (
        manifest["decision_rule"]["pointwise"]["kind"]
        == "threshold_on_validation"
    )
    assert manifest["scorer"]["class"] == "LLMInteractionListwiseRankingScorer"

    user_prompt = client.completions.calls[1]["messages"][1]["content"]
    assert "H1. item_title: Toy Story; item_genres: Animation; rating: 5" in user_prompt
    assert "C1. item_title: Lion King; item_genres: Animation" in user_prompt
    assert "C2. item_title: Heat; item_genres: Crime" in user_prompt


def test_llm_listwise_ranking_runner_formats_steam_json_lists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(["1. C2\n2. C1", "1. C2\n2. C1"])
    monkeypatch.setattr(llm_listwise_ranking, "make_llm_client", lambda _: client)
    monkeypatch.setattr(llm_listwise_ranking, "repo_root", lambda: REPO_ROOT)

    task = Task(
        name="steam-toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["g-train"],
                "item_title": ["Portal"],
                "item_genres_json": ['["Action", "Adventure"]'],
                "item_tags_json": ['["Puzzle", "Singleplayer"]'],
                "playtime_forever": [120],
                "target": [1],
            }
        ),
        val=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["v1", "v2"],
                "candidate_group": ["vcg1", "vcg1"],
                "item_title": ["Half-Life", "Team Fortress 2"],
                "item_genres_json": [
                    '["Action"]',
                    '["Action", "Free to Play"]',
                ],
                "item_tags_json": [
                    '["Singleplayer"]',
                    '["Multiplayer"]',
                ],
                "target": [0, 1],
            }
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["g1", "g2"],
                "candidate_group": ["cg1", "cg1"],
                "item_title": ["Portal 2", "Dota 2"],
                "item_genres_json": [
                    '["Action", "Adventure"]',
                    '["Action", "Free to Play"]',
                ],
                "item_tags_json": [
                    '["Puzzle", "Co-op"]',
                    '["MOBA", "Multiplayer"]',
                ],
                "target": [1, 0],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={"dataset": "steam", "seed": 0},
    )

    llm_listwise_ranking.run_method(
        task,
        tmp_path,
        method_name="llm_listwise_ranking_steam_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "game title: Portal" in user_prompt
    assert "genres: Action, Adventure" in user_prompt
    assert "tags: Puzzle, Singleplayer" in user_prompt
    assert "user playtime minutes: 120" in user_prompt
    assert "item_genres_json" not in user_prompt
    assert "item_tags_json" not in user_prompt

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["json_list_columns"] == [
        "item_genres_json",
        "item_tags_json",
    ]
