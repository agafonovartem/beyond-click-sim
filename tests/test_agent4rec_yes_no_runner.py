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


def _ml1m_summary_task() -> Task:
    enrichment = {
        "movie_summaries": {
            "enabled": True,
            "canonical_column": "summary",
            "task_column": "item_summary",
            "source_sha256": "test-summary-source-sha256",
        }
    }
    return Task(
        name="ml-1m_summary_toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["h1", "h2"],
                "item_title": ["Toy Story", "Heat"],
                "item_genres": ["Animation|Comedy", "Crime"],
                "item_rating_mean": [4.15, 3.60],
                "item_summary": ["Toys plan a rescue.", "A crime thriller."],
                "rating": [5, 2],
                "target": [1, 1],
            }
        ),
        val=pd.DataFrame(
            columns=["user_id", "item_id", "item_summary", "target"]
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "candidate_group": ["g1", "g1"],
                "item_title": ["Lion King", "Godfather"],
                "item_genres": ["Animation", "Crime"],
                "item_rating_mean": [4.153, 4.567],
                "item_summary": [
                    "A young lion reclaims his kingdom.",
                    "A mafia family passes power to a son.",
                ],
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
        manifest={
            "dataset": "ml-1m",
            "dataset_version": "v1",
            "splitter": {"seed": 0},
            "item_enrichment": enrichment,
        },
    )


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
        summary_usage="none",
    )

    assert result["scored_rows"] == 2
    assert result["requested_rows"] == 2
    assert result["llm_errors"] == 0
    assert result["test_failure_as_negative"]["micro"]["n"] == 2
    assert result["coverage"] == {
        "requested_rows": 2,
        "scored_rows": 2,
        "failed_rows": 0,
        "scored_fraction": 1.0,
        "failed_fraction": 0.0,
    }

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
    ranking_metrics = json.loads(
        (tmp_path / "metrics_ranking.json").read_text(encoding="utf-8")
    )
    assert ranking_metrics["test_failure_as_zero_group"]["macro_by_group"][
        "failed_groups"
    ] == 0

    system_prompt = client.completions.calls[0]["messages"][0]["content"]
    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "Your activity trait is described as:" in system_prompt
    assert "##recommended list##" in user_prompt
    assert "C1. <- Lion King -> <- History ratings:4.15 -> <- genres:Animation ->" in user_prompt
    assert "Use this format: ID: [candidate id]; MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]" in user_prompt


def test_agent4rec_yes_no_runner_writes_taste_manifest_and_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scoring_client = FakeClient(
        [
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated taste\n"
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: not animated"
        ]
    )
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

    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", fake_make_llm_client)
    cache_path = tmp_path / "taste-cache.jsonl"
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
        manifest={
            "dataset": "ml-1m",
            "dataset_version": "v1",
            "splitter": {"seed": 0},
        },
    )

    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="agent4rec_yes_no_traits_taste_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        profile_components=("traits", "taste"),
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=cache_path,
        summary_usage="none",
    )

    assert result["scored_rows"] == 2
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
    assert profile_manifest["taste"]["temperature"] == 0.0
    assert profile_manifest["taste"]["max_tokens"] is None
    assert profile_manifest["taste"]["cache_path"] == str(cache_path)
    assert profile_manifest["taste"]["cache_stats"] == {
        "requested_users": 1,
        "hits": 0,
        "misses": 1,
        "generated": 1,
        "max_workers": 1,
    }


def test_agent4rec_yes_no_default_adds_summary_to_candidate_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(
        [
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated taste\n"
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: not animated"
        ]
    )
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    result = agent4rec_yes_no.run_method(
        _ml1m_summary_task(),
        tmp_path,
        method_name="agent4rec_yes_no_candidate_summary_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert result["scored_rows"] == 2
    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "<- summary:A young lion reclaims his kingdom. ->" in prompt
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["summary_usage"] == "candidate"
    assert manifest["scorer"]["candidate_description_columns"][-1] == (
        "item_summary"
    )
    assert manifest["scorer"]["item_summaries"] == {
        "uses_item_summaries": True,
        "summary_column": "item_summary",
        "history_item_summaries": False,
        "profile_item_summaries": False,
        "candidate_item_summaries": True,
        "canonical_enrichment": _ml1m_summary_task().manifest["item_enrichment"],
    }


def test_agent4rec_yes_no_both_adds_summaries_to_taste_and_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scoring_client = FakeClient(
        [
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated taste\n"
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: not animated"
        ]
    )
    taste_client = FakeClient(
        [
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I rated animated movies highly.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies"
        ]
    )

    def fake_make_llm_client(client_name: str) -> FakeClient:
        return taste_client if client_name == "openai" else scoring_client

    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", fake_make_llm_client)
    cache_path = tmp_path / "taste-summary-cache.jsonl"

    result = agent4rec_yes_no.run_method(
        _ml1m_summary_task(),
        tmp_path,
        method_name="agent4rec_yes_no_both_summary_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        profile_components=("traits", "taste"),
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=cache_path,
        summary_usage="both",
    )

    assert result["scored_rows"] == 2
    taste_prompt = taste_client.completions.calls[0]["messages"][1]["content"]
    assert "Toy Story (genres: Animation, Comedy; summary: Toys plan a rescue.)" in (
        taste_prompt
    )
    scoring_prompt = scoring_client.completions.calls[0]["messages"][1]["content"]
    assert "<- summary:A young lion reclaims his kingdom. ->" in scoring_prompt
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["summary_usage"] == "both"
    assert manifest["scorer"]["profile_generator"]["summary_column"] == (
        "item_summary"
    )
    assert manifest["scorer"]["item_summaries"]["profile_item_summaries"] is True
    assert manifest["scorer"]["item_summaries"]["candidate_item_summaries"] is True


def test_agent4rec_yes_no_profile_summaries_require_taste(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires 'taste'"):
        agent4rec_yes_no.run_method(
            _ml1m_summary_task(),
            tmp_path,
            method_name="agent4rec_yes_no_invalid_profile_summary_test",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            summary_usage="profile",
        )


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
            summary_usage="none",
        )


def test_agent4rec_yes_no_runner_supports_steam_traits_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(
        [
            "ID: C1; GAME: Portal 2; WATCH: yes; REASON: puzzle games fit\n"
            "ID: C2; GAME: Dota 2; WATCH: no; REASON: not aligned"
        ]
    )
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    task = Task(
        name="steam_toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["g1", "g2", "g3"],
                "item_title": ["Portal", "Half-Life", "Stardew Valley"],
                "item_genres_json": [
                    '["Action"]',
                    '["Action"]',
                    '["RPG", "Simulation"]',
                ],
                "item_tags_json": [
                    '["Puzzle", "Singleplayer"]',
                    '["FPS", "Story Rich"]',
                    '["Farming Sim", "RPG"]',
                ],
                "playtime_forever": [120, 240, 600],
                "target": [1, 1, 1],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["g4", "g5"],
                "candidate_group": ["g1", "g1"],
                "item_title": ["Portal 2", "Dota 2"],
                "item_genres_json": [
                    '["Action", "Adventure"]',
                    '["Action", "Free to Play"]',
                ],
                "item_tags_json": [
                    '["Puzzle", "Co-op"]',
                    '["MOBA", "Multiplayer"]',
                ],
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
        manifest={"dataset": "steam", "dataset_version": "v1", "splitter": {"seed": 0}},
    )

    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="agent4rec_yes_no_steam_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        summary_usage="none",
    )

    assert result["scored_rows"] == 2
    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["score"].tolist() == [1.0, 0.0]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    profile_manifest = manifest["scorer"]["profile_generator"]
    assert profile_manifest["genre_column"] == "item_genres_json"
    assert profile_manifest["include_conformity"] is False
    assert "conformity" not in profile_manifest["trait_thresholds"]
    assert manifest["scorer"]["prompt"]["entity_field"] == "GAME"
    assert manifest["scorer"]["json_list_columns"] == [
        "item_genres_json",
        "item_tags_json",
    ]

    system_prompt = client.completions.calls[0]["messages"][0]["content"]
    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "game recommendation system" in system_prompt
    assert "Your conformity trait is described as:" not in system_prompt
    assert (
        "C1. <- Portal 2 -> <- genres:Action, Adventure -> "
        "<- tags:Puzzle, Co-op ->"
    ) in user_prompt
    assert (
        "C2. <- Dota 2 -> <- genres:Action, Free to Play -> "
        "<- tags:MOBA, Multiplayer ->"
    ) in user_prompt
    assert '["Action", "Adventure"]' not in user_prompt
    assert "Use this format: ID: [candidate id]; GAME: [game name]; WATCH:" in user_prompt
    assert "Judge each game using your available profile" in user_prompt
    assert "Judge each movie using your available profile" not in user_prompt


def test_agent4rec_yes_no_runner_supports_steam_taste_profiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scoring_client = FakeClient(
        ["ID: C1; GAME: Portal 2; WATCH: yes; REASON: puzzle taste"]
    )
    taste_client = FakeClient(
        [
            "TASTE: I enjoy puzzle games.\n"
            "REASON: I spent time with puzzle games.\n"
            "HIGH PLAYTIME: puzzle games\n"
            "LOW PLAYTIME: competitive games"
        ]
    )

    def fake_make_llm_client(client_name: str) -> FakeClient:
        if client_name == "openai":
            return taste_client
        if client_name == "fake":
            return scoring_client
        raise AssertionError(client_name)

    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", fake_make_llm_client)
    cache_path = tmp_path / "steam-taste.jsonl"
    task = Task(
        name="steam_toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["g1"],
                "item_title": ["Portal"],
                "item_genres_json": ['["Action"]'],
                "item_tags_json": ['["Puzzle"]'],
                "playtime_forever": [120],
                "target": [1],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["g2"],
                "candidate_group": ["g1"],
                "item_title": ["Portal 2"],
                "item_genres_json": ['["Action"]'],
                "item_tags_json": ['["Puzzle"]'],
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
        manifest={"dataset": "steam", "dataset_version": "v1", "splitter": {"seed": 0}},
    )

    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="agent4rec_yes_no_steam_taste_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        profile_components=("taste",),
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=cache_path,
        summary_usage="none",
    )

    assert result["scored_rows"] == 1
    assert len(taste_client.completions.calls) == 1
    taste_prompt = taste_client.completions.calls[0]["messages"][1]["content"]
    assert "game playtime history" in taste_prompt
    assert "Portal (genres: Action; tags: Puzzle)" in taste_prompt
    scoring_prompt = scoring_client.completions.calls[0]["messages"][1]["content"]
    assert "C1. <- Portal 2 -> <- genres:Action -> <- tags:Puzzle ->" in scoring_prompt
    assert '["Action"]' not in scoring_prompt
    assert "You only choose games which align with your taste" in scoring_prompt
    assert "You only watch movies which align with your taste" not in scoring_prompt
    assert "If you don't want to choose a game" in scoring_prompt
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    taste_manifest = manifest["scorer"]["profile_generator"]["taste"]
    assert taste_manifest["prompt_kind"] == "playtime"
    assert taste_manifest["prompt_version"] == "agent4rec_playtime_v1"


def test_agent4rec_qwen_port_wrappers_use_port_clients(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_yes_no, "run_method", fake_run_method)
    task = SimpleNamespace(manifest=_ml1m_summary_task().manifest)

    agent4rec_yes_no.run_qwen36_27b_port8001_full(task, tmp_path)
    agent4rec_yes_no.run_qwen36_27b_port8002_full(task, tmp_path)
    agent4rec_yes_no.run_qwen36_27b_port8001_smoke(task, tmp_path)
    agent4rec_yes_no.run_qwen36_27b_port8002_smoke(task, tmp_path)

    assert calls == [
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_port8001_full",
            "client_name": "vllm_local_8001",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_usage": "candidate",
        },
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_port8002_full",
            "client_name": "vllm_local_8002",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_usage": "candidate",
        },
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_port8001_smoke",
            "client_name": "vllm_local_8001",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_usage": "candidate",
        },
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_port8002_smoke",
            "client_name": "vllm_local_8002",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "summary_usage": "candidate",
        },
    ]


def test_agent4rec_qwen_traits_taste_wrappers_use_openai_taste(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_yes_no, "run_method", fake_run_method)
    task = SimpleNamespace(manifest=_ml1m_summary_task().manifest)

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
            "taste_temperature": 0.0,
            "taste_max_tokens": None,
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
            "taste_temperature": 0.0,
            "taste_max_tokens": None,
            "summary_usage": "candidate",
        },
    ]


def test_agent4rec_qwen3_8b_candidate_summary_wrapper_uses_litellm_and_taste(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_yes_no, "run_method", fake_run_method)
    monkeypatch.setattr(
        agent4rec_yes_no,
        "_serving_metadata",
        lambda: {"backend": "test"},
    )
    monkeypatch.setattr(
        agent4rec_yes_no,
        "_source_metadata",
        lambda: {"snapshot": "test"},
    )
    task = SimpleNamespace(manifest=_ml1m_summary_task().manifest)

    agent4rec_yes_no.run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke(
        task,
        tmp_path,
    )

    assert calls == [
        {
            "method_name": (
                "agent4rec_yes_no_litellm_qwen3_8b_traits_taste_"
                "gpt4o_mini_candidate_summary_smoke"
            ),
            "client_name": "litellm_local",
            "model": "Qwen/Qwen3-8B",
            "max_candidate_groups": 25,
            "max_workers": 64,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "profile_components": ("traits", "taste"),
            "taste_client_name": "openai",
            "taste_model": "gpt-4o-mini",
            "taste_temperature": 0.0,
            "taste_max_tokens": None,
            "summary_usage": "candidate",
            "serving_metadata": {"backend": "test"},
            "source_metadata": {"snapshot": "test"},
        }
    ]
