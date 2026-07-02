from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers import Agent4RecProfileGenerator, Agent4RecYesNoScorer
from beyond_click_sim.scorers.agent4rec.prompts import (
    AGENT4REC_FORCED_ITEMS_SYSTEM_PROMPT_TEMPLATE,
    AGENT4REC_SOCIAL_TRAITS_SYSTEM_PROMPT_TEMPLATE,
    agent4rec_system_prompt,
    agent4rec_user_prompt,
)
from beyond_click_sim.scorers.agent4rec.yes_no import parse_agent4rec_watch_response


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


def test_parse_agent4rec_watch_response_maps_reordered_decisions_by_label() -> None:
    parsed = parse_agent4rec_watch_response(
        "ID: C2; MOVIE: B; WATCH: No; REASON: not ok\n"
        "ID: C1; MOVIE: A; WATCH: yes; REASON: ok",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 1.0, "C2": 0.0}


def test_parse_agent4rec_watch_response_accepts_qwen_label_prefix() -> None:
    parsed = parse_agent4rec_watch_response(
        "C1: [1]; MOVIE: A; WATCH: yes; REASON: ok\n"
        "C2: [2]; MOVIE: B; WATCH: no; REASON: not ok",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 1.0, "C2": 0.0}


def test_parse_agent4rec_watch_response_accepts_game_field() -> None:
    parsed = parse_agent4rec_watch_response(
        "ID: C1; GAME: Portal; WATCH: yes; REASON: puzzle game\n"
        "ID: C2; GAME: Dota 2; WATCH: no; REASON: not preferred",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 1.0, "C2": 0.0}


def test_parse_agent4rec_watch_response_requires_all_decisions() -> None:
    with pytest.raises(ValueError, match="Missing Agent4Rec watch decisions"):
        parse_agent4rec_watch_response(
            "ID: C1; MOVIE: A; WATCH: yes; REASON: ok",
            labels=["C1", "C2"],
        )


def test_parse_agent4rec_watch_response_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="Duplicate Agent4Rec candidate labels"):
        parse_agent4rec_watch_response(
            "ID: C1; MOVIE: A; WATCH: yes; REASON: ok\n"
            "ID: C1; MOVIE: B; WATCH: no; REASON: not ok",
            labels=["C1", "C2"],
        )


def test_parse_agent4rec_watch_response_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unknown Agent4Rec candidate labels"):
        parse_agent4rec_watch_response(
            "ID: C1; MOVIE: A; WATCH: yes; REASON: ok\n"
            "ID: C3; MOVIE: B; WATCH: no; REASON: not ok",
            labels=["C1", "C2"],
        )


def test_agent4rec_system_prompt_matches_full_social_traits_template() -> None:
    prompt = agent4rec_system_prompt(
        taste="animated comedies; family movies",
        activity="activity description",
        conformity="conformity description",
        diversity="diversity description",
    )

    assert prompt == AGENT4REC_SOCIAL_TRAITS_SYSTEM_PROMPT_TEMPLATE.format(
        taste="animated comedies; family movies",
        activity="activity description",
        conformity="conformity description",
        diversity="diversity description",
    )


def test_agent4rec_system_prompt_omits_missing_profile_parts() -> None:
    prompt = agent4rec_system_prompt(
        taste=None,
        activity="activity description",
        conformity=None,
        diversity="diversity description",
    )

    assert "Your activity trait is described as: activity description" in prompt
    assert "Your conformity trait is described as:" not in prompt
    assert "Your diversity trait is described as: diversity description" in prompt
    assert "Beyond that, your movie tastes are:" not in prompt
    assert "The conformity characteristic measures" not in prompt


def test_agent4rec_system_prompt_uses_forced_items_template_for_taste_only() -> None:
    prompt = agent4rec_system_prompt(
        taste="animated comedies; family movies",
        activity=None,
        conformity=None,
        diversity=None,
    )

    assert prompt == AGENT4REC_FORCED_ITEMS_SYSTEM_PROMPT_TEMPLATE.format(
        taste="animated comedies; family movies"
    )


def test_agent4rec_user_prompt_uses_taste_instruction_only_when_taste_exists() -> None:
    traits_only_prompt = agent4rec_user_prompt(
        candidates="C1. <- Movie ->",
        taste=None,
    )
    taste_prompt = agent4rec_user_prompt(
        candidates="C1. <- Movie ->",
        taste="animated comedies",
    )

    assert "Judge each movie using your available profile" in traits_only_prompt
    assert "You only watch movies which align with your taste" not in traits_only_prompt
    assert "You only watch movies which align with your taste" in taste_prompt


def test_agent4rec_user_prompt_can_use_game_field() -> None:
    prompt = agent4rec_user_prompt(
        candidates="C1. <- Portal ->",
        taste=None,
        entity_field="GAME",
        entity_name="game",
        entity_plural="games",
    )

    assert "Please judge all games in the ##recommended list##" in prompt
    assert "Use this format: ID: [candidate id]; GAME: [game name]; WATCH:" in prompt


def test_agent4rec_yes_no_scorer_uses_profile_prompt() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genres": ["Animation|Comedy", "Animation", "Crime"],
            "rating": [5, 4, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Lion King", "Godfather"],
            "item_rating_mean": [4.153, 4.567],
            "item_genres": ["Animation", "Crime"],
        },
        index=["a", "b"],
    )
    client = FakeClient(
        [
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: crime movies do not fit the user's taste\n"
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated movies fit the user's taste"
        ]
    )

    scorer = Agent4RecYesNoScorer(
        client=client,
        model="fake-model",
        candidate_description_columns=("item_title", "item_rating_mean", "item_genres"),
        column_labels={
            "item_title": "movie title",
            "item_rating_mean": "History ratings",
            "item_genres": "genres",
        },
    ).fit(X_train, pd.Series([1, 1, 1], name="target"))
    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 1.0, "b": 0.0}
    assert len(client.completions.calls) == 1
    system_prompt = client.completions.calls[0]["messages"][0]["content"]
    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "You excel at role-playing." in system_prompt
    assert "Your activity trait is described as:" in system_prompt
    assert "Your conformity trait is described as:" in system_prompt
    assert "Your diversity trait is described as:" in system_prompt
    assert "##recommended list##" in user_prompt
    assert "C1. <- Lion King -> <- History ratings:4.15 -> <- genres:Animation ->" in user_prompt
    assert "C2. <- Godfather -> <- History ratings:4.57 -> <- genres:Crime ->" in user_prompt
    assert "Use this format: ID: [candidate id]; MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]" in user_prompt
    assert "Judge each movie using your available profile" in user_prompt
    assert "You only watch movies which align with your taste" not in user_prompt
    assert "rating from 1 to 5" not in user_prompt
    assert "Toy Story" not in user_prompt


def test_agent4rec_yes_no_scorer_requires_build_taste_before_score(tmp_path) -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genres": ["Animation|Comedy", "Animation", "Crime"],
            "rating": [5, 4, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_group": ["g1"],
            "item_title": ["Lion King"],
            "item_rating_mean": [4.153],
            "item_genres": ["Animation"],
        }
    )
    scoring_client = FakeClient([])
    taste_client = FakeClient([])
    profile_generator = Agent4RecProfileGenerator(
        profile_components=("traits", "taste"),
        taste_client=taste_client,
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=tmp_path / "taste.jsonl",
    )
    scorer = Agent4RecYesNoScorer(
        client=scoring_client,
        model="fake-model",
        profile_generator=profile_generator,
        candidate_description_columns=("item_title", "item_rating_mean", "item_genres"),
    ).fit(X_train, pd.Series([1, 1, 1], name="target"))

    with pytest.raises(RuntimeError, match="Call scorer.build_taste"):
        scorer.score(X_test)

    assert scoring_client.completions.calls == []
    assert taste_client.completions.calls == []


def test_agent4rec_yes_no_scorer_build_taste_then_scores_with_taste(tmp_path) -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genres": ["Animation|Comedy", "Animation", "Crime"],
            "rating": [5, 4, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Lion King", "Godfather"],
            "item_rating_mean": [4.153, 4.567],
            "item_genres": ["Animation", "Crime"],
        },
        index=["a", "b"],
    )
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
    profile_generator = Agent4RecProfileGenerator(
        profile_components=("traits", "taste"),
        taste_client=taste_client,
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=tmp_path / "taste.jsonl",
    )
    scorer = Agent4RecYesNoScorer(
        client=scoring_client,
        model="fake-model",
        profile_generator=profile_generator,
        candidate_description_columns=("item_title", "item_rating_mean", "item_genres"),
        column_labels={
            "item_title": "movie title",
            "item_rating_mean": "History ratings",
            "item_genres": "genres",
        },
    ).fit(X_train, pd.Series([1, 1, 1], name="target"))

    scorer.build_taste(X_test)
    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 1.0, "b": 0.0}
    assert len(taste_client.completions.calls) == 1
    taste_messages = taste_client.completions.calls[0]["messages"]
    assert "user gives a rating of 5 for following movies: Toy Story" in taste_messages[1]["content"]
    assert "user gives a rating of 2 for following movies: Heat" in taste_messages[1]["content"]
    system_prompt = scoring_client.completions.calls[0]["messages"][0]["content"]
    user_prompt = scoring_client.completions.calls[0]["messages"][1]["content"]
    assert "Beyond that, your movie tastes are: enjoy animated family movies." in system_prompt
    assert "You only watch movies which align with your taste" in user_prompt
