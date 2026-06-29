from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers import Agent4RecYesNoScorer
from beyond_click_sim.scorers.agent4rec.prompts import (
    AGENT4REC_FORCED_ITEMS_SYSTEM_PROMPT_TEMPLATE,
    AGENT4REC_SOCIAL_TRAITS_SYSTEM_PROMPT_TEMPLATE,
    agent4rec_system_prompt,
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
    assert "rating from 1 to 5" not in user_prompt
    assert "Toy Story" not in user_prompt
