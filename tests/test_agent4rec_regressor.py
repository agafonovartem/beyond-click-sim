from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers import Agent4RecProfileGenerator, Agent4RecRegressor
from beyond_click_sim.scorers.agent4rec.prompts import (
    agent4rec_rating_user_prompt,
)
from beyond_click_sim.scorers.agent4rec.regression import (
    parse_agent4rec_rating_response,
)


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


def test_parse_agent4rec_rating_response_accepts_rating_line() -> None:
    assert parse_agent4rec_rating_response(
        "RATING: 4",
        valid_values=(1, 2, 3, 4, 5),
    ) == 4.0


def test_parse_agent4rec_rating_response_accepts_bare_integer() -> None:
    assert parse_agent4rec_rating_response(
        " 5 \n",
        valid_values=(1, 2, 3, 4, 5),
    ) == 5.0


@pytest.mark.parametrize("text", ["RATING: 6", "RATING: 0", "bad"])
def test_parse_agent4rec_rating_response_rejects_invalid_values(text: str) -> None:
    with pytest.raises(ValueError):
        parse_agent4rec_rating_response(text, valid_values=(1, 2, 3, 4, 5))


def test_parse_agent4rec_rating_response_rejects_multiple_ratings() -> None:
    with pytest.raises(ValueError, match="multiple RATING"):
        parse_agent4rec_rating_response(
            "RATING: 4\nRATING: 5",
            valid_values=(1, 2, 3, 4, 5),
        )


def test_agent4rec_rating_user_prompt_uses_taste_specific_instruction() -> None:
    traits_prompt = agent4rec_rating_user_prompt(
        candidate="<- Movie ->",
        taste=None,
    )
    taste_prompt = agent4rec_rating_user_prompt(
        candidate="<- Movie ->",
        taste="animated comedies",
    )

    assert "Rate the movie using your available profile" in traits_prompt
    assert "Rate the movie according to your taste" not in traits_prompt
    assert "Rate the movie according to your taste" in taste_prompt
    assert "Use this format: RATING: [integer from 1 to 5]" in taste_prompt


def test_agent4rec_regressor_uses_profile_rating_prompt() -> None:
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
            "item_title": ["Lion King", "Godfather"],
            "item_rating_mean": [4.153, 4.567],
            "item_genres": ["Animation", "Crime"],
        },
        index=["a", "b"],
    )
    client = FakeClient(["RATING: 5", "RATING: 2"])

    scorer = Agent4RecRegressor(
        client=client,
        model="fake-model",
        target_description="Predict the integer rating.",
        valid_values=(1, 2, 3, 4, 5),
        candidate_description_columns=("item_title", "item_rating_mean", "item_genres"),
        column_labels={
            "item_title": "movie title",
            "item_rating_mean": "History ratings",
            "item_genres": "genres",
        },
    ).fit(X_train, pd.Series([5, 4, 2], name="target"))
    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 5.0, "b": 2.0}
    assert len(client.completions.calls) == 2
    first_messages = client.completions.calls[0]["messages"]
    system_prompt = first_messages[0]["content"]
    user_prompt = first_messages[1]["content"]
    assert "Your activity trait is described as:" in system_prompt
    assert "Your conformity trait is described as:" in system_prompt
    assert "Your diversity trait is described as:" in system_prompt
    assert "##movie##" in user_prompt
    assert "<- Lion King -> <- History ratings:4.15 -> <- genres:Animation ->" in user_prompt
    assert "Predict the integer rating." in user_prompt
    assert "Use this format: RATING: [integer from 1 to 5]" in user_prompt
    assert "Rate the movie using your available profile" in user_prompt
    assert "Toy Story" not in user_prompt


def test_agent4rec_regressor_requires_build_taste_before_score(tmp_path) -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["i1", "i2"],
            "item_title": ["Toy Story", "Heat"],
            "item_genres": ["Animation", "Crime"],
            "rating": [5, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
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
    scorer = Agent4RecRegressor(
        client=scoring_client,
        model="fake-model",
        target_description="Predict the integer rating.",
        valid_values=(1, 2, 3, 4, 5),
        profile_generator=profile_generator,
        candidate_description_columns=("item_title", "item_rating_mean", "item_genres"),
    ).fit(X_train, pd.Series([5, 2], name="target"))

    with pytest.raises(RuntimeError, match="Call scorer.build_taste"):
        scorer.score(X_test)

    assert scoring_client.completions.calls == []
    assert taste_client.completions.calls == []


def test_agent4rec_regressor_build_taste_then_scores_with_taste(tmp_path) -> None:
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
            "item_title": ["Lion King"],
            "item_rating_mean": [4.153],
            "item_genres": ["Animation"],
        },
        index=["a"],
    )
    scoring_client = FakeClient(["RATING: 5"])
    taste_client = FakeClient(
        [
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I rated animated movies highly.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies"
        ]
    )
    profile_generator = Agent4RecProfileGenerator(
        profile_components=("taste",),
        taste_client=taste_client,
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=tmp_path / "taste.jsonl",
    )
    scorer = Agent4RecRegressor(
        client=scoring_client,
        model="fake-model",
        target_description="Predict the integer rating.",
        valid_values=(1, 2, 3, 4, 5),
        profile_generator=profile_generator,
        candidate_description_columns=("item_title", "item_rating_mean", "item_genres"),
    ).fit(X_train, pd.Series([5, 4, 2], name="target"))

    scorer.build_taste(X_test)
    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 5.0}
    assert len(taste_client.completions.calls) == 1
    system_prompt = scoring_client.completions.calls[0]["messages"][0]["content"]
    user_prompt = scoring_client.completions.calls[0]["messages"][1]["content"]
    assert "Your movie tastes are: enjoy animated family movies." in system_prompt
    assert "Rate the movie according to your taste" in user_prompt
