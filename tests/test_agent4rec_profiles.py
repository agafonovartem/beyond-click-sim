from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers.agent4rec import (
    Agent4RecProfileGenerator,
    Agent4RecUserProfile,
    parse_agent4rec_modify_taste_response,
)
from beyond_click_sim.scorers.history.selection import UserHistory


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


def test_diversity_counts_agent4rec_top_genre_mass_per_user() -> None:
    X = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3"],
            "item_id": ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8"],
            "rating": [5, 4, 3, 2, 5, 4, 5, 4],
            "item_genres": [
                "Action|Drama",
                "Action",
                "Action|Comedy",
                "Horror",
                "Horror",
                "Thriller",
                "Sci-Fi",
                "Sci-Fi",
            ],
        }
    )

    diversity_num = Agent4RecProfileGenerator()._agent4rec_diversity_count_by_user(X)

    assert diversity_num.to_dict() == {"u1": 2, "u2": 1, "u3": 1}


def test_parse_agent4rec_modify_taste_response_matches_original_join_style() -> None:
    parsed = parse_agent4rec_modify_taste_response(
        "TASTE: I enjoy animated films.\n"
        "REASON: I rated animated movies highly.\n\n"
        "TASTE: I like family comedies.\n"
        "REASON: They appear in high ratings.\n"
        "HIGH RATINGS: animated family movies\n"
        "LOW RATINGS: crime movies"
    )

    assert parsed.taste == " I enjoy animated films.|  I like family comedies."
    assert parsed.reason == " I rated animated movies highly.|  They appear in high ratings."
    assert parsed.high_rating == " animated family movies"
    assert parsed.low_rating == " crime movies"


def test_parse_agent4rec_modify_taste_response_requires_taste() -> None:
    with pytest.raises(ValueError, match="does not contain TASTE"):
        parse_agent4rec_modify_taste_response("REASON: nothing useful")


def test_build_taste_generates_and_reuses_jsonl_cache(tmp_path) -> None:
    cache_path = tmp_path / "taste.jsonl"
    taste_client = FakeClient(
        [
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I gave high ratings to animated movies.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies"
        ]
    )
    generator = Agent4RecProfileGenerator(
        profile_components=("traits", "taste"),
        taste_client=taste_client,
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=cache_path,
    )
    history_rows = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["i1", "i2"],
            "item_title": ["Toy Story", "Heat"],
            "rating": [5, 2],
        }
    )
    profiles = {"u1": Agent4RecUserProfile(user_id="u1")}
    histories = {
        "u1": UserHistory(
            user_id="u1",
            rows=history_rows,
            item_ids=("i1", "i2"),
        )
    }

    updated = generator.build_taste(
        profiles=profiles,
        histories=histories,
        user_ids=["u1"],
    )
    cached = generator.build_taste(
        profiles=profiles,
        histories=histories,
        user_ids=["u1"],
    )

    assert updated["u1"].taste == " I enjoy animated family movies."
    assert cached["u1"].taste == " I enjoy animated family movies."
    assert len(taste_client.completions.calls) == 1
    rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["history_item_ids"] == ["i1", "i2"]
    assert rows[0]["history_ratings"] == [5, 2]
    assert rows[0]["history_titles"] == ["Toy Story", "Heat"]


def test_build_taste_rejects_duplicate_cache_keys(tmp_path) -> None:
    cache_path = tmp_path / "taste.jsonl"
    row = {"cache_key": "duplicate", "taste": "taste"}
    cache_path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    generator = Agent4RecProfileGenerator(
        profile_components=("traits", "taste"),
        taste_client=FakeClient([]),
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=cache_path,
    )

    with pytest.raises(ValueError, match="Duplicate Agent4Rec taste cache key"):
        generator.build_taste(
            profiles={},
            histories={},
            user_ids=[],
        )
