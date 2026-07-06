from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers.agent4rec import (
    Agent4RecProfileGenerator,
    Agent4RecUserProfile,
    parse_agent4rec_modify_taste_response,
)
from beyond_click_sim.scorers.agent4rec.profiles import (
    AGENT4REC_GAME_ACTIVITY_DESCRIPTIONS,
    AGENT4REC_GAME_DIVERSITY_DESCRIPTIONS,
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


class SlowFakeChatCompletions(FakeChatCompletions):
    def create(self, **kwargs: object):
        time.sleep(0.05)
        return super().create(**kwargs)


class SlowFakeClient(FakeClient):
    def __init__(self, responses: list[str]) -> None:
        completions = SlowFakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


class TrackingFakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def create(self, **kwargs: object):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            with self.lock:
                self.calls.append(kwargs)
                response = self.responses.pop(0)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=response),
                    )
                ]
            )
        finally:
            with self.lock:
                self.active -= 1


class TrackingFakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = TrackingFakeChatCompletions(responses)
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


def test_build_traits_can_skip_conformity_for_steam_like_histories() -> None:
    X = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["g1", "g2", "g3", "g4"],
            "item_genres_json": [
                '["Action", "Indie"]',
                '["Action"]',
                '["Strategy"]',
                '["Simulation", "Strategy"]',
            ],
        }
    )
    generator = Agent4RecProfileGenerator(
        genre_column="item_genres_json",
        include_conformity=False,
        activity_descriptions=AGENT4REC_GAME_ACTIVITY_DESCRIPTIONS,
        diversity_descriptions=AGENT4REC_GAME_DIVERSITY_DESCRIPTIONS,
    )

    profiles = generator.build_traits(X, pd.Series([1, 1, 1, 1]))

    assert set(profiles) == {"u1", "u2"}
    assert profiles["u1"].activity_description is not None
    assert profiles["u1"].diversity_description is not None
    assert profiles["u1"].conformity_group is None
    assert profiles["u1"].conformity_description is None
    assert generator.trait_thresholds_ is not None
    assert "conformity" not in generator.trait_thresholds_


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


def test_parse_agent4rec_modify_taste_response_accepts_playtime_fields() -> None:
    parsed = parse_agent4rec_modify_taste_response(
        "TASTE: I enjoy puzzle platformers.\n"
        "REASON: I spent a long time with puzzle games.\n"
        "HIGH PLAYTIME: puzzle and co-op games\n"
        "LOW PLAYTIME: competitive shooters"
    )

    assert parsed.taste == " I enjoy puzzle platformers."
    assert parsed.high_rating == " puzzle and co-op games"
    assert parsed.low_rating == " competitive shooters"


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


def test_build_taste_can_include_item_summaries_in_history_prompt(tmp_path) -> None:
    cache_path = tmp_path / "taste-summary.jsonl"
    taste_client = FakeClient(
        [
            "TASTE: I enjoy adventurous animated movies.\n"
            "REASON: The summaries show playful adventures.\n"
            "HIGH RATINGS: adventurous animated movies\n"
            "LOW RATINGS: crime movies"
        ]
    )
    generator = Agent4RecProfileGenerator(
        profile_components=("taste",),
        taste_client=taste_client,
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=cache_path,
        summary_column="item_summary",
    )
    history_rows = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["1", "2"],
            "item_title": ["Toy Story", "Heat"],
            "item_summary": [
                "Toys plan a rescue.",
                "A detective hunts criminals.",
            ],
            "rating": [5, 2],
        }
    )
    profiles = {"u1": Agent4RecUserProfile(user_id="u1")}
    histories = {
        "u1": UserHistory(
            user_id="u1",
            rows=history_rows,
            item_ids=("1", "2"),
        )
    }

    updated = generator.build_taste(
        profiles=profiles,
        histories=histories,
        user_ids=["u1"],
    )

    assert updated["u1"].taste == " I enjoy adventurous animated movies."
    prompt = taste_client.completions.calls[0]["messages"][1]["content"]
    assert "Toy Story (summary: Toys plan a rescue.)" in prompt
    assert "Heat (summary: A detective hunts criminals.)" in prompt
    rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["history_summaries"] == [
        "Toys plan a rescue.",
        "A detective hunts criminals.",
    ]
    assert generator.manifest()["summary_column"] == "item_summary"

    old_key = Agent4RecProfileGenerator._taste_cache_key(
        user_id="u1",
        history_item_ids=("1", "2"),
    )
    assert rows[0]["cache_key"] != old_key


def test_build_taste_serializes_shared_jsonl_cache_writes(tmp_path) -> None:
    cache_path = tmp_path / "taste.jsonl"
    taste_client = SlowFakeClient(
        [
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I gave high ratings to animated movies.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies"
        ]
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

    def build() -> dict[str, Agent4RecUserProfile]:
        generator = Agent4RecProfileGenerator(
            profile_components=("taste",),
            taste_client=taste_client,
            taste_client_name="openai",
            taste_model="gpt-4o-mini",
            taste_cache_path=cache_path,
        )
        return generator.build_taste(
            profiles=profiles,
            histories=histories,
            user_ids=["u1"],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(build) for _ in range(2)]
        results = [future.result() for future in futures]

    assert [result["u1"].taste for result in results] == [
        " I enjoy animated family movies.",
        " I enjoy animated family movies.",
    ]
    assert len(taste_client.completions.calls) == 1
    rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1


def test_build_taste_can_generate_cache_misses_concurrently(tmp_path) -> None:
    cache_path = tmp_path / "taste.jsonl"
    taste_client = TrackingFakeClient(
        [
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I gave high ratings to animated movies.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies",
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I gave high ratings to animated movies.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies",
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I gave high ratings to animated movies.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies",
        ]
    )
    generator = Agent4RecProfileGenerator(
        profile_components=("taste",),
        taste_client=taste_client,
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=cache_path,
        taste_max_workers=3,
    )
    profiles = {
        "u1": Agent4RecUserProfile(user_id="u1"),
        "u2": Agent4RecUserProfile(user_id="u2"),
        "u3": Agent4RecUserProfile(user_id="u3"),
    }
    histories = {
        user_id: UserHistory(
            user_id=user_id,
            rows=pd.DataFrame(
                {
                    "user_id": [user_id, user_id],
                    "item_id": [f"{user_id}-i1", f"{user_id}-i2"],
                    "item_title": ["Toy Story", "Heat"],
                    "rating": [5, 2],
                }
            ),
            item_ids=(f"{user_id}-i1", f"{user_id}-i2"),
        )
        for user_id in profiles
    }

    updated = generator.build_taste(
        profiles=profiles,
        histories=histories,
        user_ids=["u1", "u2", "u3"],
    )

    assert set(updated) == {"u1", "u2", "u3"}
    assert all(
        profile.taste == " I enjoy animated family movies."
        for profile in updated.values()
    )
    assert len(taste_client.completions.calls) == 3
    assert taste_client.completions.max_active > 1
    assert generator.taste_cache_stats_ == {
        "requested_users": 3,
        "hits": 0,
        "misses": 3,
        "generated": 3,
        "max_workers": 3,
    }
    rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    assert {row["user_id"] for row in rows} == {"u1", "u2", "u3"}


def test_build_taste_uses_playtime_prompt_for_steam_like_histories(tmp_path) -> None:
    cache_path = tmp_path / "taste.jsonl"
    taste_client = FakeClient(
        [
            "TASTE: I enjoy puzzle and co-op games.\n"
            "REASON: I spent a long time with puzzle games.\n"
            "HIGH PLAYTIME: puzzle and co-op games\n"
            "LOW PLAYTIME: competitive shooters"
        ]
    )
    generator = Agent4RecProfileGenerator(
        profile_components=("taste",),
        genre_column="item_genres_json",
        tag_column="item_tags_json",
        title_column="item_title",
        playtime_column="playtime_forever",
        taste_prompt_kind="playtime",
        taste_client=taste_client,
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=cache_path,
        taste_prompt_version="agent4rec_playtime_v1",
    )
    history_rows = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["g1", "g2", "g3"],
            "item_title": ["Portal", "Dota 2", "Stardew Valley"],
            "item_genres_json": ['["Action"]', '["Action"]', '["RPG"]'],
            "item_tags_json": ['["Puzzle"]', '["MOBA"]', '["Farming Sim"]'],
            "playtime_forever": [600, 0, 180],
        }
    )
    profiles = {"u1": Agent4RecUserProfile(user_id="u1")}
    histories = {
        "u1": UserHistory(
            user_id="u1",
            rows=history_rows,
            item_ids=("g1", "g2", "g3"),
        )
    }

    updated = generator.build_taste(
        profiles=profiles,
        histories=histories,
        user_ids=["u1"],
    )

    assert updated["u1"].taste == " I enjoy puzzle and co-op games."
    taste_prompt = taste_client.completions.calls[0]["messages"][1]["content"]
    assert "game playtime history" in taste_prompt
    assert "high playtime" in taste_prompt
    assert "Portal (genres: Action; tags: Puzzle)" in taste_prompt
    rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["history_playtime"] == [600, 0, 180]
    assert rows[0]["prompt_kind"] == "playtime"


def test_load_taste_cache_keeps_first_duplicate_cache_key(tmp_path) -> None:
    cache_path = tmp_path / "taste.jsonl"
    row = {"cache_key": "duplicate", "taste": "first"}
    duplicate_row = {"cache_key": "duplicate", "taste": "second"}
    cache_path.write_text(
        json.dumps(row) + "\n" + json.dumps(duplicate_row) + "\n",
        encoding="utf-8",
    )

    cache = Agent4RecProfileGenerator._load_taste_cache(cache_path)

    assert cache == {"duplicate": row}
