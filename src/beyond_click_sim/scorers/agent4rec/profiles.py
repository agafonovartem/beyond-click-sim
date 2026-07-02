from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.agent4rec.prompts import (
    AGENT4REC_PLAYTIME_TASTE_SYSTEM_PROMPT,
    AGENT4REC_TASTE_PROMPT_VERSION,
    AGENT4REC_TASTE_SYSTEM_PROMPT,
    agent4rec_playtime_taste_user_prompt,
    agent4rec_taste_modify_user_prompt,
)
from beyond_click_sim.scorers.history.selection import UserHistory


AGENT4REC_PROFILE_COMPONENTS = ("taste", "traits")
_SUPPORTED_AGENT4REC_PROFILE_COMPONENTS = frozenset(AGENT4REC_PROFILE_COMPONENTS)
_SUPPORTED_TASTE_PROMPT_KINDS = frozenset({"rating", "playtime"})
AGENT4REC_DIVERSITY_TOP_MASS = 0.80
_TASTE_CACHE_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_TASTE_CACHE_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class Agent4RecUserProfile:
    """User profile used by the Agent4Rec-style scorer."""

    user_id: Any
    taste: str | None = None
    activity_group: int | None = None
    conformity_group: int | None = None
    diversity_group: int | None = None
    activity_description: str | None = None
    conformity_description: str | None = None
    diversity_description: str | None = None


@dataclass(frozen=True)
class Agent4RecTasteProfile:
    """Parsed Agent4Rec `prompt_modify` taste response."""

    taste: str
    reason: str
    high_rating: str
    low_rating: str
    raw_response: str


AGENT4REC_ACTIVITY_DESCRIPTIONS = {
    1: "An Incredibly Elusive Occasional Viewer, so seldom attracted by movie recommendations that it's almost a legendary event when you do watch a movie. Your movie-watching habits are extraordinarily infrequent. And you will exit the recommender system immediately even if you just feel little unsatisfied.",
    2: "An Occasional Viewer, seldom attracted by movie recommendations. Only curious about watching movies that strictly align the taste. The movie-watching habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.",
    3: "A Movie Enthusiast with an insatiable appetite for films, willing to watch nearly every movie recommended to you. Movies are a central part of your life, and movie recommendations are integral to your existence. You are tolerant of recommender system, which means you are not easy to exit recommender system even if you have some unsatisfied memory.",
}

AGENT4REC_CONFORMITY_DESCRIPTIONS = {
    1: "A Dedicated Follower who gives ratings heavily relies on movie historical ratings, rarely expressing independent opinions. Usually give ratings that are same as historical ratings. ",
    2: "A Balanced Evaluator who considers both historical ratings and personal preferences when giving ratings to movies. Sometimes give ratings that are different from historical rating.",
    3: "A Maverick Critic who completely ignores historical ratings and evaluates movies solely based on own taste. Usually give ratings that are a lot different from historical ratings.",
}

AGENT4REC_DIVERSITY_DESCRIPTIONS = {
    1: "An Exceedingly Discerning Selective Viewer who watches movies with a level of selectivity that borders on exclusivity. The movie choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
    2: "A Niche Explorer who occasionally explores different genres and mostly sticks to preferred movie types.",
    3: "A Cinematic Trailblazer, a relentless seeker of the unique and the obscure in the world of movies. The movie choices are so diverse and avant-garde that they defy categorization.",
}

AGENT4REC_GAME_ACTIVITY_DESCRIPTIONS = {
    1: "An Incredibly Elusive Occasional Player, so seldom attracted by game recommendations that it's almost a legendary event when you do play a game. Your game-playing habits are extraordinarily infrequent. And you will exit the recommender system immediately even if you just feel little unsatisfied.",
    2: "An Occasional Player, seldom attracted by game recommendations. Only curious about playing games that strictly align the taste. The game-playing habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.",
    3: "A Game Enthusiast with an insatiable appetite for games, willing to play nearly every game recommended to you. Games are a central part of your life, and game recommendations are integral to your existence. You are tolerant of recommender system, which means you are not easy to exit recommender system even if you have some unsatisfied memory.",
}

AGENT4REC_GAME_DIVERSITY_DESCRIPTIONS = {
    1: "An Exceedingly Discerning Selective Player who plays games with a level of selectivity that borders on exclusivity. The game choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
    2: "A Niche Explorer who occasionally explores different genres and mostly sticks to preferred game types.",
    3: "A Gaming Trailblazer, a relentless seeker of the unique and the obscure in the world of games. The game choices are so diverse and avant-garde that they defy categorization.",
}


class Agent4RecProfileGenerator:
    """Build Agent4Rec-style profiles from fitted train rows and histories."""

    def __init__(
        self,
        *,
        profile_components: Sequence[str] = ("traits",),
        user_column: str = "user_id",
        item_column: str = "item_id",
        rating_column: str = "rating",
        playtime_column: str = "playtime_forever",
        genre_column: str = "item_genres",
        tag_column: str | None = None,
        title_column: str = "item_title",
        include_conformity: bool = True,
        activity_descriptions: dict[int, str] | None = None,
        conformity_descriptions: dict[int, str] | None = None,
        diversity_descriptions: dict[int, str] | None = None,
        taste_client: Any | None = None,
        taste_client_name: str | None = None,
        taste_model: str | None = None,
        taste_cache_path: Path | str | None = None,
        taste_prompt_version: str = AGENT4REC_TASTE_PROMPT_VERSION,
        taste_prompt_kind: str = "rating",
        taste_temperature: float = 0.0,
        taste_max_tokens: int | None = None,
        taste_max_attempts: int = 5,
    ) -> None:
        profile_components_tuple = tuple(profile_components)
        if not profile_components_tuple:
            raise ValueError("profile_components must be non-empty")
        unknown_components = [
            component
            for component in profile_components_tuple
            if component not in _SUPPORTED_AGENT4REC_PROFILE_COMPONENTS
        ]
        if unknown_components:
            raise ValueError(
                "Unsupported profile components: "
                f"{unknown_components}. Supported components: "
                f"{sorted(_SUPPORTED_AGENT4REC_PROFILE_COMPONENTS)}"
            )
        if len(set(profile_components_tuple)) != len(profile_components_tuple):
            raise ValueError(
                f"profile_components contains duplicates: {profile_components_tuple}"
            )
        if taste_prompt_kind not in _SUPPORTED_TASTE_PROMPT_KINDS:
            raise ValueError(
                "Unsupported Agent4Rec taste_prompt_kind: "
                f"{taste_prompt_kind!r}. Supported: {sorted(_SUPPORTED_TASTE_PROMPT_KINDS)}"
            )

        self.profile_components = profile_components_tuple
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        self.playtime_column = playtime_column
        self.genre_column = genre_column
        self.tag_column = tag_column
        self.title_column = title_column
        self.include_conformity = include_conformity
        self.activity_descriptions = (
            AGENT4REC_ACTIVITY_DESCRIPTIONS
            if activity_descriptions is None
            else dict(activity_descriptions)
        )
        self.conformity_descriptions = (
            AGENT4REC_CONFORMITY_DESCRIPTIONS
            if conformity_descriptions is None
            else dict(conformity_descriptions)
        )
        self.diversity_descriptions = (
            AGENT4REC_DIVERSITY_DESCRIPTIONS
            if diversity_descriptions is None
            else dict(diversity_descriptions)
        )
        self.taste_client = taste_client
        self.taste_client_name = taste_client_name
        self.taste_model = taste_model
        self.taste_cache_path = (
            None if taste_cache_path is None else Path(taste_cache_path)
        )
        self.taste_prompt_version = taste_prompt_version
        self.taste_prompt_kind = taste_prompt_kind
        self.taste_temperature = taste_temperature
        self.taste_max_tokens = taste_max_tokens
        self.taste_max_attempts = taste_max_attempts
        self.trait_thresholds_: dict[str, dict[str, float]] | None = None
        self.taste_cache_stats_: dict[str, Any] | None = None

        if "taste" in self.profile_components:
            if self.taste_client is None:
                raise ValueError("taste_client is required for taste profiles")
            if not self.taste_model:
                raise ValueError("taste_model is required for taste profiles")
            if self.taste_cache_path is None:
                raise ValueError("taste_cache_path is required for taste profiles")
            if self.taste_max_attempts < 1:
                raise ValueError("taste_max_attempts must be positive")
            if self.taste_max_tokens is not None and self.taste_max_tokens < 1:
                raise ValueError("taste_max_tokens must be positive when set")

    def build_traits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        user_ids: Sequence[Any] | None = None,
    ) -> dict[Any, Agent4RecUserProfile]:
        """Build deterministic Agent4Rec social-trait profiles from train rows."""

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.user_column])

        if user_ids is None:
            requested_user_ids = list(dict.fromkeys(X[self.user_column].tolist()))
        else:
            requested_user_ids = list(dict.fromkeys(user_ids))
        profiles = {
            user_id: Agent4RecUserProfile(user_id=user_id)
            for user_id in requested_user_ids
        }
        traits = self._build_traits(X)
        missing_users = [
            user_id
            for user_id in requested_user_ids
            if user_id not in traits
        ]
        if missing_users:
            raise ValueError(
                "No train rows for requested Agent4Rec profile users: "
                f"{missing_users[:5]}"
            )
        profiles = {
            user_id: self._with_traits(profile, traits[user_id])
            for user_id, profile in profiles.items()
        }
        return profiles

    def build_taste(
        self,
        *,
        profiles: dict[Any, Agent4RecUserProfile],
        histories: dict[Any, UserHistory],
        user_ids: Sequence[Any],
    ) -> dict[Any, Agent4RecUserProfile]:
        """Fill taste profiles for selected users using cache and LLM misses."""

        if "taste" not in self.profile_components:
            return profiles
        if self.taste_client is None or not self.taste_model:
            raise RuntimeError("Taste profile generator is not configured")
        if self.taste_cache_path is None:
            raise RuntimeError("taste_cache_path is not configured")

        requested_user_ids = list(dict.fromkeys(user_ids))
        updated_profiles = dict(profiles)
        hits = 0
        misses = 0

        with self._taste_cache_lock(self.taste_cache_path):
            cache = self._load_taste_cache(self.taste_cache_path)
            for user_id in requested_user_ids:
                if user_id not in updated_profiles:
                    raise ValueError(
                        f"No fitted Agent4Rec profile for user: {user_id!r}"
                    )
                if user_id not in histories:
                    raise ValueError(f"No fitted taste history for user: {user_id!r}")

                history = histories[user_id]
                cache_key = self._taste_cache_key(
                    user_id=user_id,
                    history_item_ids=history.item_ids,
                )
                if cache_key in cache:
                    cached = cache[cache_key]
                    taste = str(cached.get("taste", ""))
                    if not taste.strip():
                        raise ValueError(
                            f"Cached Agent4Rec taste is empty for user: {user_id!r}"
                        )
                    hits += 1
                else:
                    parsed = self._generate_taste(user_id=user_id, history=history)
                    taste = parsed.taste
                    row = self._taste_cache_row(
                        user_id=user_id,
                        history=history,
                        cache_key=cache_key,
                        parsed=parsed,
                    )
                    self._append_taste_cache_row(self.taste_cache_path, row)
                    cache[cache_key] = row
                    misses += 1

                updated_profiles[user_id] = self._with_taste(
                    updated_profiles[user_id],
                    taste=taste,
                )

        self.taste_cache_stats_ = {
            "requested_users": len(requested_user_ids),
            "hits": hits,
            "misses": misses,
            "generated": misses,
        }
        return updated_profiles

    def manifest(self) -> dict[str, Any]:
        """Return profile-generation metadata for experiment manifests."""

        manifest: dict[str, Any] = {
            "class": type(self).__name__,
            "profile_components": list(self.profile_components),
            "user_column": self.user_column,
            "item_column": self.item_column,
            "rating_column": self.rating_column,
            "playtime_column": self.playtime_column,
            "genre_column": self.genre_column,
            "tag_column": self.tag_column,
            "title_column": self.title_column,
            "include_conformity": self.include_conformity,
            "diversity_top_mass": AGENT4REC_DIVERSITY_TOP_MASS,
            "trait_thresholds": self.trait_thresholds_,
        }
        if "taste" in self.profile_components:
            manifest["taste"] = {
                "client_name": self.taste_client_name,
                "model": self.taste_model,
                "temperature": self.taste_temperature,
                "max_tokens": self.taste_max_tokens,
                "max_attempts": self.taste_max_attempts,
                "prompt_version": self.taste_prompt_version,
                "prompt_kind": self.taste_prompt_kind,
                "cache_path": (
                    None
                    if self.taste_cache_path is None
                    else str(self.taste_cache_path)
                ),
                "cache_stats": self.taste_cache_stats_,
            }
        return manifest

    def _build_traits(self, X: pd.DataFrame) -> dict[Any, dict[str, int]]:
        required_columns = [self.user_column, self.item_column, self.genre_column]
        if self.include_conformity:
            required_columns.append(self.rating_column)
        self._require_columns(X, required_columns)

        activity_num = X.groupby(self.user_column, sort=False).size()
        activity_low = float(activity_num.quantile(0.60))
        activity_high = float(activity_num.quantile(0.90))

        diversity_num = self._agent4rec_diversity_count_by_user(X)
        diversity_low = float(diversity_num.quantile(0.33))
        diversity_high = float(diversity_num.quantile(0.66))

        self.trait_thresholds_ = {
            "activity": {"p60": activity_low, "p90": activity_high},
            "diversity": {"p33": diversity_low, "p66": diversity_high},
        }
        if self.include_conformity:
            item_mean_rating = X.groupby(self.item_column, sort=False)[
                self.rating_column
            ].mean()
            rows = X[[self.user_column, self.item_column, self.rating_column]].copy()
            rows["_item_mean_rating_"] = rows[self.item_column].map(item_mean_rating)
            rows["_squared_deviation_"] = (
                rows[self.rating_column] - rows["_item_mean_rating_"]
            ) ** 2
            conformity_deviation = rows.groupby(self.user_column, sort=False)[
                "_squared_deviation_"
            ].mean()
            conformity_low = float(conformity_deviation.quantile(0.25))
            conformity_high = float(conformity_deviation.quantile(0.80))
            self.trait_thresholds_["conformity"] = {
                "p25": conformity_low,
                "p80": conformity_high,
            }
        else:
            conformity_deviation = None
            conformity_low = None
            conformity_high = None

        traits: dict[Any, dict[str, int]] = {}
        for user_id in activity_num.index:
            traits[user_id] = {
                "activity": _three_tier_group(
                    float(activity_num.loc[user_id]),
                    low=activity_low,
                    high=activity_high,
                ),
                "diversity": _three_tier_group(
                    float(diversity_num.loc[user_id]),
                    low=diversity_low,
                    high=diversity_high,
                ),
            }
            if self.include_conformity:
                traits[user_id]["conformity"] = _three_tier_group(
                    float(conformity_deviation.loc[user_id]),
                    low=conformity_low,
                    high=conformity_high,
                )
        return traits

    def _agent4rec_diversity_count_by_user(self, X: pd.DataFrame) -> pd.Series:
        """Count dominant genres per user using Agent4Rec-style top-mass logic.

        Genres are counted across the fitted history, converted to per-user
        genre probabilities, sorted from most to least frequent, and retained
        while cumulative probability is at most ``AGENT4REC_DIVERSITY_TOP_MASS``.
        We clamp non-empty histories to at least one genre because the released
        Agent4Rec condition can otherwise assign zero diversity to single-genre
        users.
        """

        rows: list[dict[str, Any]] = []
        for row in X[[self.user_column, self.genre_column]].itertuples(index=False):
            user_id = getattr(row, self.user_column)
            for genre in _split_genres(getattr(row, self.genre_column)):
                rows.append({self.user_column: user_id, "genre": genre})
        if not rows:
            raise ValueError("Cannot compute diversity: no genres found")
        genre_rows = pd.DataFrame(rows)
        user_order = list(dict.fromkeys(X[self.user_column].tolist()))
        genre_counts = (
            genre_rows.groupby([self.user_column, "genre"], sort=False)
            .size()
            .rename("count")
            .reset_index()
        )

        diversity_counts: dict[Any, int] = {}
        for user_id, user_genres in genre_counts.groupby(self.user_column, sort=False):
            sorted_genres = user_genres.sort_values(
                by=["count", "genre"],
                ascending=[False, True],
                kind="mergesort",
            )
            ratios = sorted_genres["count"].astype(float) / float(
                sorted_genres["count"].sum()
            )
            top_mass_genre_count = int(
                (ratios.cumsum() <= AGENT4REC_DIVERSITY_TOP_MASS).sum()
            )
            diversity_counts[user_id] = max(1, top_mass_genre_count)

        return pd.Series(
            {user_id: diversity_counts.get(user_id, 0) for user_id in user_order}
        )

    def _with_traits(
        self,
        profile: Agent4RecUserProfile,
        traits: dict[str, int],
    ) -> Agent4RecUserProfile:
        activity_group = traits.get("activity")
        conformity_group = traits.get("conformity")
        diversity_group = traits.get("diversity")
        return Agent4RecUserProfile(
            user_id=profile.user_id,
            taste=profile.taste,
            activity_group=activity_group,
            conformity_group=conformity_group,
            diversity_group=diversity_group,
            activity_description=(
                None
                if activity_group is None
                else self.activity_descriptions[activity_group]
            ),
            conformity_description=(
                None
                if conformity_group is None
                else self.conformity_descriptions[conformity_group]
            ),
            diversity_description=(
                None
                if diversity_group is None
                else self.diversity_descriptions[diversity_group]
            ),
        )

    @staticmethod
    def _with_taste(
        profile: Agent4RecUserProfile,
        *,
        taste: str,
    ) -> Agent4RecUserProfile:
        return Agent4RecUserProfile(
            user_id=profile.user_id,
            taste=taste,
            activity_group=profile.activity_group,
            conformity_group=profile.conformity_group,
            diversity_group=profile.diversity_group,
            activity_description=profile.activity_description,
            conformity_description=profile.conformity_description,
            diversity_description=profile.diversity_description,
        )

    def _generate_taste(self, *, user_id: Any, history: UserHistory) -> Agent4RecTasteProfile:
        last_error: Exception | None = None
        for attempt in range(1, self.taste_max_attempts + 1):
            try:
                messages = self._taste_messages(history)
                request: dict[str, Any] = {
                    "model": self.taste_model,
                    "messages": messages,
                    "temperature": self.taste_temperature,
                }
                if self.taste_max_tokens is not None:
                    request["max_tokens"] = self.taste_max_tokens
                response = self.taste_client.chat.completions.create(**request)
                raw_response = _chat_completion_text(response)
                return parse_agent4rec_modify_taste_response(raw_response)
            except Exception as error:
                last_error = error
        raise RuntimeError(
            "Could not generate Agent4Rec taste profile "
            f"for user {user_id!r} after {self.taste_max_attempts} attempts"
        ) from last_error

    def _taste_messages(self, history: UserHistory) -> list[dict[str, str]]:
        if self.taste_prompt_kind == "rating":
            rating_movies = self._rating_movies(history)
            user_prompt = agent4rec_taste_modify_user_prompt(
                rating_movies=rating_movies,
            )
            system_prompt = AGENT4REC_TASTE_SYSTEM_PROMPT
        elif self.taste_prompt_kind == "playtime":
            playtime_games = self._playtime_games(history)
            user_prompt = agent4rec_playtime_taste_user_prompt(
                playtime_games=playtime_games,
            )
            system_prompt = AGENT4REC_PLAYTIME_TASTE_SYSTEM_PROMPT
        else:
            raise RuntimeError(f"Unsupported taste_prompt_kind: {self.taste_prompt_kind}")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _rating_movies(self, history: UserHistory) -> dict[int, list[str]]:
        self._require_columns(
            history.rows,
            [
                self.rating_column,
                self.title_column,
            ],
        )
        rating_movies: dict[int, list[str]] = {rating: [] for rating in range(1, 6)}
        for _, row in history.rows.iterrows():
            raw_rating = row[self.rating_column]
            raw_title = row[self.title_column]
            if pd.isna(raw_rating):
                raise ValueError(
                    f"Missing rating in Agent4Rec taste history for user: {history.user_id!r}"
                )
            if pd.isna(raw_title) or str(raw_title).strip() == "":
                continue
            rating = int(raw_rating)
            if rating not in rating_movies:
                raise ValueError(
                    "Agent4Rec taste generation expects ratings in 1..5. "
                    f"Got {raw_rating!r} for user {history.user_id!r}"
                )
            rating_movies[rating].append(str(raw_title).strip())
        return rating_movies

    def _playtime_games(self, history: UserHistory) -> dict[str, list[str]]:
        required_columns = [self.playtime_column, self.title_column]
        if self.genre_column:
            required_columns.append(self.genre_column)
        if self.tag_column:
            required_columns.append(self.tag_column)
        self._require_columns(history.rows, required_columns)

        playtime_games: dict[str, list[str]] = {
            "zero": [],
            "low": [],
            "medium": [],
            "high": [],
        }
        for _, row in history.rows.iterrows():
            raw_playtime = row[self.playtime_column]
            if pd.isna(raw_playtime):
                raise ValueError(
                    "Missing playtime in Agent4Rec taste history for user: "
                    f"{history.user_id!r}"
                )
            raw_title = row[self.title_column]
            if pd.isna(raw_title) or str(raw_title).strip() == "":
                continue

            playtime = float(raw_playtime)
            if playtime <= 0:
                bucket = "zero"
            elif playtime < 120:
                bucket = "low"
            elif playtime < 600:
                bucket = "medium"
            else:
                bucket = "high"
            playtime_games[bucket].append(self._history_item_description(row))
        return playtime_games

    def _history_item_description(self, row: pd.Series) -> str:
        parts = [str(row[self.title_column]).strip()]
        if self.genre_column and self.genre_column in row.index:
            genres = _split_genres(row[self.genre_column])
            if genres:
                parts.append(f"genres: {', '.join(genres[:8])}")
        if self.tag_column and self.tag_column in row.index:
            tags = _split_genres(row[self.tag_column])
            if tags:
                parts.append(f"tags: {', '.join(tags[:8])}")
        if len(parts) == 1:
            return parts[0]
        return f"{parts[0]} ({'; '.join(parts[1:])})"

    def _taste_cache_row(
        self,
        *,
        user_id: Any,
        history: UserHistory,
        cache_key: str,
        parsed: Agent4RecTasteProfile,
    ) -> dict[str, Any]:
        return {
            "cache_key": cache_key,
            "user_id": _json_safe(user_id),
            "history_item_ids": [_json_safe(item_id) for item_id in history.item_ids],
            **self._taste_cache_history_values(history),
            "history_titles": [
                _json_safe(value)
                for value in history.rows[self.title_column].tolist()
            ],
            "taste": parsed.taste,
            "reason": parsed.reason,
            "high_rating": parsed.high_rating,
            "low_rating": parsed.low_rating,
            "raw_response": parsed.raw_response,
            "model": self.taste_model,
            "prompt_version": self.taste_prompt_version,
            "prompt_kind": self.taste_prompt_kind,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }

    def _taste_cache_history_values(self, history: UserHistory) -> dict[str, Any]:
        if self.taste_prompt_kind == "rating":
            return {
                "history_ratings": [
                    _json_safe(value)
                    for value in history.rows[self.rating_column].tolist()
                ]
            }
        if self.taste_prompt_kind == "playtime":
            return {
                "history_playtime": [
                    _json_safe(value)
                    for value in history.rows[self.playtime_column].tolist()
                ]
            }
        raise RuntimeError(f"Unsupported taste_prompt_kind: {self.taste_prompt_kind}")

    @staticmethod
    def _taste_cache_key(
        *,
        user_id: Any,
        history_item_ids: Sequence[Any],
    ) -> str:
        payload = {
            "user_id": _json_safe(user_id),
            "history_item_ids": [_json_safe(item_id) for item_id in history_item_ids],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_taste_cache(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        rows: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"Malformed Agent4Rec taste cache row {line_number}: empty line"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed Agent4Rec taste cache row {line_number}"
                    ) from error
                cache_key = row.get("cache_key")
                if not cache_key:
                    raise ValueError(
                        f"Malformed Agent4Rec taste cache row {line_number}: missing cache_key"
                    )
                if cache_key in rows:
                    # Older parallel runs could leave duplicate rows in the shared
                    # JSONL cache. Keep the first complete taste profile so a stale
                    # cache artifact does not fail an otherwise reproducible run.
                    continue
                rows[str(cache_key)] = row
        return rows

    @staticmethod
    def _append_taste_cache_row(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    @contextmanager
    def _taste_cache_lock(path: Path) -> Iterator[None]:
        resolved_path = path.resolve()
        with _TASTE_CACHE_THREAD_LOCKS_GUARD:
            thread_lock = _TASTE_CACHE_THREAD_LOCKS.setdefault(
                resolved_path,
                threading.Lock(),
            )
        with thread_lock:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = resolved_path.with_name(f"{resolved_path.name}.lock")
            with lock_path.open("a", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def _three_tier_group(value: float, *, low: float, high: float) -> int:
    if value < low:
        return 1
    if value < high:
        return 2
    return 3


def _split_genres(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw_parts = value
    else:
        if pd.isna(value) or value == "":
            return []
        text = str(value).strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                raw_parts = parsed
                return [
                    str(part).strip()
                    for part in raw_parts
                    if str(part).strip()
                ]
        separator = "|" if "|" in text else ","
        raw_parts = text.split(separator)
    return [str(part).strip() for part in raw_parts if str(part).strip()]


def parse_agent4rec_modify_taste_response(text: str) -> Agent4RecTasteProfile:
    """Parse Agent4Rec `prompt_modify` persona text using the released regex style."""

    taste = "| ".join(re.findall(r"TASTE:(.+)", text))
    reason = "| ".join(re.findall(r"REASON:(.+)", text))
    high_rating = "| ".join(re.findall(r"HIGH (?:RATINGS|PLAYTIME):(.+)", text))
    low_rating = "| ".join(re.findall(r"LOW (?:RATINGS|PLAYTIME):(.+)", text))
    if not taste.strip():
        raise ValueError("Agent4Rec taste response does not contain TASTE")
    return Agent4RecTasteProfile(
        taste=taste,
        reason=reason,
        high_rating=high_rating,
        low_rating=low_rating,
        raw_response=text,
    )


def _chat_completion_text(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        raise ValueError("Chat completion response has no text content")
    return str(content)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
