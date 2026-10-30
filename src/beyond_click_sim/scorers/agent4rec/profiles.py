from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from tqdm import tqdm

from beyond_click_sim.scorers.agent4rec.prompts import (
    AGENT4REC_PLAYTIME_TASTE_SYSTEM_PROMPT,
    AGENT4REC_TASTE_PROMPT_VERSION,
    AGENT4REC_COAT_TASTE_SYSTEM_PROMPT,
    AGENT4REC_TASTE_SYSTEM_PROMPT,
    agent4rec_coat_taste_user_prompt,
    agent4rec_playtime_taste_user_prompt,
    agent4rec_taste_modify_user_prompt,
)
from beyond_click_sim.scorers.history.selection import UserHistory


AGENT4REC_PROFILE_COMPONENTS = ("taste", "traits")
_SUPPORTED_AGENT4REC_PROFILE_COMPONENTS = frozenset(AGENT4REC_PROFILE_COMPONENTS)
_SUPPORTED_TASTE_PROMPT_KINDS = frozenset({"rating", "playtime", "coat_rating"})
AGENT4REC_IMPLEMENTATION_VERSION = "agent4rec_offline_v2"
AGENT4REC_TASTE_CACHE_VERSION = "taste_patterns_v2"
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
    high_rating: str | None = None
    low_rating: str | None = None
    taste_reason: str | None = None
    pattern_kind: str = "rating"


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


AGENT4REC_COAT_CONFORMITY_DESCRIPTIONS = {
    1: "A Dedicated Follower who gives ratings that heavily rely on how other shoppers rated the same coats, rarely expressing independent opinions. Usually give ratings that are same as historical ratings. ",
    2: "A Balanced Evaluator who considers both how other shoppers rated a coat and personal preferences when giving ratings. Sometimes give ratings that are different from historical rating.",
    3: "A Maverick Critic who completely ignores how other shoppers rated a coat and evaluates coats solely based on own taste. Usually give ratings that are a lot different from historical ratings.",
}

# Coat substitutes gender-section/style/colour categories for movie genres.
# Union diversity therefore includes changes in gender section as well as style
# and colour. This domain substitution is recorded in the scorer documentation.
AGENT4REC_COAT_DIVERSITY_DESCRIPTIONS = {
    1: "An Exceedingly Discerning Selective Shopper who considers coats with a level of selectivity that borders on exclusivity. The coat choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
    2: "A Niche Explorer who occasionally looks beyond a favourite style or colour and mostly sticks to preferred coat types.",
    3: "A Wardrobe Trailblazer, a relentless seeker of the unusual in coats. The coat choices span styles and colours so widely that they defy categorization.",
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
        summary_column: str | None = None,
        include_activity: bool = True,
        include_conformity: bool = True,
        activity_counts: Mapping[Any, int] | None = None,
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
        taste_max_workers: int = 16,
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
        self.summary_column = summary_column
        self.include_activity = include_activity
        self.include_conformity = include_conformity
        self.activity_counts = (
            None if activity_counts is None else dict(activity_counts)
        )
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
        self.taste_max_workers = taste_max_workers
        self.trait_thresholds_: dict[str, dict[str, float]] | None = None
        self.taste_cache_stats_: dict[str, Any] | None = None
        self.trait_provenance_: dict[str, Any] = {}
        self.history_provenance_: dict[str, Any] = {}

        if "taste" in self.profile_components:
            if self.taste_client is None:
                raise ValueError("taste_client is required for taste profiles")
            if not self.taste_model:
                raise ValueError("taste_model is required for taste profiles")
            if self.taste_cache_path is None:
                raise ValueError("taste_cache_path is required for taste profiles")
            if self.taste_max_attempts < 1:
                raise ValueError("taste_max_attempts must be positive")
            if self.taste_max_workers < 1:
                raise ValueError("taste_max_workers must be positive")
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
            user_id for user_id in requested_user_ids if user_id not in traits
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
        error_count = 0
        cached_tastes: dict[Any, Agent4RecTasteProfile] = {}
        generated_tastes: dict[Any, Agent4RecTasteProfile] = {}
        misses_to_generate: list[tuple[Any, UserHistory, str]] = []

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
                    prompt_context=self._taste_cache_prompt_context(history),
                )
                if cache_key in cache:
                    cached = cache[cache_key]
                    taste = str(cached.get("taste", ""))
                    if not taste.strip():
                        raise ValueError(
                            f"Cached Agent4Rec taste is empty for user: {user_id!r}"
                        )
                    if cached.get("cache_version") != AGENT4REC_TASTE_CACHE_VERSION:
                        raise ValueError("Incompatible Agent4Rec taste cache version")
                    cached_tastes[user_id] = parse_agent4rec_modify_taste_response(
                        str(cached["raw_response"])
                    )
                    hits += 1
                else:
                    misses_to_generate.append((user_id, history, cache_key))
                    misses += 1

            if misses_to_generate:
                max_workers = min(self.taste_max_workers, len(misses_to_generate))
                error_count_lock = threading.Lock()

                def _count_error() -> None:
                    nonlocal error_count
                    with error_count_lock:
                        error_count += 1

                progress = tqdm(
                    total=len(misses_to_generate),
                    desc=f"agent4rec taste ({self.taste_model})",
                    unit="user",
                )
                try:
                    if max_workers == 1:
                        for user_id, history, cache_key in misses_to_generate:
                            parsed = self._generate_taste(
                                user_id=user_id,
                                history=history,
                                on_error=_count_error,
                            )
                            row = self._taste_cache_row(
                                user_id=user_id,
                                history=history,
                                cache_key=cache_key,
                                parsed=parsed,
                            )
                            self._append_taste_cache_row(self.taste_cache_path, row)
                            cache[cache_key] = row
                            generated_tastes[user_id] = parsed
                            progress.set_postfix(errors=error_count)
                            progress.update(1)
                    else:
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = {
                                executor.submit(
                                    self._generate_taste,
                                    user_id=user_id,
                                    history=history,
                                    on_error=_count_error,
                                ): (user_id, history, cache_key)
                                for user_id, history, cache_key in misses_to_generate
                            }
                            for future in as_completed(futures):
                                user_id, history, cache_key = futures[future]
                                parsed = future.result()
                                row = self._taste_cache_row(
                                    user_id=user_id,
                                    history=history,
                                    cache_key=cache_key,
                                    parsed=parsed,
                                )
                                self._append_taste_cache_row(self.taste_cache_path, row)
                                cache[cache_key] = row
                                generated_tastes[user_id] = parsed
                                progress.set_postfix(errors=error_count)
                                progress.update(1)
                finally:
                    progress.close()

            for user_id in requested_user_ids:
                taste = cached_tastes.get(user_id)
                if taste is None:
                    taste = generated_tastes[user_id]
                updated_profiles[user_id] = self._with_taste(
                    updated_profiles[user_id],
                    parsed=taste,
                    pattern_kind=self.taste_prompt_kind,
                )

        self.taste_cache_stats_ = {
            "requested_users": len(requested_user_ids),
            "hits": hits,
            "misses": misses,
            "generated": misses,
            "errors": error_count,
            "max_workers": min(self.taste_max_workers, max(1, misses)),
        }
        return updated_profiles

    def manifest(self) -> dict[str, Any]:
        """Return profile-generation metadata for experiment manifests."""

        manifest: dict[str, Any] = {
            "class": type(self).__name__,
            "implementation_version": AGENT4REC_IMPLEMENTATION_VERSION,
            "cache_version": AGENT4REC_TASTE_CACHE_VERSION,
            "profile_components": list(self.profile_components),
            "user_column": self.user_column,
            "item_column": self.item_column,
            "rating_column": self.rating_column,
            "playtime_column": self.playtime_column,
            "genre_column": self.genre_column,
            "tag_column": self.tag_column,
            "title_column": self.title_column,
            "summary_column": self.summary_column,
            "include_activity": self.include_activity,
            "include_conformity": self.include_conformity,
            "activity_source": (
                "fit_frame_row_counts"
                if self.activity_counts is None
                else "external_activity_counts"
            ),
            "diversity_definition": "distinct_genre_union_count",
            "tier_rule": "value <= low: 1; value <= high: 2; otherwise: 3",
            "trait_evidence": self.trait_provenance_,
            "taste_evidence": self.history_provenance_,
            "trait_thresholds": self.trait_thresholds_,
        }
        if "taste" in self.profile_components:
            manifest["taste"] = {
                "client_name": self.taste_client_name,
                "client_identity": client_identity(self.taste_client),
                "model": self.taste_model,
                "temperature": self.taste_temperature,
                "max_tokens": self.taste_max_tokens,
                "max_attempts": self.taste_max_attempts,
                "max_workers": self.taste_max_workers,
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
        self._require_columns(X, [self.user_column, self.item_column])
        users = X[self.user_column].drop_duplicates().tolist()
        traits: dict[Any, dict[str, int]] = {user: {} for user in users}
        omissions: dict[str, dict[str, str]] = {str(user): {} for user in users}
        self.trait_thresholds_ = {}
        values_by_trait = {}
        counts = X.groupby(self.user_column, sort=False).size()
        item_means = None

        def assign(name, values, quantiles, enabled=True):
            values = values.reindex(users).astype(float)
            finite = values.where(
                values.map(lambda x: pd.notna(x) and abs(x) != float("inf"))
            )
            valid = finite.dropna()
            values_by_trait[name] = {str(k): _json_safe(v) for k, v in finite.items()}
            reason = None
            if not enabled:
                reason = "disabled_by_task_or_configuration"
            elif valid.empty:
                reason = "missing_required_evidence"
            elif valid.nunique() == 1:
                reason = "constant_reference_population"
            else:
                low, high = (float(valid.quantile(q)) for q in quantiles)
                self.trait_thresholds_[name] = {
                    f"p{int(q * 100)}": v for q, v in zip(quantiles, (low, high))
                }
            for user in users:
                if reason or pd.isna(finite.loc[user]):
                    omissions[str(user)][name] = (
                        "disabled_by_task_or_configuration"
                        if not enabled
                        else "missing_required_evidence"
                        if pd.isna(finite.loc[user])
                        else reason
                    )
                else:
                    traits[user][name] = _three_tier_group(
                        finite.loc[user], low=low, high=high
                    )

        activity = self._activity_num(X) if self.include_activity else counts
        assign("activity", activity, (0.60, 0.90), self.include_activity)
        assign("diversity", self._agent4rec_diversity_count_by_user(X), (0.33, 0.66))
        conformity = pd.Series(float("nan"), index=users)
        if self.include_conformity and self.rating_column in X:
            ratings = pd.to_numeric(X[self.rating_column], errors="coerce")
            ratings = ratings.where(ratings.between(1, 5))
            rows = X[[self.user_column, self.item_column]].assign(_rating=ratings)
            item_means = rows.groupby(self.item_column, sort=False)._rating.mean()
            rows["_deviation"] = (ratings - rows[self.item_column].map(item_means)) ** 2
            conformity = rows.groupby(self.user_column, sort=False)._deviation.mean()
            # A partial rating history is not a complete conformity measurement.
            missing = rows._rating.isna().groupby(rows[self.user_column]).any()
            conformity.loc[missing[missing].index] = float("nan")
        assign("conformity", conformity, (0.25, 0.80), self.include_conformity)
        self.trait_provenance_ = {
            "source": "permitted_fit_frame_before_taste_selection",
            "population_users": len(users),
            "population_rows": len(X),
            "rows_by_user": {str(k): int(v) for k, v in counts.items()},
            "values_by_trait": values_by_trait,
            "omitted_traits": omissions,
            "genre_source_column": self.genre_column,
            "conformity_item_means": {}
            if item_means is None
            else {str(k): _json_safe(v) for k, v in item_means.items()},
            "population_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    X[
                        [
                            c
                            for c in (
                                self.user_column,
                                self.item_column,
                                self.rating_column,
                                self.genre_column,
                            )
                            if c in X
                        ]
                    ],
                    index=False,
                ).values.tobytes()
            ).hexdigest(),
        }
        return traits

    def _activity_num(self, X: pd.DataFrame) -> pd.Series:
        """Count permitted fit rows; explicit overrides must obey the same boundary."""
        if self.activity_counts is None:
            return X.groupby(self.user_column, sort=False).size()
        return pd.Series(
            {
                user: self.activity_counts.get(user, float("nan"))
                for user in X[self.user_column].drop_duplicates()
            }
        )

    def _agent4rec_diversity_count_by_user(self, X: pd.DataFrame) -> pd.Series:
        """Paper definition: cardinality of the union of observed canonical genres."""
        counts = {}
        for user, rows in X.groupby(self.user_column, sort=False):
            parsed = (
                [_split_genres(v) for v in rows[self.genre_column]]
                if self.genre_column in rows
                else []
            )
            counts[user] = (
                len(set().union(*map(set, parsed)))
                if parsed and all(parsed)
                else float("nan")
            )
        return pd.Series(counts, dtype=float)

    def select_taste_histories(self, X, *, user_column, item_column, max_history_items):
        """Select taste evidence only; trait statistics continue to use the full fit frame."""
        from beyond_click_sim.scorers.history.selection import select_history_by_user

        timestamp = next(
            (c for c in ("timestamp", "interaction_timestamp") if c in X), None
        )
        if timestamp:
            if X[timestamp].isna().any():
                raise ValueError("Missing timestamp in Agent4Rec taste evidence")
            X = X.sort_values(timestamp, kind="stable")
            rule = f"latest_by_{timestamp}_chronological_stable_ties"
        elif self.taste_prompt_kind == "coat_rating":
            # Coat is timestamp-free; both Coat runners use this deterministic subset.
            X = (
                X.assign(_agent4rec_item_order=X[item_column].astype(str))
                .sort_values("_agent4rec_item_order", kind="stable")
                .drop(columns="_agent4rec_item_order")
            )
            rule = "last_items_in_stable_string_item_id_order_no_timestamps"
        else:
            # Materialized temporal tasks omit timestamps but preserve split order.
            # Do not sort their rows by item id and destroy that chronology.
            rule = "last_rows_in_preserved_training_order"
        histories = select_history_by_user(
            X,
            user_column=user_column,
            item_column=item_column,
            max_history_items=max_history_items,
        )
        self.history_provenance_ = {
            "selection": rule,
            "max_items": max_history_items,
            "rows_by_user": {str(u): len(h.rows) for u, h in histories.items()},
            "source": "permitted_fit_frame_only",
        }
        return histories

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
    def _with_taste(profile, *, parsed: Agent4RecTasteProfile, pattern_kind: str):
        return replace(
            profile,
            taste=parsed.taste,
            high_rating=parsed.high_rating,
            low_rating=parsed.low_rating,
            taste_reason=parsed.reason,
            pattern_kind=pattern_kind,
        )

    def _generate_taste(
        self,
        *,
        user_id: Any,
        history: UserHistory,
        on_error: Callable[[], None] | None = None,
    ) -> Agent4RecTasteProfile:
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
                if on_error is not None:
                    on_error()
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
        elif self.taste_prompt_kind == "coat_rating":
            # Same 1-5 buckets as "rating"; the wording differs because the
            # movie suite asks a "movie taste analyst" about "movies".
            user_prompt = agent4rec_coat_taste_user_prompt(
                rating_coats=self._rating_movies(history),
            )
            system_prompt = AGENT4REC_COAT_TASTE_SYSTEM_PROMPT
        else:
            raise RuntimeError(
                f"Unsupported taste_prompt_kind: {self.taste_prompt_kind}"
            )
        if self.taste_prompt_kind in ("rating", "coat_rating"):
            entity = "coats" if self.taste_prompt_kind == "coat_rating" else "movies"
            lines = []
            for _, row in history.rows.iterrows():
                description = (
                    self._history_item_description(row)
                    if self.summary_column
                    else str(row[self.title_column])
                )
                lines.append(
                    f"user gives a rating of {int(row[self.rating_column])} for following {entity}: {description}"
                )
            instructions = user_prompt[user_prompt.index("My first request") :]
            user_prompt = (
                "Given a user's selected rating history in evidence order (oldest to newest when timestamps exist):\n"
                "Ratings 3-5 are liked; ratings 1-2 are disliked.\n"
                + "\n".join(lines)
                + "\n\n"
                + instructions
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _rating_movies(self, history: UserHistory) -> dict[int, list[str]]:
        required_columns = [self.rating_column, self.title_column]
        if self.summary_column:
            required_columns.append(self.summary_column)
        self._require_columns(history.rows, required_columns)
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
            if float(raw_rating) != rating or rating not in rating_movies:
                raise ValueError(
                    "Agent4Rec taste generation expects ratings in 1..5. "
                    f"Got {raw_rating!r} for user {history.user_id!r}"
                )
            if self.summary_column:
                rating_movies[rating].append(self._history_item_description(row))
            else:
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
        if self.summary_column and self.summary_column in row.index:
            summary = row[self.summary_column]
            if not pd.isna(summary) and str(summary).strip():
                parts.append(f"summary: {str(summary).strip()}")
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
            "cache_version": AGENT4REC_TASTE_CACHE_VERSION,
            "generation": _json_safe(self._taste_cache_prompt_context(history)),
            "user_id": _json_safe(user_id),
            "history_item_ids": [_json_safe(item_id) for item_id in history.item_ids],
            **self._taste_cache_history_values(history),
            "history_titles": [
                _json_safe(value) for value in history.rows[self.title_column].tolist()
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
        if self.taste_prompt_kind in ("rating", "coat_rating"):
            values = {
                "history_ratings": [
                    _json_safe(value)
                    for value in history.rows[self.rating_column].tolist()
                ]
            }
            values.update(self._taste_cache_summary_values(history))
            return values
        if self.taste_prompt_kind == "playtime":
            values = {
                "history_playtime": [
                    _json_safe(value)
                    for value in history.rows[self.playtime_column].tolist()
                ]
            }
            values.update(self._taste_cache_summary_values(history))
            return values
        raise RuntimeError(f"Unsupported taste_prompt_kind: {self.taste_prompt_kind}")

    def _taste_cache_summary_values(self, history: UserHistory) -> dict[str, Any]:
        if not self.summary_column:
            return {}
        self._require_columns(history.rows, [self.summary_column])
        return {
            "history_summaries": [
                _json_safe(value)
                for value in history.rows[self.summary_column].tolist()
            ]
        }

    def _taste_cache_prompt_context(self, history: UserHistory) -> dict[str, Any]:
        columns = [
            self.item_column,
            self.title_column,
            self.playtime_column
            if self.taste_prompt_kind == "playtime"
            else self.rating_column,
        ]
        if self.summary_column or self.taste_prompt_kind == "playtime":
            columns += [self.genre_column, self.tag_column, self.summary_column]
        columns = list(dict.fromkeys(c for c in columns if c and c in history.rows))
        return {
            "cache_version": AGENT4REC_TASTE_CACHE_VERSION,
            "ordered_evidence": history.rows[columns].to_dict("records"),
            "messages": self._taste_messages(history),
            "prompt_version": self.taste_prompt_version,
            "prompt_kind": self.taste_prompt_kind,
            "client_name": self.taste_client_name,
            "client_identity": client_identity(self.taste_client),
            "model": self.taste_model,
            "temperature": self.taste_temperature,
            "max_tokens": self.taste_max_tokens,
        }

    @staticmethod
    def _taste_cache_key(
        *,
        user_id: Any,
        history_item_ids: Sequence[Any],
        prompt_context: Mapping[str, Any] | None = None,
    ) -> str:
        payload = {
            "user_id": _json_safe(user_id),
            "history_item_ids": [_json_safe(item_id) for item_id in history_item_ids],
        }
        if prompt_context is not None:
            payload["prompt_context"] = _json_safe(prompt_context)
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
    if value <= low:
        return 1
    if value <= high:
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
                return [str(part).strip() for part in raw_parts if str(part).strip()]
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
    if not high_rating.strip() or not low_rating.strip():
        raise ValueError(
            "Agent4Rec taste response requires both HIGH and LOW pattern fields"
        )
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
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
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


def agent4rec_pattern_prompt(profile: Agent4RecUserProfile) -> str:
    """Share the generated rating/playtime patterns across every response interface."""
    if profile.high_rating is None and profile.low_rating is None:
        return ""
    label = "playtime" if profile.pattern_kind == "playtime" else "rating"
    return (
        f"\nYour high {label} patterns: {profile.high_rating or 'Insufficient evidence'}."
        f"\nYour low {label} patterns: {profile.low_rating or 'Insufficient evidence'}."
    )


def client_identity(client: Any) -> dict[str, str | None]:
    """Record endpoint identity without credentials, query parameters or API keys."""
    endpoint = getattr(client, "base_url", None)
    clean = None
    if endpoint is not None:
        parsed = urlsplit(str(endpoint))
        host = parsed.hostname or ""
        if parsed.port is not None:
            host += f":{parsed.port}"
        clean = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return {
        "class": f"{type(client).__module__}.{type(client).__qualname__}",
        "endpoint": clean,
    }
