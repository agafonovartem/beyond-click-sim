from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


AGENT4REC_PROFILE_COMPONENTS = ("taste", "traits")
_SUPPORTED_AGENT4REC_PROFILE_COMPONENTS = frozenset(AGENT4REC_PROFILE_COMPONENTS)
AGENT4REC_DIVERSITY_TOP_MASS = 0.80


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


class Agent4RecProfileGenerator:
    """Build Agent4Rec-style profiles from fitted train rows.

    V1 implements only the deterministic social-traits branch. Taste profile
    generation/cache will be added separately so scorer logic stays independent
    from profile-generation storage and model choices.
    """

    def __init__(
        self,
        *,
        profile_components: Sequence[str] = ("traits",),
        user_column: str = "user_id",
        item_column: str = "item_id",
        rating_column: str = "rating",
        genre_column: str = "item_genres",
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

        self.profile_components = profile_components_tuple
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        self.genre_column = genre_column
        self.trait_thresholds_: dict[str, dict[str, float]] | None = None

    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[Any, Agent4RecUserProfile]:
        """Build profiles using only the rows passed to fit."""

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.user_column])
        if "taste" in self.profile_components:
            raise NotImplementedError(
                "Agent4Rec taste profile generation/cache is not implemented yet"
            )

        user_ids = list(dict.fromkeys(X[self.user_column].tolist()))
        profiles = {
            user_id: Agent4RecUserProfile(user_id=user_id)
            for user_id in user_ids
        }
        if "traits" in self.profile_components:
            traits = self._build_traits(X)
            profiles = {
                user_id: self._with_traits(profile, traits[user_id])
                for user_id, profile in profiles.items()
            }
        return profiles

    def manifest(self) -> dict[str, Any]:
        """Return profile-generation metadata for experiment manifests."""

        return {
            "class": type(self).__name__,
            "profile_components": list(self.profile_components),
            "user_column": self.user_column,
            "item_column": self.item_column,
            "rating_column": self.rating_column,
            "genre_column": self.genre_column,
            "diversity_top_mass": AGENT4REC_DIVERSITY_TOP_MASS,
            "trait_thresholds": self.trait_thresholds_,
        }

    def _build_traits(self, X: pd.DataFrame) -> dict[Any, dict[str, int]]:
        self._require_columns(
            X,
            [
                self.user_column,
                self.item_column,
                self.rating_column,
                self.genre_column,
            ],
        )

        activity_num = X.groupby(self.user_column, sort=False).size()
        activity_low = float(activity_num.quantile(0.60))
        activity_high = float(activity_num.quantile(0.90))

        diversity_num = self._agent4rec_diversity_count_by_user(X)
        diversity_low = float(diversity_num.quantile(0.33))
        diversity_high = float(diversity_num.quantile(0.66))

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

        self.trait_thresholds_ = {
            "activity": {"p60": activity_low, "p90": activity_high},
            "diversity": {"p33": diversity_low, "p66": diversity_high},
            "conformity": {"p25": conformity_low, "p80": conformity_high},
        }

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
                "conformity": _three_tier_group(
                    float(conformity_deviation.loc[user_id]),
                    low=conformity_low,
                    high=conformity_high,
                ),
            }
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

    @staticmethod
    def _with_traits(
        profile: Agent4RecUserProfile,
        traits: dict[str, int],
    ) -> Agent4RecUserProfile:
        activity_group = traits["activity"]
        conformity_group = traits["conformity"]
        diversity_group = traits["diversity"]
        return Agent4RecUserProfile(
            user_id=profile.user_id,
            taste=profile.taste,
            activity_group=activity_group,
            conformity_group=conformity_group,
            diversity_group=diversity_group,
            activity_description=AGENT4REC_ACTIVITY_DESCRIPTIONS[activity_group],
            conformity_description=AGENT4REC_CONFORMITY_DESCRIPTIONS[conformity_group],
            diversity_description=AGENT4REC_DIVERSITY_DESCRIPTIONS[diversity_group],
        )

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
        text = str(value)
        separator = "|" if "|" in text else ","
        raw_parts = text.split(separator)
    return [str(part).strip() for part in raw_parts if str(part).strip()]
