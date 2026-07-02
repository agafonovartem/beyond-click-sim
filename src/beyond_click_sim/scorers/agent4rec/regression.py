from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
import re
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.agent4rec.profiles import (
    Agent4RecProfileGenerator,
    Agent4RecUserProfile,
)
from beyond_click_sim.scorers.agent4rec.prompts import (
    agent4rec_rating_user_prompt,
    agent4rec_system_prompt,
)
from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.history.selection import (
    UserHistory,
    select_history_by_user,
)


class Agent4RecRegressor(Scorer):
    """Agent4Rec-style rating regressor with a configurable profile module."""

    name = "agent4rec_regressor"

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        target_description: str,
        valid_values: Sequence[int],
        profile_generator: Agent4RecProfileGenerator | None = None,
        item_description_columns: tuple[str, ...] | None = None,
        candidate_description_columns: tuple[str, ...] | None = None,
        user_column: str = "user_id",
        max_history_items: int | None = 20,
        temperature: float = 0.0,
        max_tokens: int = 64,
        column_labels: dict[str, str] | None = None,
        extra_body: dict | None = None,
    ) -> None:
        if candidate_description_columns is None:
            candidate_description_columns = item_description_columns
        if not candidate_description_columns:
            raise ValueError("candidate_description_columns must be non-empty")
        if max_history_items is not None and max_history_items < 0:
            raise ValueError("max_history_items must be non-negative")
        if not target_description:
            raise ValueError("target_description must be non-empty")

        valid_values_tuple = tuple(valid_values)
        if not valid_values_tuple:
            raise ValueError("valid_values must be non-empty")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in valid_values_tuple
        ):
            raise ValueError("valid_values must contain integers")

        self.client = client
        self.model = model
        self.target_description = target_description
        self.valid_values = valid_values_tuple
        self.profile_generator = (
            Agent4RecProfileGenerator()
            if profile_generator is None
            else profile_generator
        )
        self.item_description_columns = item_description_columns
        self.candidate_description_columns = candidate_description_columns
        self.user_column = user_column
        self.max_history_items = max_history_items
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.column_labels = {} if column_labels is None else dict(column_labels)
        self.extra_body = extra_body
        self.profile_by_user_: dict[Any, Agent4RecUserProfile] | None = None
        self.history_by_user_: dict[Any, UserHistory] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        profile_user_ids: Sequence[Any] | None = None,
    ) -> "Agent4RecRegressor":
        """Build train-derived profile state and select train-history rows."""

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if "traits" in self.profile_generator.profile_components:
            self.profile_by_user_ = self.profile_generator.build_traits(
                X,
                y,
                user_ids=profile_user_ids,
            )
        else:
            self._require_columns(X, [self.user_column])
            user_ids = (
                list(dict.fromkeys(X[self.user_column].tolist()))
                if profile_user_ids is None
                else list(dict.fromkeys(profile_user_ids))
            )
            self.profile_by_user_ = {
                user_id: Agent4RecUserProfile(user_id=user_id)
                for user_id in user_ids
            }
        if "taste" in self.profile_generator.profile_components:
            history_rows = (
                X
                if profile_user_ids is None
                else X[X[self.user_column].isin(profile_user_ids)].copy()
            )
            self.history_by_user_ = select_history_by_user(
                history_rows,
                user_column=self.user_column,
                item_column=self.profile_generator.item_column,
                max_history_items=self.max_history_items,
            )
        else:
            self.history_by_user_ = {}
        return self

    def build_taste(self, X: pd.DataFrame) -> "Agent4RecRegressor":
        """Build/cache taste profiles for users present in an eval frame."""

        if self.profile_by_user_ is None or self.history_by_user_ is None:
            raise RuntimeError("Agent4RecRegressor is not fitted")
        self._require_columns(X, [self.user_column])
        user_ids = list(dict.fromkeys(X[self.user_column].tolist()))
        self.profile_by_user_ = self.profile_generator.build_taste(
            profiles=self.profile_by_user_,
            histories=self.history_by_user_,
            user_ids=user_ids,
        )
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Score each row with one LLM call and a strict 1-5 rating parser."""

        if self.profile_by_user_ is None:
            raise RuntimeError("Agent4RecRegressor is not fitted")
        self._require_columns(
            X,
            [
                self.user_column,
                *self.candidate_description_columns,
            ],
        )
        if "taste" in self.profile_generator.profile_components:
            self._require_taste_for_users(X[self.user_column].drop_duplicates().tolist())

        scores = pd.Series(index=X.index, dtype=float, name="score")
        for index, row in X.iterrows():
            messages = self._build_messages(row=row)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **({"extra_body": self.extra_body} if self.extra_body else {}),
            )
            scores.loc[index] = parse_agent4rec_rating_response(
                _chat_completion_text(response),
                valid_values=self.valid_values,
            )

        return scores

    def _build_messages(self, *, row: pd.Series) -> list[dict[str, str]]:
        if self.profile_by_user_ is None:
            raise RuntimeError("Agent4RecRegressor is not fitted")
        user_id = row[self.user_column]
        if user_id not in self.profile_by_user_:
            raise ValueError(f"No fitted Agent4Rec profile for user: {user_id!r}")

        profile = self.profile_by_user_[user_id]
        if "taste" in self.profile_generator.profile_components and not profile.taste:
            raise RuntimeError(
                "Agent4RecRegressor requires taste profiles for this method. "
                "Call scorer.build_taste(X_eval) before score()."
            )
        formatted_taste = _format_agent4rec_taste(profile.taste) or None
        system_prompt = self._format_system_prompt(profile, taste=formatted_taste)
        candidate = self._format_item_description(
            row=row,
            columns=self.candidate_description_columns,
        )
        user_prompt = agent4rec_rating_user_prompt(
            candidate=candidate,
            taste=formatted_taste,
            target_description=self.target_description,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _format_system_prompt(
        profile: Agent4RecUserProfile,
        *,
        taste: str | None = None,
    ) -> str:
        if taste is None:
            taste = _format_agent4rec_taste(profile.taste) or None
        return agent4rec_system_prompt(
            taste=taste,
            activity=profile.activity_description,
            conformity=profile.conformity_description,
            diversity=profile.diversity_description,
        )

    def _format_item_description(
        self,
        *,
        row: pd.Series,
        columns: tuple[str, ...],
    ) -> str:
        parts: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value) or value == "":
                continue
            column_label = self.column_labels.get(column, column)
            formatted_value = _format_prompt_value(value)
            if not parts:
                parts.append(f"<- {formatted_value} ->")
            else:
                parts.append(f"<- {column_label}:{formatted_value} ->")
        return " ".join(parts) if parts else "<- no item description ->"

    def _require_taste_for_users(self, user_ids: Sequence[Any]) -> None:
        if self.profile_by_user_ is None:
            raise RuntimeError("Agent4RecRegressor is not fitted")
        missing_users = [
            user_id
            for user_id in user_ids
            if user_id not in self.profile_by_user_
            or not self.profile_by_user_[user_id].taste
        ]
        if missing_users:
            raise RuntimeError(
                "Agent4RecRegressor requires taste profiles for this method. "
                "Call scorer.build_taste(X_eval) before score(). "
                f"Missing users: {missing_users[:5]}"
            )

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def parse_agent4rec_rating_response(
    text: str,
    *,
    valid_values: Sequence[int],
) -> float:
    """Parse an Agent4Rec `RATING: k` response and validate allowed values."""

    raw_text = text.strip()
    if re.fullmatch(r"[+-]?\d+", raw_text):
        raw_value = raw_text
    else:
        matches = re.findall(
            r"(?:^|\n)\s*RATING:\s*([+-]?\d+)\b",
            text,
            flags=re.IGNORECASE,
        )
        if not matches:
            raise ValueError(f"Agent4Rec rating response has no RATING value: {text!r}")
        if len(matches) > 1:
            raise ValueError(
                f"Agent4Rec rating response has multiple RATING values: {text!r}"
            )
        raw_value = matches[0]

    value = int(raw_value)
    if value not in set(valid_values):
        raise ValueError(
            f"Agent4Rec rating response {value!r} is outside valid values: "
            f"{list(valid_values)}"
        )
    return float(value)


def _chat_completion_text(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        raise ValueError("Chat completion response has no text content")
    return str(content)


def _format_agent4rec_taste(taste: str | None) -> str:
    if not taste:
        return ""
    taste_parts = [part.strip() for part in taste.split("| ") if part.strip()]
    return "; ".join(taste_parts).replace("I ", "")


def _format_prompt_value(value: Any) -> str:
    """Return compact text for scalar values shown in LLM prompts."""

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.2f}"
    return str(value)
