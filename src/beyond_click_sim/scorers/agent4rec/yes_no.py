from __future__ import annotations

import json
from collections.abc import Sequence
from numbers import Integral, Real
import re
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.agent4rec.prompts import (
    agent4rec_itemwise_user_prompt,
    agent4rec_preference_user_prompt,
    agent4rec_system_prompt,
    agent4rec_user_prompt,
)
from beyond_click_sim.scorers.agent4rec.profiles import (
    Agent4RecProfileGenerator,
    Agent4RecUserProfile,
)
from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.history.selection import (
    UserHistory,
    select_history_by_user,
)
from beyond_click_sim.scorers.history.yes_no import parse_single_yes_no_response


class Agent4RecYesNoScorer(Scorer):
    """Agent4Rec-style yes/no scorer with a configurable profile module."""

    name = "agent4rec_yes_no"

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        profile_generator: Agent4RecProfileGenerator | None = None,
        item_description_columns: tuple[str, ...] | None = None,
        candidate_description_columns: tuple[str, ...] | None = None,
        user_column: str = "user_id",
        candidate_group_column: str = "candidate_group",
        max_history_items: int | None = 20,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        column_labels: dict[str, str] | None = None,
        json_list_columns: tuple[str, ...] = (),
        extra_body: dict | None = None,
        domain_name: str = "movie",
        taste_label: str = "movie tastes",
        entity_field: str = "MOVIE",
        entity_name: str = "movie",
        entity_plural: str = "movies",
        prompt_style: str = "batch",
    ) -> None:
        if candidate_description_columns is None:
            candidate_description_columns = item_description_columns
        if not candidate_description_columns:
            raise ValueError("candidate_description_columns must be non-empty")
        if max_history_items is not None and max_history_items < 0:
            raise ValueError("max_history_items must be non-negative")
        if prompt_style not in ("batch", "itemwise"):
            raise ValueError(
                f"prompt_style must be 'batch' or 'itemwise', got {prompt_style!r}"
            )

        self.client = client
        self.model = model
        self.profile_generator = (
            Agent4RecProfileGenerator()
            if profile_generator is None
            else profile_generator
        )
        self.item_description_columns = item_description_columns
        self.candidate_description_columns = candidate_description_columns
        self.user_column = user_column
        self.candidate_group_column = candidate_group_column
        self.max_history_items = max_history_items
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.column_labels = {} if column_labels is None else dict(column_labels)
        self.json_list_columns = tuple(json_list_columns)
        self.extra_body = extra_body
        self.domain_name = domain_name
        self.taste_label = taste_label
        self.entity_field = entity_field
        self.entity_name = entity_name
        self.entity_plural = entity_plural
        self.prompt_style = prompt_style
        self.profile_by_user_: dict[Any, Agent4RecUserProfile] | None = None
        self.history_by_user_: dict[Any, UserHistory] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        profile_user_ids: Sequence[Any] | None = None,
    ) -> "Agent4RecYesNoScorer":
        """Build train-derived profile state and select train-history rows.

        Taste generation is intentionally a separate explicit stage: call
        `build_taste(X_eval)` after fitting and before `score(X_eval)` for
        methods whose profile includes `taste`. If traits are disabled, `fit`
        still creates empty per-user profile shells so the later taste stage has
        a profile object to fill.
        """

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

    def build_taste(self, X: pd.DataFrame) -> "Agent4RecYesNoScorer":
        """Build/cache taste profiles for users present in an eval frame."""

        if self.profile_by_user_ is None or self.history_by_user_ is None:
            raise RuntimeError("Agent4RecYesNoScorer is not fitted")
        self._require_columns(X, [self.user_column])
        user_ids = list(dict.fromkeys(X[self.user_column].tolist()))
        self.profile_by_user_ = self.profile_generator.build_taste(
            profiles=self.profile_by_user_,
            histories=self.history_by_user_,
            user_ids=user_ids,
        )
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Score explicit candidate groups.

        In `prompt_style="batch"`, all candidates in a group share one LLM call.
        In `prompt_style="itemwise"` (default), each group must contain exactly
        one candidate row and gets its own LLM call with a bare yes/no answer.
        """

        if self.profile_by_user_ is None:
            raise RuntimeError("Agent4RecYesNoScorer is not fitted")
        self._require_columns(
            X,
            [
                self.user_column,
                self.candidate_group_column,
                *self.candidate_description_columns,
            ],
        )
        if "taste" in self.profile_generator.profile_components:
            self._require_taste_for_users(X[self.user_column].drop_duplicates().tolist())

        scores = pd.Series(index=X.index, dtype=float, name="score")
        for _, group in X.groupby(self.candidate_group_column, sort=False):
            user_ids = group[self.user_column].drop_duplicates()
            if len(user_ids) != 1:
                raise ValueError("Each candidate group must contain exactly one user")

            if self.prompt_style == "itemwise":
                if len(group) != 1:
                    raise ValueError(
                        "prompt_style='itemwise' requires groups of exactly one row; "
                        f"got {len(group)} rows in group"
                    )
                candidate_row = next(group.itertuples(index=False))
                messages = self._build_itemwise_messages(
                    user_id=user_ids.iloc[0],
                    candidate=candidate_row,
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **({"extra_body": self.extra_body} if self.extra_body else {}),
                )
                text = _chat_completion_text(response)
                try:
                    scores.loc[group.index[0]] = parse_single_yes_no_response(text)
                except ValueError as error:
                    raise ValueError(f"{error} | raw_response={text!r}") from error
                continue

            candidate_labels = _candidate_labels(len(group))
            messages = self._build_messages(
                user_id=user_ids.iloc[0],
                candidates=group,
                candidate_labels=candidate_labels,
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **({"extra_body": self.extra_body} if self.extra_body else {}),
            )
            text = _chat_completion_text(response)
            try:
                parsed = self._parse_response(text, labels=candidate_labels)
            except ValueError as error:
                raise ValueError(f"{error} | raw_response={text!r}") from error
            scores.loc[group.index] = [parsed[label] for label in candidate_labels]

        return scores

    def _build_messages(
        self,
        *,
        user_id: Any,
        candidates: pd.DataFrame,
        candidate_labels: Sequence[str],
    ) -> list[dict[str, str]]:
        if self.profile_by_user_ is None:
            raise RuntimeError("Agent4RecYesNoScorer is not fitted")
        if user_id not in self.profile_by_user_:
            raise ValueError(f"No fitted Agent4Rec profile for user: {user_id!r}")

        profile = self.profile_by_user_[user_id]
        if "taste" in self.profile_generator.profile_components and not profile.taste:
            raise RuntimeError(
                "Agent4RecYesNoScorer requires taste profiles for this method. "
                "Call scorer.build_taste(X_eval) before score()."
            )
        formatted_taste = _format_agent4rec_taste(profile.taste) or None
        system_prompt = self._format_system_prompt(profile, taste=formatted_taste)
        if len(candidate_labels) != len(candidates):
            raise ValueError("candidate_labels length must match candidates length")

        candidate_lines = [
            f"{label}. "
            + self._format_item_description(
                row=row,
                columns=self.candidate_description_columns,
            )
            for label, row in zip(
                candidate_labels,
                candidates.itertuples(index=False),
                strict=True,
            )
        ]
        user_prompt = self._build_user_prompt(
            candidates="\n".join(candidate_lines),
            taste=formatted_taste,
            candidate_labels=candidate_labels,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_itemwise_messages(
        self,
        *,
        user_id: Any,
        candidate: Any,
    ) -> list[dict[str, str]]:
        if self.profile_by_user_ is None:
            raise RuntimeError("Agent4RecYesNoScorer is not fitted")
        if user_id not in self.profile_by_user_:
            raise ValueError(f"No fitted Agent4Rec profile for user: {user_id!r}")

        profile = self.profile_by_user_[user_id]
        if "taste" in self.profile_generator.profile_components and not profile.taste:
            raise RuntimeError(
                "Agent4RecYesNoScorer requires taste profiles for this method. "
                "Call scorer.build_taste(X_eval) before score()."
            )
        formatted_taste = _format_agent4rec_taste(profile.taste) or None
        system_prompt = self._format_system_prompt(profile, taste=formatted_taste)
        candidate_description = self._format_item_description(
            row=candidate,
            columns=self.candidate_description_columns,
        )
        user_prompt = self._build_itemwise_user_prompt(
            candidate=candidate_description,
            taste=formatted_taste,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_itemwise_user_prompt(
        self,
        *,
        candidate: str,
        taste: str | None,
    ) -> str:
        return agent4rec_itemwise_user_prompt(
            candidate=candidate,
            taste=taste,
            entity_name=self.entity_name,
            entity_plural=self.entity_plural,
        )

    def _build_user_prompt(
        self,
        *,
        candidates: str,
        taste: str | None,
        candidate_labels: Sequence[str],
    ) -> str:
        del candidate_labels
        return agent4rec_user_prompt(
            candidates=candidates,
            taste=taste,
            entity_field=self.entity_field,
            entity_name=self.entity_name,
            entity_plural=self.entity_plural,
        )

    def _parse_response(
        self,
        text: str,
        *,
        labels: Sequence[str],
    ) -> dict[str, float]:
        return parse_agent4rec_watch_response(text, labels=labels)

    def _format_system_prompt(
        self,
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
            domain_name=self.domain_name,
            taste_label=self.taste_label,
        )

    def _format_item_description(
        self,
        *,
        row: Any,
        columns: tuple[str, ...],
    ) -> str:
        parts: list[str] = []
        for column in columns:
            value = getattr(row, column)
            if pd.isna(value) or value == "":
                continue
            column_label = self.column_labels.get(column, column)
            formatted_value = _format_prompt_value(
                value,
                parse_json_list=column in self.json_list_columns,
            )
            if not parts:
                parts.append(f"<- {formatted_value} ->")
            else:
                parts.append(f"<- {column_label}:{formatted_value} ->")
        return " ".join(parts) if parts else "<- no item description ->"

    def _require_taste_for_users(self, user_ids: Sequence[Any]) -> None:
        if self.profile_by_user_ is None:
            raise RuntimeError("Agent4RecYesNoScorer is not fitted")
        missing_users = [
            user_id
            for user_id in user_ids
            if user_id not in self.profile_by_user_
            or not self.profile_by_user_[user_id].taste
        ]
        if missing_users:
            raise RuntimeError(
                "Agent4RecYesNoScorer requires taste profiles for this method. "
                "Call scorer.build_taste(X_eval) before score(). "
                f"Missing users: {missing_users[:5]}"
            )

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class Agent4RecPreferenceYesNoScorer(Agent4RecYesNoScorer):
    """Agent4Rec profile-module adaptation for a binary preference target."""

    name = "agent4rec_preference_yes_no"

    def __init__(self, *, target_description: str, **kwargs: Any) -> None:
        if not target_description.strip():
            raise ValueError("target_description must be non-empty")
        super().__init__(**kwargs)
        self.target_description = target_description

    def _build_user_prompt(
        self,
        *,
        candidates: str,
        taste: str | None,
        candidate_labels: Sequence[str],
    ) -> str:
        del taste, candidate_labels
        return agent4rec_preference_user_prompt(
            candidates=candidates,
            target_description=self.target_description,
            entity_field=self.entity_field,
            entity_name=self.entity_name,
            entity_plural=self.entity_plural,
        )

    def _parse_response(
        self,
        text: str,
        *,
        labels: Sequence[str],
    ) -> dict[str, float]:
        return parse_agent4rec_preference_response(text, labels=labels)


def _chat_completion_text(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        raise ValueError("Chat completion response has no text content")
    return str(content)


def parse_agent4rec_watch_response(
    text: str,
    *,
    labels: Sequence[str],
) -> dict[str, float]:
    """Parse Agent4Rec `ID/{MOVIE|GAME|ITEM}/WATCH/REASON` responses by label."""

    return _parse_agent4rec_binary_response(
        text,
        labels=labels,
        decision_field="WATCH",
        decision_name="watch",
    )


def parse_agent4rec_preference_response(
    text: str,
    *,
    labels: Sequence[str],
) -> dict[str, float]:
    """Parse target-aware Agent4Rec `PREFERENCE: yes/no` responses by label."""

    return _parse_agent4rec_binary_response(
        text,
        labels=labels,
        decision_field="PREFERENCE",
        decision_name="preference",
    )


def _parse_agent4rec_binary_response(
    text: str,
    *,
    labels: Sequence[str],
    decision_field: str,
    decision_name: str,
) -> dict[str, float]:
    """Parse one strict labeled Agent4Rec binary-decision response."""

    if not labels:
        raise ValueError("labels must be non-empty")
    pattern = re.compile(
        r"(?:^|\n)\s*(?:(?:ID|LABEL):\s*)?(C\d+)\s*(?::|;)\s*"
        r"(?:\[[^\]]+\]\s*;?\s*)?"
        r"(?:MOVIE|GAME|ITEM):\s*(.*?)\s*;?\s*"
        + re.escape(decision_field)
        + r":\s*(.*?)\s*;?\s*"
        r"REASON:\s*(.*?)(?=\n\s*(?:(?:ID|LABEL):\s*)?C\d+\s*(?::|;)|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = pattern.findall(text)
    expected_labels = set(labels)
    parsed: dict[str, float] = {}
    duplicate_labels: list[str] = []
    unknown_labels: list[str] = []

    for raw_label, _, raw_decision, _ in matches:
        label = raw_label.strip().upper()
        if label not in expected_labels:
            unknown_labels.append(label)
            continue
        if label in parsed:
            duplicate_labels.append(label)
            continue

        decision = raw_decision.strip().strip(";").lower()
        if decision == "yes":
            parsed[label] = 1.0
        elif decision == "no":
            parsed[label] = 0.0
        else:
            raise ValueError(
                f"Invalid Agent4Rec {decision_field} value: {raw_decision!r}"
            )

    if unknown_labels:
        raise ValueError(
            "Unknown Agent4Rec candidate labels: "
            f"{sorted(set(unknown_labels))}"
        )
    if duplicate_labels:
        raise ValueError(
            "Duplicate Agent4Rec candidate labels: "
            f"{sorted(set(duplicate_labels))}"
        )

    missing_labels = [label for label in labels if label not in parsed]
    if missing_labels:
        raise ValueError(
            f"Missing Agent4Rec {decision_name} decisions: "
            f"{missing_labels}"
        )
    return parsed


def _candidate_labels(count: int) -> list[str]:
    if count < 1:
        raise ValueError("count must be positive")
    return [f"C{i}" for i in range(1, count + 1)]


def _format_agent4rec_taste(taste: str | None) -> str:
    if not taste:
        return ""
    taste_parts = [part.strip() for part in taste.split("| ") if part.strip()]
    return "; ".join(taste_parts).replace("I ", "")


def _format_prompt_value(value: Any, *, parse_json_list: bool = False) -> str:
    """Return compact text for scalar values shown in LLM prompts."""

    if parse_json_list and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list):
            return ", ".join(str(item) for item in parsed) if parsed else "none"
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
