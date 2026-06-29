from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
import re
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.agent4rec.prompts import (
    AGENT4REC_FORCED_ITEMS_USER_PROMPT_TEMPLATE,
    agent4rec_system_prompt,
)
from beyond_click_sim.scorers.agent4rec.profiles import (
    Agent4RecProfileGenerator,
    Agent4RecUserProfile,
)
from beyond_click_sim.scorers.base import Scorer


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
        temperature: float = 0.2,
        max_tokens: int = 1000,
        column_labels: dict[str, str] | None = None,
        extra_body: dict | None = None,
    ) -> None:
        if candidate_description_columns is None:
            candidate_description_columns = item_description_columns
        if not candidate_description_columns:
            raise ValueError("candidate_description_columns must be non-empty")

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
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.column_labels = {} if column_labels is None else dict(column_labels)
        self.extra_body = extra_body
        self.profile_by_user_: dict[Any, Agent4RecUserProfile] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Agent4RecYesNoScorer":
        """Build Agent4Rec profiles from the scorer's fitted train rows."""

        self.profile_by_user_ = self.profile_generator.build(X, y)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Score explicit candidate groups with one LLM call per group."""

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

        scores = pd.Series(index=X.index, dtype=float, name="score")
        for _, group in X.groupby(self.candidate_group_column, sort=False):
            user_ids = group[self.user_column].drop_duplicates()
            if len(user_ids) != 1:
                raise ValueError("Each candidate group must contain exactly one user")

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
            parsed = parse_agent4rec_watch_response(
                _chat_completion_text(response),
                labels=candidate_labels,
            )
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
        system_prompt = self._format_system_prompt(profile)
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
        user_prompt = AGENT4REC_FORCED_ITEMS_USER_PROMPT_TEMPLATE.format(
            candidates="\n".join(candidate_lines),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _format_system_prompt(profile: Agent4RecUserProfile) -> str:
        return agent4rec_system_prompt(
            taste=_format_agent4rec_taste(profile.taste) or None,
            activity=profile.activity_description,
            conformity=profile.conformity_description,
            diversity=profile.diversity_description,
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
            formatted_value = _format_prompt_value(value)
            if not parts:
                parts.append(f"<- {formatted_value} ->")
            else:
                parts.append(f"<- {column_label}:{formatted_value} ->")
        return " ".join(parts) if parts else "<- no item description ->"

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


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
    """Parse Agent4Rec `ID/MOVIE/WATCH/REASON` responses by candidate label."""

    if not labels:
        raise ValueError("labels must be non-empty")
    pattern = re.compile(
        r"(?:^|\n)\s*(?:ID|LABEL):\s*(C\d+)\s*;?\s*"
        r"MOVIE:\s*(.*?)\s*;?\s*WATCH:\s*(.*?)\s*;?\s*"
        r"REASON:\s*(.*?)(?=\n\s*(?:ID|LABEL):\s*C\d+\b|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = pattern.findall(text)
    expected_labels = set(labels)
    parsed: dict[str, float] = {}
    duplicate_labels: list[str] = []
    unknown_labels: list[str] = []

    for raw_label, _, raw_watch, _ in matches:
        label = raw_label.strip().upper()
        if label not in expected_labels:
            unknown_labels.append(label)
            continue
        if label in parsed:
            duplicate_labels.append(label)
            continue

        watch = raw_watch.strip().strip(";").lower()
        if watch == "yes":
            parsed[label] = 1.0
        elif watch == "no":
            parsed[label] = 0.0
        else:
            raise ValueError(f"Invalid Agent4Rec WATCH value: {raw_watch!r}")

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
            "Missing Agent4Rec watch decisions: "
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
