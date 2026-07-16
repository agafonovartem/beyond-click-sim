from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.history.common import (
    _chat_completion_text,
    _format_prompt_value,
)
from beyond_click_sim.scorers.history.prompts import (
    INTERACTION_LISTWISE_RANKING_SYSTEM_PROMPT,
    INTERACTION_LISTWISE_RANKING_USER_PROMPT_TEMPLATE,
    PREFERENCE_LISTWISE_RANKING_SYSTEM_PROMPT,
    PREFERENCE_LISTWISE_RANKING_USER_PROMPT_TEMPLATE,
)
from beyond_click_sim.scorers.history.selection import select_history_by_user


class LLMInteractionListwiseRankingScorer(Scorer):
    """LLM listwise ranking scorer for explicit interaction candidate groups.

    The scorer asks the model to rank all candidates in a group from most likely
    to least likely to be interacted with by the user. It then maps the returned
    order to numeric scores, preserving the existing ranking-evaluation path.
    This is a listwise preference protocol, not a pointwise yes/no simulator.
    """

    name = "llm_interaction_listwise_ranking"

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        history_description_columns: tuple[str, ...],
        candidate_description_columns: tuple[str, ...],
        user_column: str = "user_id",
        candidate_group_column: str = "candidate_group",
        max_history_items: int | None = 30,
        temperature: float = 0.0,
        max_tokens: int = 256,
        column_labels: dict[str, str] | None = None,
        json_list_columns: tuple[str, ...] = (),
        extra_body: dict | None = None,
    ) -> None:
        if not history_description_columns:
            raise ValueError("history_description_columns must be non-empty")
        if not candidate_description_columns:
            raise ValueError("candidate_description_columns must be non-empty")
        if max_history_items is not None and max_history_items < 0:
            raise ValueError("max_history_items must be non-negative")

        self.client = client
        self.model = model
        self.history_description_columns = history_description_columns
        self.candidate_description_columns = candidate_description_columns
        self.user_column = user_column
        self.candidate_group_column = candidate_group_column
        self.max_history_items = max_history_items
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.column_labels = {} if column_labels is None else dict(column_labels)
        self.json_list_columns = tuple(json_list_columns)
        self.extra_body = extra_body
        self.history_by_user_: dict[Any, list[str]] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        history_user_ids: Sequence[Any] | None = None,
    ) -> "LLMInteractionListwiseRankingScorer":
        """Store formatted train rows as per-user interaction history."""

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.user_column, *self.history_description_columns])
        history_rows = (
            X
            if history_user_ids is None
            else X[X[self.user_column].isin(history_user_ids)].copy()
        )

        history_by_user: dict[Any, list[str]] = {}
        for user_id, history in select_history_by_user(
            history_rows,
            user_column=self.user_column,
            item_column=None,
            max_history_items=self.max_history_items,
        ).items():
            history_by_user[user_id] = [
                self._format_item_description(
                    row=row,
                    label=f"H{position}",
                    columns=self.history_description_columns,
                )
                for position, (_, row) in enumerate(history.rows.iterrows(), start=1)
            ]

        self.history_by_user_ = history_by_user
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Score explicit candidate groups with one listwise LLM call per group."""

        if self.history_by_user_ is None:
            raise RuntimeError("LLMInteractionListwiseRankingScorer is not fitted")
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

            labels = [f"C{position}" for position in range(1, len(group) + 1)]
            messages = self._build_messages(
                user_id=user_ids.iloc[0],
                candidates=group,
                labels=labels,
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **({"extra_body": self.extra_body} if self.extra_body else {}),
            )
            ranked_labels = parse_ranked_labels_response(
                _chat_completion_text(response),
                labels=labels,
            )
            label_scores = {
                label: float(len(ranked_labels) - rank)
                for rank, label in enumerate(ranked_labels, start=1)
            }
            scores.loc[group.index] = [label_scores[label] for label in labels]

        return scores

    def _build_messages(
        self,
        *,
        user_id: Any,
        candidates: pd.DataFrame,
        labels: Sequence[str],
    ) -> list[dict[str, str]]:
        history = self.history_by_user_.get(user_id, []) if self.history_by_user_ else []
        candidate_lines = [
            self._format_item_description(
                row=row,
                label=label,
                columns=self.candidate_description_columns,
            )
            for label, row in zip(labels, candidates.itertuples(index=False), strict=True)
        ]
        user_prompt = INTERACTION_LISTWISE_RANKING_USER_PROMPT_TEMPLATE.format(
            history="\n".join(history)
            if history
            else "- No interaction history available.",
            candidates="\n".join(candidate_lines),
            output_labels=", ".join(labels),
            candidate_count=len(labels),
        )
        return [
            {"role": "system", "content": INTERACTION_LISTWISE_RANKING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _format_item_description(
        self,
        *,
        row: Any,
        label: str,
        columns: tuple[str, ...],
    ) -> str:
        parts = []
        for column in columns:
            value = getattr(row, column)
            if pd.isna(value) or value == "":
                continue
            column_label = self.column_labels.get(column, column)
            formatted_value = _format_prompt_value(
                value,
                parse_json_list=column in self.json_list_columns,
            )
            parts.append(f"{column_label}: {formatted_value}")
        description = "; ".join(parts) if parts else "no item description"
        return f"{label}. {description}"

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class LLMPreferenceListwiseRankingScorer(LLMInteractionListwiseRankingScorer):
    """History-conditioned listwise scorer for a positive-preference target."""

    name = "llm_preference_listwise_ranking"

    def __init__(self, *, target_description: str, **kwargs: Any) -> None:
        if not target_description.strip():
            raise ValueError("target_description must be non-empty")
        super().__init__(**kwargs)
        self.target_description = target_description

    def _build_messages(
        self,
        *,
        user_id: Any,
        candidates: pd.DataFrame,
        labels: Sequence[str],
    ) -> list[dict[str, str]]:
        history = self.history_by_user_.get(user_id, []) if self.history_by_user_ else []
        candidate_lines = [
            self._format_item_description(
                row=row,
                label=label,
                columns=self.candidate_description_columns,
            )
            for label, row in zip(labels, candidates.itertuples(index=False), strict=True)
        ]
        user_prompt = PREFERENCE_LISTWISE_RANKING_USER_PROMPT_TEMPLATE.format(
            history="\n".join(history)
            if history
            else "- No feedback history available.",
            target_description=self.target_description,
            candidates="\n".join(candidate_lines),
            output_labels=", ".join(labels),
            candidate_count=len(labels),
        )
        return [
            {"role": "system", "content": PREFERENCE_LISTWISE_RANKING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]


def parse_ranked_labels_response(text: str, *, labels: Sequence[str]) -> list[str]:
    """Parse a strict ranked list of candidate labels."""

    expected = set(labels)
    if len(expected) != len(labels):
        raise ValueError("labels must be unique")

    ranked = [
        match.group(0).upper()
        for match in re.finditer(r"\bC\d+\b", text, re.IGNORECASE)
    ]
    if not ranked:
        raise ValueError("LLM ranking response contains no candidate labels")

    unknown = [label for label in ranked if label not in expected]
    if unknown:
        raise ValueError(f"Unknown candidate labels: {sorted(set(unknown))}")

    duplicates = sorted({label for label in ranked if ranked.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate candidate labels: {duplicates}")

    missing = [label for label in labels if label not in ranked]
    if missing:
        raise ValueError(f"Missing candidate labels: {missing}")
    return ranked
