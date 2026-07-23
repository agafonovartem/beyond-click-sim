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
    HistoryPromptFamily,
    OPENP5_STYLE_REGRESSION_USER_PROMPT_TEMPLATE,
    REGRESSION_SYSTEM_PROMPT,
    REGRESSION_USER_PROMPT_TEMPLATE,
    history_prompt_messages,
    history_prompt_metadata,
    validate_history_prompt_family,
)
from beyond_click_sim.scorers.history.selection import select_history_by_user


class LLMRegressor(Scorer):
    """LLM scorer for one-row numeric response prediction.

    V1 supports strict discrete numeric outputs: the model must return exactly
    one bare integer contained in `valid_values`. This matches rating-simulation
    tasks such as MovieLens 1-5 ratings while keeping continuous playtime-style
    targets for a later protocol.
    """

    name = "llm_regressor"

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        target_description: str,
        output_instructions: str,
        valid_values: Sequence[int],
        item_description_columns: tuple[str, ...] | None = None,
        history_description_columns: tuple[str, ...] | None = None,
        candidate_description_columns: tuple[str, ...] | None = None,
        user_column: str = "user_id",
        max_history_items: int | None = 30,
        temperature: float = 0.0,
        max_tokens: int = 64,
        column_labels: dict[str, str] | None = None,
        extra_body: dict | None = None,
        prompt_family: str = "simulator",
    ) -> None:
        if candidate_description_columns is None:
            candidate_description_columns = item_description_columns
        if history_description_columns is None:
            history_description_columns = item_description_columns
        if history_description_columns is None:
            history_description_columns = candidate_description_columns
        if not history_description_columns:
            raise ValueError("history_description_columns must be non-empty")
        if not candidate_description_columns:
            raise ValueError("candidate_description_columns must be non-empty")
        if max_history_items is not None and max_history_items < 0:
            raise ValueError("max_history_items must be non-negative")
        if not target_description:
            raise ValueError("target_description must be non-empty")
        if not output_instructions:
            raise ValueError("output_instructions must be non-empty")
        resolved_prompt_family = validate_history_prompt_family(prompt_family)

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
        self.output_instructions = output_instructions
        self.valid_values = valid_values_tuple
        self.item_description_columns = item_description_columns
        self.history_description_columns = history_description_columns
        self.candidate_description_columns = candidate_description_columns
        self.user_column = user_column
        self.max_history_items = max_history_items
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.column_labels = {} if column_labels is None else dict(column_labels)
        self.extra_body = extra_body
        self.prompt_family: HistoryPromptFamily = resolved_prompt_family
        self.history_by_user_: dict[Any, list[str]] | None = None

    @property
    def prompt_metadata(self) -> dict[str, object]:
        """Return prompt provenance suitable for an experiment manifest."""

        return history_prompt_metadata(self.prompt_family)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LLMRegressor":
        """Store formatted train rows as per-user response history."""

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.user_column, *self.history_description_columns])

        history_by_user: dict[Any, list[str]] = {}
        for user_id, history in select_history_by_user(
            X,
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
        """Score rows with one chat request per row."""

        if self.history_by_user_ is None:
            raise RuntimeError("LLMRegressor is not fitted")
        self._require_columns(
            X,
            [
                self.user_column,
                *self.candidate_description_columns,
            ],
        )

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
            scores.loc[index] = parse_regression_value_response(
                _chat_completion_text(response),
                valid_values=self.valid_values,
            )

        return scores

    def _build_messages(self, *, row: pd.Series) -> list[dict[str, str]]:
        history = (
            self.history_by_user_.get(row[self.user_column], [])
            if self.history_by_user_
            else []
        )
        candidate = self._format_item_description(
            row=row,
            label="Candidate",
            columns=self.candidate_description_columns,
        )
        prompt_values = {
            "history": "\n".join(history)
            if history
            else "- No interaction history available.",
            "candidate": candidate,
            "target_description": self.target_description,
            "output_instructions": self.output_instructions,
        }
        simulator_user_prompt = REGRESSION_USER_PROMPT_TEMPLATE.format(
            **prompt_values
        )
        openp5_style_user_prompt = (
            OPENP5_STYLE_REGRESSION_USER_PROMPT_TEMPLATE.format(**prompt_values)
        )
        return history_prompt_messages(
            prompt_family=self.prompt_family,
            simulator_system_prompt=REGRESSION_SYSTEM_PROMPT,
            simulator_user_prompt=simulator_user_prompt,
            openp5_style_user_prompt=openp5_style_user_prompt,
        )

    def _format_item_description(
        self,
        *,
        row: pd.Series,
        label: str,
        columns: tuple[str, ...],
    ) -> str:
        parts = []
        for column in columns:
            value = row[column]
            if pd.isna(value) or value == "":
                continue
            column_label = self.column_labels.get(column, column)
            parts.append(f"{column_label}: {_format_prompt_value(value)}")
        description = "; ".join(parts) if parts else "no item description"
        return f"{label}. {description}"

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def parse_regression_value_response(
    text: str,
    *,
    valid_values: Sequence[int],
) -> float:
    """Parse a strict bare integer response and validate it against allowed values."""

    raw_value = text.strip()
    if not re.fullmatch(r"[+-]?\d+", raw_value):
        raise ValueError(f"LLM response is not a bare integer: {text!r}")

    value = int(raw_value)
    if value not in set(valid_values):
        raise ValueError(
            f"LLM response {value!r} is outside valid values: {list(valid_values)}"
        )
    return float(value)
