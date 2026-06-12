from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.prompts import (
    INTERACTION_YES_NO_SYSTEM_PROMPT,
    INTERACTION_YES_NO_USER_PROMPT_TEMPLATE,
    REGRESSION_SYSTEM_PROMPT,
    REGRESSION_USER_PROMPT_TEMPLATE,
)

# TODO: Think about creating a separate base class for LLMScorers.
class LLMInteractionYesNoScorer(Scorer):
    """LLM yes/no scorer for explicit interaction-alignment candidate groups.

    This scorer implements the simple Agent4Rec/SimUSER-style setup: given a
    user's observed interaction history and a small explicit candidate set, ask
    an OpenAI-compatible chat model whether the user would interact with each
    candidate. It is not a candidate-free recommender and does not generate a
    top-k list by itself.

    Input rows are regular task rows. The scorer only needs:
    - `user_column` to group train history by user and identify the test user;
    - `candidate_group_column` in `score(X)` to batch candidates from one choice
      set into a single prompt;
    - `history_description_columns` to format fitted history rows;
    - `candidate_description_columns` to format scored candidate rows.

    Keeping history and candidate columns separate avoids target leakage. For
    example, train history may include a known `rating`, while candidate rows
    should usually expose only item metadata such as title, genre, year, or
    description. `item_description_columns` is a backward-compatible shortcut
    that uses the same columns for both sides.

    During `fit(X, y)`, `y` is only checked for length. All train rows are stored
    as user history in input order, formatted as:
    `H1. item_title: Toy Story; item_genre: Animation`.
    `max_history_items=None` keeps the full history; an integer keeps the latest
    N train rows per user and renumbers them from `H1`.

    During `score(X)`, every candidate group must contain exactly one user.
    Candidates are formatted in group row order as `C1`, `C2`, ... and sent in
    one chat request per group. The model must answer with lines like
    `C1: yes` / `C2: no`. The returned Series is named `"score"` and preserves
    the input index, with `1.0` for yes and `0.0` for no.
    """

    name = "llm_interaction_yes_no"

    def __init__(
        self,
        client: Any,
        model: str,
        item_description_columns: tuple[str, ...] | None = None,
        history_description_columns: tuple[str, ...] | None = None,
        candidate_description_columns: tuple[str, ...] | None = None,
        user_column: str = "user_id",
        candidate_group_column: str = "candidate_group",
        max_history_items: int | None = 30,
        temperature: float = 0.0,
        max_tokens: int = 512,
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

        self.client = client
        self.model = model
        self.item_description_columns = item_description_columns
        self.history_description_columns = history_description_columns
        self.candidate_description_columns = candidate_description_columns
        self.user_column = user_column
        self.candidate_group_column = candidate_group_column
        self.max_history_items = max_history_items
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_by_user_: dict[Any, list[str]] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LLMInteractionYesNoScorer":
        """Store formatted train rows as per-user interaction history.

        The target is intentionally ignored in v1: the train table is assumed to
        already contain the history we want to show to the LLM. This keeps the
        scorer independent from a particular target definition.
        """

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.user_column, *self.history_description_columns])

        history_rows_by_user: dict[Any, list[Any]] = {}
        for row in X.itertuples(index=False):
            user_id = getattr(row, self.user_column)
            history_rows_by_user.setdefault(user_id, []).append(row)

        history_by_user: dict[Any, list[str]] = {}
        for user_id, rows in history_rows_by_user.items():
            if self.max_history_items is None:
                selected_rows = rows
            elif self.max_history_items == 0:
                selected_rows = []
            else:
                selected_rows = rows[-self.max_history_items :]
            history_by_user[user_id] = [
                self._format_item_description(
                    row=row,
                    label=f"H{position}",
                    columns=self.history_description_columns,
                )
                for position, row in enumerate(selected_rows, start=1)
            ]

        self.history_by_user_ = history_by_user
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Score explicit candidate groups with one LLM call per group.

        Each group is converted into a single prompt containing the fitted user
        history and all candidates from that group. The strict parser maps the
        model's yes/no answer for every `Ck` label back to the original row index.
        """

        if self.history_by_user_ is None:
            raise RuntimeError("LLMInteractionYesNoScorer is not fitted")
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
            # print(messages)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # print(response)
            parsed = parse_yes_no_response(
                _chat_completion_text(response),
                labels=labels,
            )
            scores.loc[group.index] = [parsed[label] for label in labels]

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
        user_prompt = INTERACTION_YES_NO_USER_PROMPT_TEMPLATE.format(
            history="\n".join(history) if history else "- No interaction history available.",
            candidates="\n".join(candidate_lines),
            output_labels="\n".join(f"{label}:" for label in labels),
        )
        return [
            {"role": "system", "content": INTERACTION_YES_NO_SYSTEM_PROMPT},
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
            parts.append(f"{column}: {value}")
        description = "; ".join(parts) if parts else "no item description"
        return f"{label}. {description}"

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def parse_yes_no_response(text: str, *, labels: Sequence[str]) -> dict[str, float]:
    """Parse strict `Ck: yes/no` responses into numeric scores."""

    expected = set(labels)
    parsed: dict[str, float] = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("LLM response is empty")

    for line in lines:
        if ":" not in line:
            raise ValueError(f"Could not parse response line: {line!r}")
        label, raw_answer = [part.strip() for part in line.split(":", 1)]
        if label not in expected:
            raise ValueError(f"Unknown candidate label: {label!r}")
        if label in parsed:
            raise ValueError(f"Duplicate candidate label: {label!r}")

        answer = raw_answer.lower()
        if answer == "yes":
            parsed[label] = 1.0
        elif answer == "no":
            parsed[label] = 0.0
        else:
            raise ValueError(f"Invalid yes/no answer for {label!r}: {raw_answer!r}")

    missing = [label for label in labels if label not in parsed]
    if missing:
        raise ValueError(f"Missing candidate labels: {missing}")
    return parsed


def _chat_completion_text(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        raise ValueError("Chat completion response has no text content")
    return str(content)


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

        valid_values_tuple = tuple(valid_values)
        if not valid_values_tuple:
            raise ValueError("valid_values must be non-empty")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in valid_values_tuple):
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
        self.history_by_user_: dict[Any, list[str]] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LLMRegressor":
        """Store formatted train rows as per-user response history."""

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.user_column, *self.history_description_columns])

        history_rows_by_user: dict[Any, list[pd.Series]] = {}
        for _, row in X.iterrows():
            user_id = row[self.user_column]
            history_rows_by_user.setdefault(user_id, []).append(row)

        history_by_user: dict[Any, list[str]] = {}
        for user_id, rows in history_rows_by_user.items():
            if self.max_history_items is None:
                selected_rows = rows
            elif self.max_history_items == 0:
                selected_rows = []
            else:
                selected_rows = rows[-self.max_history_items :]
            history_by_user[user_id] = [
                self._format_item_description(
                    row=row,
                    label=f"H{position}",
                    columns=self.history_description_columns,
                )
                for position, row in enumerate(selected_rows, start=1)
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
            )
            scores.loc[index] = parse_regression_value_response(
                _chat_completion_text(response),
                valid_values=self.valid_values,
            )

        return scores

    def _build_messages(self, *, row: pd.Series) -> list[dict[str, str]]:
        history = self.history_by_user_.get(row[self.user_column], []) if self.history_by_user_ else []
        candidate = self._format_item_description(
            row=row,
            label="Candidate",
            columns=self.candidate_description_columns,
        )
        user_prompt = REGRESSION_USER_PROMPT_TEMPLATE.format(
            history="\n".join(history) if history else "- No interaction history available.",
            candidate=candidate,
            target_description=self.target_description,
            output_instructions=self.output_instructions,
        )
        return [
            {"role": "system", "content": REGRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

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
            parts.append(f"{column}: {value}")
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
