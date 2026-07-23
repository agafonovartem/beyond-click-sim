from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.history.common import (
    _chat_completion_text,
    _format_prompt_value,
)
from beyond_click_sim.scorers.history.prompts import (
    HistoryPromptFamily,
    INTERACTION_YES_NO_SYSTEM_PROMPT,
    INTERACTION_YES_NO_USER_PROMPT_TEMPLATE,
    OPENP5_STYLE_INTERACTION_YES_NO_USER_PROMPT_TEMPLATE,
    OPENP5_STYLE_POLICY_RANKING_ITEMWISE_USER_PROMPT_TEMPLATE,
    OPENP5_STYLE_PREFERENCE_YES_NO_USER_PROMPT_TEMPLATE,
    POLICY_RANKING_ITEMWISE_SYSTEM_PROMPT,
    POLICY_RANKING_ITEMWISE_USER_PROMPT_TEMPLATE,
    PREFERENCE_YES_NO_SYSTEM_PROMPT,
    PREFERENCE_YES_NO_USER_PROMPT_TEMPLATE,
    history_prompt_messages,
    history_prompt_metadata,
    validate_history_prompt_family,
)
from beyond_click_sim.scorers.history.selection import select_history_by_user


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
        column_labels: dict[str, str] | None = None,
        json_list_columns: tuple[str, ...] = (),
        extra_body: dict | None = None,
        prompt_style: str = "batch",
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
        if prompt_style not in ("batch", "itemwise"):
            raise ValueError(
                f"prompt_style must be 'batch' or 'itemwise', got {prompt_style!r}"
            )
        resolved_prompt_family = validate_history_prompt_family(prompt_family)

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
        self.column_labels = {} if column_labels is None else dict(column_labels)
        self.json_list_columns = tuple(json_list_columns)
        self.extra_body = extra_body
        self.prompt_style = prompt_style
        self.prompt_family: HistoryPromptFamily = resolved_prompt_family
        self.history_by_user_: dict[Any, list[str]] | None = None

    @property
    def prompt_metadata(self) -> dict[str, object]:
        """Return prompt provenance suitable for an experiment manifest."""

        return history_prompt_metadata(self.prompt_family)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        history_user_ids: Sequence[Any] | None = None,
    ) -> "LLMInteractionYesNoScorer":
        """Store formatted train rows as per-user interaction history.

        The target is intentionally ignored in v1: the train table is assumed to
        already contain the history we want to show to the LLM. This keeps the
        scorer independent from a particular target definition.
        """

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
            else:
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
                text = _chat_completion_text(response)
                try:
                    parsed = parse_yes_no_response(text, labels=labels)
                except ValueError as error:
                    raise ValueError(f"{error} | raw_response={text!r}") from error
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
        prompt_values = {
            "history": "\n".join(history)
            if history
            else "- No interaction history available.",
            "candidates": "\n".join(candidate_lines),
            "output_labels": "\n".join(f"{label}:" for label in labels),
        }
        simulator_user_prompt = INTERACTION_YES_NO_USER_PROMPT_TEMPLATE.format(
            **prompt_values
        )
        openp5_style_user_prompt = (
            OPENP5_STYLE_INTERACTION_YES_NO_USER_PROMPT_TEMPLATE.format(
                **prompt_values
            )
        )
        return history_prompt_messages(
            prompt_family=self.prompt_family,
            simulator_system_prompt=INTERACTION_YES_NO_SYSTEM_PROMPT,
            simulator_user_prompt=simulator_user_prompt,
            openp5_style_user_prompt=openp5_style_user_prompt,
        )

    def _build_itemwise_messages(
        self,
        *,
        user_id: Any,
        candidate: Any,
    ) -> list[dict[str, str]]:
        history = self.history_by_user_.get(user_id, []) if self.history_by_user_ else []
        candidate_fields = self._format_item_fields(
            row=candidate,
            columns=self.candidate_description_columns,
        )
        prompt_values = {
            "history": "\n".join(history)
            if history
            else "- No interaction history available.",
            "candidate": candidate_fields,
        }
        simulator_user_prompt = POLICY_RANKING_ITEMWISE_USER_PROMPT_TEMPLATE.format(
            **prompt_values
        )
        openp5_style_user_prompt = (
            OPENP5_STYLE_POLICY_RANKING_ITEMWISE_USER_PROMPT_TEMPLATE.format(
                **prompt_values
            )
        )
        return history_prompt_messages(
            prompt_family=self.prompt_family,
            simulator_system_prompt=POLICY_RANKING_ITEMWISE_SYSTEM_PROMPT,
            simulator_user_prompt=simulator_user_prompt,
            openp5_style_user_prompt=openp5_style_user_prompt,
        )

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

    def _format_item_fields(
        self,
        *,
        row: Any,
        columns: tuple[str, ...],
    ) -> str:
        """Format item fields without a label prefix."""
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
        return "; ".join(parts) if parts else "no item description"

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


# TODO: unify with interaction scorer. The only difference seems to be prompts
class LLMPreferenceYesNoScorer(LLMInteractionYesNoScorer):
    """LLM scorer for an explicit binary positive-preference target.

    The formatting, grouping, strict parser, and history selection are shared
    with :class:`LLMInteractionYesNoScorer`. The prompt is intentionally
    separate: preference prediction asks whether an observed response meets a
    dataset-specific target, rather than whether an interaction occurs.
    """

    name = "llm_preference_yes_no"

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
        prompt_values = {
            "history": "\n".join(history)
            if history
            else "- No feedback history available.",
            "target_description": self.target_description,
            "candidates": "\n".join(candidate_lines),
            "output_labels": "\n".join(f"{label}:" for label in labels),
        }
        simulator_user_prompt = PREFERENCE_YES_NO_USER_PROMPT_TEMPLATE.format(
            **prompt_values
        )
        openp5_style_user_prompt = (
            OPENP5_STYLE_PREFERENCE_YES_NO_USER_PROMPT_TEMPLATE.format(
                **prompt_values
            )
        )
        return history_prompt_messages(
            prompt_family=self.prompt_family,
            simulator_system_prompt=PREFERENCE_YES_NO_SYSTEM_PROMPT,
            simulator_user_prompt=simulator_user_prompt,
            openp5_style_user_prompt=openp5_style_user_prompt,
        )


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


def parse_single_yes_no_response(text: str) -> float:
    """Parse a bare yes/no response into a numeric score (1.0 or 0.0).

    Accepts a response that is exactly the word 'yes' or 'no' (case-insensitive,
    surrounding whitespace stripped). Raises ValueError for anything else.
    """
    answer = text.strip().lower()
    if answer == "yes":
        return 1.0
    if answer == "no":
        return 0.0
    raise ValueError(f"Expected a bare 'yes' or 'no' response, got: {text!r}")
