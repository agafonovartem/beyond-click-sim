from __future__ import annotations

from collections.abc import Sequence
import json
import math
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.history.common import _chat_completion_text
from beyond_click_sim.scorers.history.listwise import (
    LLMInteractionListwiseRankingScorer,
)
from beyond_click_sim.scorers.history.prompts import (
    TARGET_RESPONSE_PROBABILITY_SYSTEM_PROMPT,
    TARGET_RESPONSE_PROBABILITY_USER_PROMPT_TEMPLATE,
)


class DuplicateJSONKeyError(ValueError):
    """Raised when a supposedly strict model JSON object repeats a key."""


class LLMTargetResponseProbabilityScorer(
    LLMInteractionListwiseRankingScorer
):
    """Predict one response probability per candidate in a batched request.

    Candidate groups are only an API batching boundary.  Every returned value
    is parsed as an independent probability, so downstream code may evaluate
    natural-rate pointwise metrics as well as ranking within each group.
    """

    name = "llm_target_response_probability"

    def __init__(self, *, target_description: str, **kwargs: Any) -> None:
        if not target_description.strip():
            raise ValueError("target_description must be non-empty")
        super().__init__(**kwargs)
        self.target_description = target_description

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.history_by_user_ is None:
            raise RuntimeError("Scorer is not fitted")
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
                raise ValueError("Each candidate group must contain one user")
            labels = [f"C{position}" for position in range(1, len(group) + 1)]
            response = self.client.chat.completions.create(
                **self._request(
                    messages=self._build_messages(
                        user_id=user_ids.iloc[0],
                        candidates=group,
                        labels=labels,
                    )
                )
            )
            self._record_usage(getattr(response, "usage", None))
            probabilities = parse_probability_object_response(
                _chat_completion_text(response),
                labels=labels,
            )
            scores.loc[group.index] = [
                probabilities[label] for label in labels
            ]
        return scores

    def _request(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        request.update(self.request_kwargs)
        if self.extra_body:
            request["extra_body"] = self.extra_body
        return request

    def _build_messages(
        self,
        *,
        user_id: Any,
        candidates: pd.DataFrame,
        labels: Sequence[str],
    ) -> list[dict[str, str]]:
        history = (
            self.history_by_user_.get(user_id, [])
            if self.history_by_user_
            else []
        )
        candidate_lines = [
            self._format_item_description(
                row=row,
                label=label,
                columns=self.candidate_description_columns,
            )
            for label, row in zip(
                labels,
                candidates.itertuples(index=False),
                strict=True,
            )
        ]
        prompt = TARGET_RESPONSE_PROBABILITY_USER_PROMPT_TEMPLATE.format(
            history="\n".join(history)
            if history
            else "- No response history available.",
            target_description=self.target_description,
            candidates="\n".join(candidate_lines),
            output_labels=", ".join(labels),
        )
        return [
            {
                "role": "system",
                "content": TARGET_RESPONSE_PROBABILITY_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ]


def parse_probability_object_response(
    text: str,
    *,
    labels: Sequence[str],
) -> dict[str, float]:
    """Parse a strict JSON label-to-probability object."""

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateJSONKeyError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(text.strip(), object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, DuplicateJSONKeyError) as error:
        raise ValueError(f"Invalid probability JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("Probability response must be one JSON object")

    expected = list(labels)
    if set(parsed) != set(expected) or len(parsed) != len(expected):
        raise ValueError(
            "Probability response keys must exactly match candidate labels"
        )
    result: dict[str, float] = {}
    for label in expected:
        raw_value = parsed[label]
        if isinstance(raw_value, bool) or not isinstance(
            raw_value,
            (int, float),
        ):
            raise ValueError(f"Probability for {label} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Probability for {label} must be finite and in [0, 1]"
            )
        result[label] = value
    return result
