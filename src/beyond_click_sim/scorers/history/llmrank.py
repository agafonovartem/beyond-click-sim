from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
from numbers import Integral
import random
from threading import Lock
from time import perf_counter
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.history.common import _chat_completion_text
from beyond_click_sim.scorers.history.listwise import (
    LLMInteractionListwiseRankingScorer,
    parse_ranked_labels_response,
)
from beyond_click_sim.scorers.history.prompts import (
    LLMRANK_PROMPT_VERSION,
    LLMRANK_SYSTEM_PROMPT,
    LLMRANK_USER_PROMPT_TEMPLATE,
)


class LLMRankScorer(LLMInteractionListwiseRankingScorer):
    """Recency-focused ranking with three independently permuted presentations.

    Fit rows must already be chronological within each user. The temporal runner
    verifies this upstream contract through the task manifest. A failed round
    raises so the existing runner retries the entire ensemble, with the same
    permutations. Neither scoring nor retries inspect evaluation targets.
    """

    name = "llmrank"
    bootstrap_rounds = 3
    score_mapping = "score = sum(n_candidates - rank_b for b in 1..3)"
    seed_algorithm = "sha256_typed_group_and_ordered_item_ids_python_random_v1"

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        bootstrap_seed: int = 0,
        item_column: str = "item_id",
        **kwargs: Any,
    ) -> None:
        if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral):
            raise ValueError("bootstrap_seed must be an integer")
        if kwargs.get("prompt_family", "simulator") != "simulator":
            raise ValueError("LLMRank supports only its dedicated system/user prompt")
        kwargs.setdefault("max_history_items", 20)
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_tokens", 512)
        super().__init__(client, model, **kwargs)
        self.bootstrap_seed = int(bootstrap_seed)
        self.item_column = item_column
        self._diagnostics: list[dict[str, Any]] = []
        self._attempts: dict[str, int] = {}
        self._diagnostics_lock = Lock()

    @property
    def prompt_metadata(self) -> dict[str, object]:
        return {
            "family": "llmrank",
            "version": LLMRANK_PROMPT_VERSION,
            "source": "https://arxiv.org/abs/2305.08845v2",
            "system_role": True,
            "temporal_language": True,
            "recency_reference": "last_selected_history_label",
            "output_format": "complete_candidate_label_ranking",
        }

    def drain_diagnostics(self) -> list[dict[str, Any]]:
        """Collect completed attempts after a scoring phase, including failures."""

        with self._diagnostics_lock:
            records = sorted(
                self._diagnostics,
                key=lambda record: (record["request_key"], record["attempt"]),
            )
            self._diagnostics = []
            self._attempts = {}
        return records

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.history_by_user_ is None:
            raise RuntimeError("LLMRankScorer is not fitted")
        self._require_columns(
            X,
            [
                self.user_column,
                self.candidate_group_column,
                self.item_column,
                *self.candidate_description_columns,
            ],
        )
        if not X.index.is_unique:
            raise ValueError("LLMRank requires unique input row indices")
        identifiers = [self.user_column, self.candidate_group_column, self.item_column]
        if X[identifiers].isna().any().any():
            raise ValueError(
                "LLMRank requires non-null user, group and item identifiers"
            )

        scores = pd.Series(index=X.index, dtype=float, name="score")
        for group_id, group in X.groupby(self.candidate_group_column, sort=False):
            user_ids = group[self.user_column].drop_duplicates()
            if len(user_ids) != 1:
                raise ValueError("Each candidate group must contain exactly one user")
            if group[self.item_column].duplicated().any():
                raise ValueError(
                    "LLMRank requires unique candidate item IDs within a group"
                )
            user_id = user_ids.iloc[0]
            if not self.history_by_user_.get(user_id):
                raise ValueError(
                    f"LLMRank requires non-empty fitted history for user {user_id!r}"
                )
            scores.loc[group.index] = self._score_group(group_id, user_id, group)
        return scores

    def _score_group(
        self,
        group_id: Any,
        user_id: Any,
        group: pd.DataFrame,
    ) -> list[float]:
        item_ids = group[self.item_column].tolist()
        key_payload = [
            self.bootstrap_seed,
            _typed_identifier(user_id),
            _typed_identifier(group_id),
            [_typed_identifier(item_id) for item_id in item_ids],
        ]
        request_key = sha256(
            json.dumps(key_payload, ensure_ascii=False).encode()
        ).hexdigest()
        rng = random.Random(int(request_key, 16))
        with self._diagnostics_lock:
            attempt = self._attempts.get(request_key, 0) + 1
            self._attempts[request_key] = attempt
        record: dict[str, Any] = {
            "request_key": request_key,
            "user_id": _json_scalar(user_id),
            "candidate_group": _json_scalar(group_id),
            "candidate_item_ids": [_json_scalar(item_id) for item_id in item_ids],
            "attempt": attempt,
            "completed": False,
            "rounds": [],
        }
        n = len(group)
        labels = [f"C{position}" for position in range(1, n + 1)]
        aggregate = [0.0] * n
        try:
            for round_number in range(1, self.bootstrap_rounds + 1):
                permutation = rng.sample(range(n), n)
                round_items = [
                    _json_scalar(item_ids[position]) for position in permutation
                ]
                messages = self._build_messages(
                    user_id=user_id,
                    candidates=group.iloc[permutation],
                    labels=labels,
                )
                round_record: dict[str, Any] = {
                    "round": round_number,
                    "permutation": permutation,
                    "label_to_item_id": dict(zip(labels, round_items, strict=True)),
                    "request_issued": True,
                    "usage": None,
                }
                record["rounds"].append(round_record)
                started = perf_counter()
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        **({"extra_body": self.extra_body} if self.extra_body else {}),
                    )
                    round_record["usage"] = _response_usage(response)
                    text = _chat_completion_text(response)
                    round_record["response_text"] = text
                    ranked = parse_ranked_labels_response(text, labels=labels)
                    round_record["ranked_labels"] = ranked
                    positions = dict(zip(labels, permutation, strict=True))
                    round_record["ranked_item_ids"] = [
                        _json_scalar(item_ids[positions[label]]) for label in ranked
                    ]
                    for rank, label in enumerate(ranked, start=1):
                        aggregate[positions[label]] += n - rank
                except Exception as error:
                    round_record["error"] = repr(error)
                    raise
                finally:
                    round_record["elapsed_seconds"] = perf_counter() - started
            record["completed"] = True
            record["scores"] = aggregate
            return aggregate
        finally:
            with self._diagnostics_lock:
                self._diagnostics.append(record)

    def _build_messages(
        self,
        *,
        user_id: Any,
        candidates: pd.DataFrame,
        labels: Sequence[str],
    ) -> list[dict[str, str]]:
        history = (
            self.history_by_user_.get(user_id, []) if self.history_by_user_ else []
        )
        if not history:
            raise ValueError("LLMRank requires non-empty fitted history")
        candidate_lines = [
            self._format_item_description(
                row=row,
                label=label,
                columns=self.candidate_description_columns,
            )
            for label, row in zip(
                labels, candidates.itertuples(index=False), strict=True
            )
        ]
        prompt = LLMRANK_USER_PROMPT_TEMPLATE.format(
            history="\n".join(history),
            recent_history_label=f"H{len(history)}",
            candidates="\n".join(candidate_lines),
            output_labels=", ".join(labels),
            candidate_count=len(labels),
        )
        return [
            {"role": "system", "content": LLMRANK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if hasattr(value, "item"):
        value = value.item()
    return (
        value
        if value is None or isinstance(value, (str, int, float, bool))
        else str(value)
    )


def _typed_identifier(value: Any) -> list[Any]:
    scalar = _json_scalar(value)
    return [type(scalar).__name__, scalar]


def _response_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    result = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = (
            usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
        )
        if value is not None:
            result[field] = int(value)
    return result or None
