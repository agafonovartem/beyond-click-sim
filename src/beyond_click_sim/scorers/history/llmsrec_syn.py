"""LLMSRec-Syn with training-only, target-cutoff-specific demonstrations."""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Integral
import random
import re
from threading import RLock
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.history.common import _chat_completion_text
from beyond_click_sim.scorers.history.listwise import parse_ranked_labels_response
from beyond_click_sim.scorers.history.llmsrec_syn_prompts import (
    AUTHOR_REVISION,
    PROMPT_VERSION,
    messages,
)
from beyond_click_sim.scorers.history.llmsrec_syn_retrieval import (
    CachedEmbeddings,
    digest,
    identity,
    scalar,
    serialize_titles,
    token_usage,
)


@dataclass(frozen=True)
class Event:
    event_id: str | int
    item_id: str | int
    title: str
    timestamp: int | float


def normalized_timestamps(series: pd.Series) -> tuple[list[int | float], str]:
    if series.isna().any():
        raise ValueError("Training timestamps must be non-null")
    if pd.api.types.is_datetime64_any_dtype(series):
        values = pd.to_datetime(series, utc=True).dt.as_unit("ns").astype("int64")
        return [int(v) for v in values], "utc_nanoseconds"
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        raise ValueError("Training timestamps must be numeric or datetime values")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("Training timestamps must be finite")
    return [
        v.item() if hasattr(v, "item") else v for v in series
    ], "numeric_source_units"


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class LLMSRecSynScorer(Scorer):
    """One aggregated ICL example followed by one candidate ranking.

    `fit` never fits model weights. `train_event_metadata` is aligned one-to-one
    with X and contains user_id, item_id, event_id and timestamp. The runner
    verifies it against canonical data. Only titles are visible to either model.
    """

    name = "llmsrec_syn"
    score_mapping = "score = n_candidates - rank"
    synthetic_candidate_count = 20

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        embeddings: CachedEmbeddings,
        dataset: str = "ml-1m",
        k: int = 2,
        max_history_items: int = 50,
        aggregate_history_items: int = 50,
        seed: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        extra_body: dict | None = None,
        candidate_group_column: str = "candidate_group",
        title_column: str = "item_title",
    ) -> None:
        self.k = positive_integer(k, "k")
        if self.k > self.synthetic_candidate_count:
            raise ValueError("k cannot exceed 20 synthetic candidates")
        self.max_history_items = positive_integer(
            max_history_items, "max_history_items"
        )
        self.aggregate_history_items = positive_integer(
            aggregate_history_items, "aggregate_history_items"
        )
        self.max_tokens = positive_integer(max_tokens, "max_tokens")
        if dataset not in ("ml-1m", "steam"):
            raise ValueError("dataset must be ml-1m or steam")
        if not isinstance(seed, Integral) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not model or not np.isfinite(temperature) or temperature < 0:
            raise ValueError("Model and finite nonnegative temperature are required")
        self.client, self.model, self.embeddings = client, model, embeddings
        self.dataset, self.seed, self.temperature = dataset, int(seed), temperature
        self.extra_body = deepcopy(extra_body)
        self.candidate_group_column = candidate_group_column
        self.title_column = title_column
        self._users: dict[str, tuple[Event, ...]] | None = None
        self._ids: dict[str, str | int] = {}
        self._times: dict[str, list[int | float]] = {}
        self._catalog: dict[str, tuple[str | int, str]] = {}
        self._retrieved: dict[str, tuple[list[dict[str, Any]], dict[str, int]]] = {}
        self._eligible_counts: dict[str, int] = {}
        self._demonstrations: dict[tuple[str, int], dict[str, Any]] = {}
        self._lock = RLock()
        self._diagnostics: list[dict[str, Any]] = []
        self._attempts: dict[str, int] = {}

    @property
    def prompt_metadata(self) -> dict[str, Any]:
        return {
            "version": PROMPT_VERSION,
            "author_revision": AUTHOR_REVISION,
            "inputs": "titles_only",
            "frozen_llm": True,
            "target_history_order": "oldest_to_newest",
            "aggregate_order": "recent_first_round_robin_decreasing_similarity",
            "output": "complete_candidate_label_ranking",
            "title_normalization": "ml1m_remove_trailing_year_move_The_A; steam_verbatim",
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        train_event_metadata: pd.DataFrame,
        catalog: pd.DataFrame,
    ) -> "LLMSRecSynScorer":
        self._require(X, ["user_id", "item_id", self.title_column])
        self._require(
            train_event_metadata, ["user_id", "item_id", "event_id", "timestamp"]
        )
        self._require(catalog, ["item_id", self.title_column])
        if X.empty or not X.index.is_unique or not X.index.equals(y.index):
            raise ValueError("Train must be nonempty with unique indices aligned to y")
        if not X.index.equals(train_event_metadata.index):
            raise ValueError(
                "Training event metadata must have exactly the train row index/order"
            )
        if not (y == 1).all():
            raise ValueError(
                "LLMSRec-Syn requires observed-only interaction training rows"
            )
        if (
            train_event_metadata.event_id.isna().any()
            or train_event_metadata.event_id.duplicated().any()
        ):
            raise ValueError("Training event identities must be unique and non-null")
        times, self.timestamp_kind = normalized_timestamps(
            train_event_metadata.timestamp
        )
        items = {}
        for item, title in catalog[["item_id", self.title_column]].itertuples(
            index=False, name=None
        ):
            key = identity(item)
            if key in items:
                raise ValueError("Catalog item IDs must be unique")
            items[key] = (scalar(item), self._title(title))
        if len(items) < self.synthetic_candidate_count:
            raise ValueError(
                "LLMSRec-Syn requires a catalog of at least 20 distinct items"
            )
        users: dict[str, list[Event]] = {}
        ids = {}
        input_rows = X[["user_id", "item_id", self.title_column]].itertuples(
            index=False, name=None
        )
        metadata_rows = train_event_metadata[
            ["user_id", "item_id", "event_id"]
        ].itertuples(index=False, name=None)
        for pos, ((user, item, title), (meta_user, meta_item, event_id)) in enumerate(
            zip(input_rows, metadata_rows, strict=True)
        ):
            if identity(user) != identity(meta_user) or identity(item) != identity(
                meta_item
            ):
                raise ValueError(
                    "Training event metadata does not match task user/item rows"
                )
            key, item_key = identity(user), identity(item)
            rendered_title = self._title(title)
            if item_key not in items or items[item_key][1] != rendered_title:
                raise ValueError(
                    "Training titles/items must match the permitted catalog"
                )
            ids[key] = scalar(user)
            users.setdefault(key, []).append(
                Event(scalar(event_id), scalar(item), rendered_title, times[pos])
            )
        with self._lock:
            # Python's sort is stable: equal times keep original event order.
            self._users = {
                key: tuple(sorted(events, key=lambda e: e.timestamp))
                for key, events in users.items()
            }
            self._ids = ids
            self._times = {
                key: [e.timestamp for e in events]
                for key, events in self._users.items()
            }
            self._catalog = dict(sorted(items.items()))
            self._retrieved.clear()
            self._eligible_counts.clear()
            self._demonstrations.clear()
            self._diagnostics.clear()
            self._attempts.clear()
        return self

    def _retrieve(self, target: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
        if self._users is None:
            raise RuntimeError("LLMSRecSynScorer is not fitted")
        if target not in self._users:
            raise ValueError("Target user has no training history")
        if target in self._retrieved:
            return self._retrieved[target]
        cutoff = self._times[target][-1]
        eligible = []
        excluded = {"self": 1, "fewer_than_two_eligible_events": 0}
        for user in sorted(self._users):
            if user == target:
                continue
            end = bisect_right(self._times[user], cutoff)
            if end < 2:
                excluded["fewer_than_two_eligible_events"] += 1
                continue
            sequence = self._users[user][:end]
            prefix = sequence[:-1][-self.max_history_items :]
            eligible.append(
                {
                    "user_id": self._ids[user],
                    "user_key": user,
                    "prefix": [asdict(e) for e in prefix],
                    "next_item": asdict(sequence[-1]),
                    "eligible_event_count": end,
                    "prefix_fingerprint": digest([asdict(e) for e in prefix]),
                }
            )
        self._eligible_counts[target] = len(eligible)
        if not eligible:
            self._retrieved[target] = ([], excluded)
            return [], excluded
        texts = [
            serialize_titles(
                [e.title for e in self._users[target][-self.max_history_items :]]
            )
        ]
        texts += [
            serialize_titles([e["title"] for e in row["prefix"]]) for row in eligible
        ]
        vectors = self.embeddings.encode(texts)
        similarities = vectors[1:] @ vectors[0]
        for row, similarity in zip(eligible, similarities, strict=True):
            row["similarity"] = float(similarity)
        eligible.sort(key=lambda row: (-row["similarity"], row["user_key"]))
        # K cannot exceed 20. Retain only usable neighbors, rather than every
        # user's complete prefix for every evaluated target user.
        self._retrieved[target] = (eligible[: self.synthetic_candidate_count], excluded)
        return self._retrieved[target]

    def demonstration(self, user_id: Any) -> dict[str, Any]:
        """Return an inspectable copy of the exact synthetic demonstration."""
        target = identity(user_id)
        with self._lock:
            cache_key = (target, self.k)
            if cache_key in self._demonstrations:
                return deepcopy(self._demonstrations[cache_key])
            eligible, excluded = self._retrieve(target)
            if len(eligible) < self.k:
                raise ValueError(
                    f"Insufficient eligible demonstration users: need {self.k}, have {len(eligible)}"
                )
            selected = eligible[: self.k]
            aggregate = []
            for depth in range(max(len(row["prefix"]) for row in selected)):
                for row in selected:
                    if depth < len(row["prefix"]):
                        aggregate.append(
                            {"user_id": row["user_id"], **row["prefix"][-1 - depth]}
                        )
                        if len(aggregate) == self.aggregate_history_items:
                            break
                if len(aggregate) == self.aggregate_history_items:
                    break
            positives = list(
                dict.fromkeys(identity(row["next_item"]["item_id"]) for row in selected)
            )
            request_seed = digest(
                {
                    "version": PROMPT_VERSION,
                    "seed": self.seed,
                    "target": target,
                    "cutoff": self._times[target][-1],
                    "selected": [
                        {
                            "user": row["user_key"],
                            "prefix": row["prefix_fingerprint"],
                            "label": row["next_item"],
                        }
                        for row in selected
                    ],
                }
            )
            rng = random.Random(int(request_seed, 16))
            # Unique item identities, not titles. Shared next items keep the first
            # (most similar) user's position; fill the remaining slots to 20.
            filler_keys = rng.sample(
                [key for key in self._catalog if key not in positives],
                self.synthetic_candidate_count - len(positives),
            )
            rng.shuffle(filler_keys)
            ranking = positives + filler_keys
            candidates = ranking.copy()
            rng.shuffle(candidates)
            labels = {key: f"C{i}" for i, key in enumerate(candidates, 1)}
            demo = {
                "target_user_id": self._ids[target],
                "cutoff": self._times[target][-1],
                "timestamp_kind": self.timestamp_kind,
                "k": self.k,
                "eligible_user_count": self._eligible_counts[target],
                "excluded_users": excluded,
                "selected_users": selected,
                "aggregate_history": aggregate,
                "candidates": [
                    {
                        "label": labels[key],
                        "item_id": self._catalog[key][0],
                        "title": self._catalog[key][1],
                    }
                    for key in candidates
                ],
                "answer": [labels[key] for key in ranking],
                "seed_digest": request_seed,
                "prompt_version": PROMPT_VERSION,
            }
            self._demonstrations[cache_key] = demo
            return deepcopy(demo)

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self._users is None:
            raise RuntimeError("LLMSRecSynScorer is not fitted")
        self._require(
            X, ["user_id", "item_id", self.title_column, self.candidate_group_column]
        )
        if not X.index.is_unique or X[self.candidate_group_column].isna().any():
            raise ValueError(
                "Candidate row indices must be unique and group IDs non-null"
            )
        scores = pd.Series(index=X.index, dtype=float, name="score")
        for group_id, group in X.groupby(self.candidate_group_column, sort=False):
            users = {identity(u) for u in group.user_id}
            item_keys = [identity(i) for i in group.item_id]
            if len(users) != 1 or len(set(item_keys)) != len(item_keys):
                raise ValueError(
                    "Each group must contain one user and distinct candidate item IDs"
                )
            target = next(iter(users))
            titles = [self._title(v) for v in group[self.title_column]]
            if any(
                key not in self._catalog or self._catalog[key][1] != title
                for key, title in zip(item_keys, titles, strict=True)
            ):
                raise ValueError(
                    "Candidate titles/items must match the permitted catalog"
                )
            key = digest(
                {
                    "user": target,
                    "group": identity(group_id),
                    "items": item_keys,
                    "k": self.k,
                }
            )
            with self._lock:
                attempt = self._attempts.get(key, 0) + 1
                self._attempts[key] = attempt
            record: dict[str, Any] = {
                "request_key": key,
                "candidate_group": scalar(group_id),
                "user_id": scalar(group.user_id.iloc[0]),
                "k": self.k,
                "attempt": attempt,
                "completed": False,
                "request_issued": False,
                "usage": None,
            }
            started = perf_counter()
            try:
                demo = self.demonstration(group.user_id.iloc[0])
                record["demonstration"] = demo
                history = self._users[target][-self.max_history_items :]
                labels = [f"C{i}" for i in range(1, len(group) + 1)]
                prompt = messages(
                    demo_history=[e["title"] for e in demo["aggregate_history"]],
                    demo_candidates=[e["title"] for e in demo["candidates"]],
                    demo_ranking=demo["answer"],
                    target_history=[e.title for e in history],
                    target_candidates=titles,
                    dataset=self.dataset,
                )
                record.update(
                    {
                        "messages": prompt,
                        "target_history": [asdict(e) for e in history],
                        "label_to_item_id": dict(
                            zip(labels, [scalar(i) for i in group.item_id], strict=True)
                        ),
                        "request_issued": True,
                    }
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **(
                        {"extra_body": deepcopy(self.extra_body)}
                        if self.extra_body
                        else {}
                    ),
                )
                record["usage"] = token_usage(response)
                record["response"] = _chat_completion_text(response)
                ranked = parse_ranked_labels_response(record["response"], labels=labels)
                mapping = {
                    label: len(labels) - rank for rank, label in enumerate(ranked, 1)
                }
                scores.loc[group.index] = [mapping[label] for label in labels]
                record.update({"ranked_labels": ranked, "completed": True})
            except Exception as error:
                record["error"] = repr(error)
                raise
            finally:
                record["elapsed_seconds"] = perf_counter() - started
                with self._lock:
                    self._diagnostics.append(record)
        return scores

    def drain_diagnostics(self) -> list[dict[str, Any]]:
        with self._lock:
            records, self._diagnostics = self._diagnostics, []
        return sorted(records, key=lambda r: (r["request_key"], r["attempt"]))

    def _title(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Every item must have a nonempty string title")
        if self.dataset == "ml-1m":
            # Canonical MovieLens includes release years; the authors' item
            # input separates them and moves trailing English articles forward.
            value = re.sub(r"\s+\(\d{4}\)$", "", value)
            for article in ("The", "A"):
                if value.endswith(f", {article}"):
                    value = f"{article} {value[: -(len(article) + 2)]}"
                    break
        return value

    @staticmethod
    def _require(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
