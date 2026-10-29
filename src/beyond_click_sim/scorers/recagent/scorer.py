"""Offline, independently reset RecAgent profile–memory–action scorer."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
from threading import RLock
from time import perf_counter
from typing import Callable, Self
from uuid import uuid4

import pandas as pd

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.history.common import _chat_completion_text
from beyond_click_sim.scorers.history.yes_no import parse_yes_no_response

from . import prompts
from .embeddings import CachedEmbeddings, EmbeddingProvider, LocalMPNet, vector
from .memory import RecAgentMemory
from .profiles import (
    ALLOWED_CANDIDATE_COLUMNS,
    ALLOWED_HISTORY_COLUMNS,
    CANDIDATE_COLUMNS,
    HISTORY_COLUMNS,
    demographics,
    describe,
    unix_time,
    utc_time,
)
from .state import (
    AUTHOR_REVISION,
    PORT_VERSION,
    AgentSnapshot,
    MemoryConfig,
    atomic_json,
    digest,
)


class RecAgentStageError(RuntimeError):
    pass


def bounded_summary(text: str) -> str:
    if len(text.split()) > 120:
        raise ValueError("Profile summary exceeds the declared 120-word bound")
    return text


class RecAgentYesNoScorer(Scorer):
    def __init__(
        self,
        *,
        client=None,
        model: str = "Qwen/Qwen3-8B",
        model_revision: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_revision: str = "main",
        memory_config: MemoryConfig | None = None,
        training_events: pd.DataFrame | None = None,
        candidate_group_column: str = "candidate_group",
        history_description_columns=HISTORY_COLUMNS,
        candidate_description_columns=CANDIDATE_COLUMNS,
        use_demographics: bool = True,
        max_history_items: int | None = None,
        profile_block_size: int = 20,
        seed: int = 2023,
        temperature: float = 0.0,
        max_tokens: int = 256,
        importance_max_tokens: int = 16,
        extra_body: dict | None = None,
        max_initialization_attempts: int = 5,
        cache_dir: Path | None = None,
        diagnostic_sink: Callable[[dict], None] | None = None,
    ):
        self.client, self.model, self.model_revision = client, model, model_revision
        self.provider = embedding_provider or LocalMPNet(revision=embedding_revision)
        self.memory_config = memory_config or MemoryConfig()
        self.training_events = training_events
        self.candidate_group_column = candidate_group_column
        self.history_columns = tuple(history_description_columns)
        self.candidate_columns = tuple(candidate_description_columns)
        if (
            not set(self.history_columns) <= ALLOWED_HISTORY_COLUMNS
            or "item_title" not in self.history_columns
        ):
            raise ValueError(
                "History columns must be whitelisted movie metadata and training ratings"
            )
        if (
            not set(self.candidate_columns) <= ALLOWED_CANDIDATE_COLUMNS
            or "item_title" not in self.candidate_columns
        ):
            raise ValueError("Candidate columns must be whitelisted item metadata")
        for name, value in (
            ("profile_block_size", profile_block_size),
            ("max_tokens", max_tokens),
            ("importance_max_tokens", importance_max_tokens),
            ("max_initialization_attempts", max_initialization_attempts),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if profile_block_size > 20:
            raise ValueError("Profile blocks must contain at most 20 events")
        if max_history_items is not None and (
            not isinstance(max_history_items, int) or max_history_items < 1
        ):
            raise ValueError("max_history_items must be positive or None")
        self.use_demographics, self.max_history_items = (
            use_demographics,
            max_history_items,
        )
        self.profile_block_size, self.seed = profile_block_size, seed
        self.temperature, self.max_tokens = temperature, max_tokens
        self.importance_max_tokens, self.extra_body = (
            importance_max_tokens,
            deepcopy(extra_body),
        )
        self.max_initialization_attempts = max_initialization_attempts
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.diagnostic_sink = diagnostic_sink
        self._lock = RLock()
        self._diagnostics = []
        self._counts = Counter()
        self._group_attempts = Counter()
        self._agents: dict[str, AgentSnapshot] = {}
        self._histories = {}
        self._profiles = {}
        self._clocks = {}
        self._cache_keys = {}
        self.initialization_errors = {}
        self.cache_hits = 0
        self.cache_resumes = 0
        self.embeddings = None
        self._fitted = False
        # An unreported server checkpoint cannot safely reuse another process's agents.
        self._unversioned_session = uuid4().hex

    @property
    def configuration(self) -> dict:
        return {
            "port_version": PORT_VERSION,
            "author_revision": AUTHOR_REVISION,
            "prompt_version": prompts.PROMPT_VERSION,
            "source_sha256": {
                p.name: sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "model": self.model,
            "model_revision": self.model_revision,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "importance_max_tokens": self.importance_max_tokens,
            "extra_body": self.extra_body,
            "history_columns": list(self.history_columns),
            "candidate_columns": list(self.candidate_columns),
            "use_demographics": self.use_demographics,
            "max_history_items": self.max_history_items,
            "profile_block_size": self.profile_block_size,
            "summary_max_words": 120,
            "recent_history_items": 10,
            "seed": self.seed,
            "memory": asdict(self.memory_config),
            "clock": "last_training_event_plus_one_second",
            "durable_cache_reuse": self.model_revision is not None,
        }

    @property
    def agent_snapshots(self) -> dict[str, AgentSnapshot]:
        return deepcopy(self._agents)

    def _emit(self, **record):
        with self._lock:
            self._counts[record.get("stage", record.get("event", "unknown"))] += 1
            if self.diagnostic_sink is not None:
                self.diagnostic_sink(record)
            else:
                self._diagnostics.append(record)

    def drain_diagnostics(self) -> list[dict]:
        with self._lock:
            records, self._diagnostics = self._diagnostics, []
        return records

    def _request(self, context: dict):
        def request(stage: str, prompt: str, validator=None):
            if self.client is None:
                raise RecAgentStageError("No LLM client configured")
            messages = [
                {"role": "system", "content": prompts.SYSTEM},
                {
                    "role": "user",
                    "content": f"[{prompts.PROMPT_VERSION}:{stage}]\n{prompt}",
                },
            ]
            options = dict(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.importance_max_tokens
                if stage == "importance"
                else self.max_tokens,
            )
            if self.extra_body is not None:
                options["extra_body"] = deepcopy(self.extra_body)
            started = perf_counter()
            record = {
                **context,
                "stage": stage,
                "messages": messages,
                "model": self.model,
            }
            try:
                response = self.client.chat.completions.create(**options)
                text = _chat_completion_text(response).strip()
                record["response"] = text
                usage = getattr(response, "usage", None)
                record["usage"] = (
                    usage.model_dump()
                    if hasattr(usage, "model_dump")
                    else (usage if isinstance(usage, dict) else None)
                )
                reason = getattr(response.choices[0], "finish_reason", None)
                record["finish_reason"] = reason
                if reason in {"length", "content_filter"}:
                    raise ValueError(f"Incomplete LLM output: {reason}")
                if not text:
                    raise ValueError("Empty LLM output")
                result = text if validator is None else validator(text)
                record["ok"] = True
                return result
            except Exception as error:
                record.update(ok=False, error=repr(error))
                raise RecAgentStageError(f"{stage}: {error}") from error
            finally:
                record["elapsed_seconds"] = perf_counter() - started
                self._emit(**record)

        return request

    def fit(self, X: pd.DataFrame, y: pd.Series, *, profile_user_ids=None) -> Self:
        """Validate and serialize training evidence; no inference or model loading."""
        self._fitted = False
        if not X.index.is_unique or not X.index.equals(y.index) or X.empty:
            raise ValueError(
                "Training rows and labels need matching, unique, nonempty indices"
            )
        if y.isna().any() or not y.eq(1).all():
            raise ValueError(
                "RecAgent initialization requires observed training interactions only"
            )
        required = {"user_id", "item_id", *self.history_columns}
        if not required <= set(X.columns) or X[list(required)].isna().any().any():
            raise ValueError("Missing training movie metadata")
        events = self.training_events
        if events is None or not events.index.equals(X.index):
            raise ValueError("Provide verified, index-aligned training_events")
        if not {"user_id", "item_id", "event_id", "timestamp"} <= set(events.columns):
            raise ValueError(
                "Training events require user/item/event identities and timestamps"
            )
        if events.event_id.isna().any() or events.event_id.duplicated().any():
            raise ValueError("Training event identities must be unique and non-null")
        if not events[["user_id", "item_id"]].equals(X[["user_id", "item_id"]]):
            raise ValueError("Training event identities do not match X")
        frame = X.copy(deep=True)
        frame["_recagent_time"] = [unix_time(v) for v in events.timestamp]
        frame["_recagent_event"] = [str(v) for v in events.event_id]
        users = (
            list(X.user_id.drop_duplicates())
            if profile_user_ids is None
            else list(profile_user_ids)
        )
        if len({str(u) for u in X.user_id.unique()}) != X.user_id.nunique():
            raise ValueError("User IDs must remain unique when serialized")
        self._agents.clear()
        self._histories.clear()
        self._profiles.clear()
        self._clocks.clear()
        self._cache_keys.clear()
        self.initialization_errors.clear()
        self._group_attempts.clear()
        self.cache_hits = self.cache_resumes = 0
        for user in dict.fromkeys(users):
            rows = frame.loc[frame.user_id.eq(user)].sort_values(
                "_recagent_time", kind="stable"
            )
            if rows.empty:
                raise ValueError(f"No observed training events for user {user}")
            key = str(user)
            self._profiles[key] = demographics(rows, enabled=self.use_demographics)
            self._clocks[key] = float(rows._recagent_time.iloc[-1]) + 1.0
            if self.max_history_items is not None:
                rows = rows.tail(self.max_history_items)
            self._histories[key] = [
                {
                    "event_id": row["_recagent_event"],
                    "item_id": str(row["item_id"]),
                    "timestamp": float(row["_recagent_time"]),
                    "title": str(row["item_title"]),
                    "observation": f"Movie user has a recorded interaction at {utc_time(row['_recagent_time'])}: "
                    + describe(row, self.history_columns),
                }
                for _, row in rows.iterrows()
            ]
        self._fitted = True
        return self

    def preflight(self, *, candidate_groups: int = 0) -> dict:
        if not self._fitted:
            raise ValueError("Call fit first")
        events = sum(map(len, self._histories.values()))
        blocks = sum(
            (len(h) + self.profile_block_size - 1) // self.profile_block_size
            for h in self._histories.values()
        )
        return {
            "users": len(self._histories),
            "training_events": events,
            "profile_blocks": blocks,
            "candidate_groups": candidate_groups,
            "base_llm_calls_without_promotions_reflections_retries": blocks
            + 2 * events
            + 2 * candidate_groups,
            "cache_complete_users_in_this_instance": len(self._agents),
            "cache_coverage": "unverified_until_embedding_revision_is_resolved",
            "existing_checkpoint_files": 0
            if self.cache_dir is None
            else len(list(self.cache_dir.glob("agents/*.json"))),
            "configuration": self.configuration,
        }

    def _key(self, user: str) -> str:
        return digest(
            {
                "config": self.configuration,
                "embedding": self.embeddings.identity,
                "revision_scope": self.model_revision or self._unversioned_session,
                "user": user,
                "history": self._histories[user],
                "demographics": self._profiles[user],
                "decision_time": self._clocks[user],
            }
        )

    def _checkpoint(self, snapshot: AgentSnapshot, key: str):
        if self.cache_dir is not None:
            payload = snapshot.to_dict()
            atomic_json(
                self.cache_dir / "agents" / f"{key}.json",
                {"key": key, "payload": payload, "sha256": digest(payload)},
            )

    def _load(self, user: str, key: str) -> AgentSnapshot | None:
        path = (
            None
            if self.cache_dir is None
            else self.cache_dir / "agents" / f"{key}.json"
        )
        if path is None or not path.exists():
            return None
        record = json.loads(path.read_text())
        if record.get("key") != key or record.get("sha256") != digest(
            record["payload"]
        ):
            raise ValueError("RecAgent checkpoint identity/checksum mismatch")
        snapshot = AgentSnapshot.from_dict(record["payload"])
        count = len(self._histories[user])
        blocks = (count + self.profile_block_size - 1) // self.profile_block_size
        if (
            snapshot.user_id != user
            or snapshot.demographics != self._profiles[user]
            or not 0 <= snapshot.replay_events <= count
            or not 0 <= snapshot.profile_blocks <= blocks
            or (snapshot.replay_events and snapshot.profile_blocks != blocks)
            or (
                snapshot.completed
                and (
                    snapshot.profile_blocks != blocks or snapshot.replay_events != count
                )
            )
        ):
            raise ValueError("Invalid RecAgent checkpoint progress")
        snapshot.rng()  # Validate serialized random state without executing code.
        for memory in (*snapshot.memory.short_term, *snapshot.memory.long_term):
            vector(memory.embedding, self.embeddings.identity["dimensions"])
        self.cache_hits += int(snapshot.completed)
        self.cache_resumes += int(not snapshot.completed)
        self._emit(
            event="agent_cache_hit",
            user_id=user,
            complete=snapshot.completed,
            cache_key=key,
        )
        return snapshot

    def _memory(self, snapshot: AgentSnapshot, request, rng, context) -> RecAgentMemory:
        return RecAgentMemory(
            snapshot.memory,
            config=self.memory_config,
            embeddings=self.embeddings,
            request=request,
            rng=rng,
            emit=lambda **data: self._emit(**context, **data),
        )

    def _transaction(
        self, base: AgentSnapshot, key: str, phase: str, index: int, operation
    ):
        last_error = None
        for attempt in range(1, self.max_initialization_attempts + 1):
            state = deepcopy(base)
            context = {
                "user_id": state.user_id,
                "phase": phase,
                "step": index,
                "attempt": attempt,
            }
            rng = state.rng()
            try:
                operation(state, self._request(context), rng, context)
                state.rng_state = rng.getstate()
                self._checkpoint(state, key)
                return state
            except Exception as error:
                last_error = error
                self._emit(**context, event="initialization_retry", error=repr(error))
        raise RecAgentStageError(
            f"{phase} step {index} exhausted retries: {last_error}"
        ) from last_error

    def build_agents(self) -> dict:
        """Resume atomic profile blocks and replay events; failed users remain unavailable."""
        if not self._fitted:
            raise ValueError("Call fit before build_agents")
        if self.embeddings is None:
            self.embeddings = CachedEmbeddings(
                self.provider,
                None if self.cache_dir is None else self.cache_dir / "embeddings",
            )
        for user, history in self._histories.items():
            if user in self._agents:
                continue
            key = self._key(user)
            self._cache_keys[user] = key
            try:
                state = self._load(user, key)
                if state is None:
                    rng = random.Random(
                        int(digest({"seed": self.seed, "user": user}), 16)
                    )
                    state = AgentSnapshot(
                        user,
                        self._profiles[user].copy(),
                        clock=history[0]["timestamp"],
                        rng_state=rng.getstate(),
                    )
                blocks = [
                    history[i : i + self.profile_block_size]
                    for i in range(0, len(history), self.profile_block_size)
                ]
                for index in range(state.profile_blocks, len(blocks)):

                    def update_profile(s, request, rng, context):
                        s.interests = request(
                            "profile_update",
                            prompts.PROFILE_UPDATE.format(
                                demographics=json.dumps(
                                    s.demographics, ensure_ascii=False
                                ),
                                interests=s.interests,
                                events="\n".join(
                                    e["observation"] for e in blocks[index]
                                ),
                            ),
                            validator=bounded_summary,
                        )
                        s.profile_blocks = index + 1

                    state = self._transaction(
                        state, key, "profile", index, update_profile
                    )
                for index in range(state.replay_events, len(history)):

                    def replay(s, request, rng, context):
                        event = history[index]
                        s.clock = event["timestamp"]
                        self._memory(s, request, rng, context).observe(
                            event["observation"], now=s.clock
                        )
                        s.recent = (s.recent + [event["title"]])[-10:]
                        s.replay_events = index + 1

                    state = self._transaction(state, key, "replay", index, replay)
                state.clock, state.completed = self._clocks[user], True
                self._checkpoint(state, key)
                self._agents[user] = state
                self.initialization_errors.pop(user, None)
            except Exception as error:
                self.initialization_errors[user] = repr(error)
                self._emit(
                    event="initialization_failed",
                    user_id=user,
                    error=repr(error),
                    cache_key=key,
                )
        return self.initialization_metadata()

    def initialization_metadata(self) -> dict:
        return {
            "requested_users": len(self._histories),
            "initialized_users": len(self._agents),
            "unavailable_users": self.initialization_errors.copy(),
            "cache_hits": self.cache_hits,
            "cache_resumes": self.cache_resumes,
            "cache_keys": self._cache_keys.copy(),
            "embedding": None if self.embeddings is None else self.embeddings.stats(),
            "diagnostic_counts": dict(self._counts),
        }

    def score(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted or self.embeddings is None:
            raise ValueError("Call fit and build_agents before scoring")
        required = {
            "user_id",
            "item_id",
            self.candidate_group_column,
            *self.candidate_columns,
        }
        if (
            X.empty
            or not X.index.is_unique
            or not required <= set(X.columns)
            or X[list(required)].isna().any().any()
        ):
            raise ValueError(
                "Invalid candidate group or missing visible candidate metadata"
            )
        if (
            X.user_id.nunique() != 1
            or X[self.candidate_group_column].nunique() != 1
            or X.item_id.duplicated().any()
        ):
            raise ValueError(
                "Score exactly one user/group with distinct candidates at a time"
            )
        user = str(X.user_id.iloc[0])
        if user not in self._agents:
            raise RecAgentStageError(
                f"User initialization unavailable: {self.initialization_errors.get(user, user)}"
            )
        group_id = str(X[self.candidate_group_column].iloc[0])
        labels = [f"C{i}" for i in range(1, len(X) + 1)]
        candidates = "\n".join(
            f"{label}: {describe(row, self.candidate_columns)}"
            for label, (_, row) in zip(labels, X.iterrows(), strict=True)
        )
        state = deepcopy(self._agents[user])
        group_key = digest(
            {
                "seed": self.seed,
                "state": digest(state.to_dict()),
                "group": group_id,
                "items": [str(v) for v in X.item_id],
                "candidates": candidates,
            }
        )
        rng = random.Random(int(group_key, 16))
        with self._lock:
            self._group_attempts[group_key] += 1
            attempt = self._group_attempts[group_key]
        context = {
            "user_id": user,
            "candidate_group": group_id,
            "phase": "score",
            "attempt": attempt,
        }
        request = self._request(context)
        memory = self._memory(state, request, rng, context)
        # The inherited source chain loads memory for both summary and action.
        memory.read(candidates, now=state.clock)
        profile = json.dumps(
            {**state.demographics, "interests": state.interests}, ensure_ascii=False
        )
        summary = request(
            "profile_summary",
            prompts.PROFILE_SUMMARY.format(profile=profile),
            validator=bounded_summary,
        )
        recalled = memory.read(candidates, now=state.clock)
        parsed = request(
            "action",
            prompts.ACTION.format(
                summary=summary,
                demographics=json.dumps(state.demographics, ensure_ascii=False),
                memories=recalled,
                recent=json.dumps(state.recent, ensure_ascii=False),
                time=utc_time(state.clock),
                candidates=candidates,
            ),
            validator=lambda s: parse_yes_no_response(s, labels=labels),
        )
        return pd.Series(
            [parsed[label] for label in labels],
            index=X.index,
            dtype=float,
            name="score",
        )
