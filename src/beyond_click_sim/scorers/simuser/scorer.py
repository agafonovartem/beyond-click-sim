"""Independent SimUSER(zero, text-only) watch/skip response pipeline."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import RLock
from time import perf_counter
from uuid import uuid4

import pandas as pd

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.history.common import _chat_completion_text
from beyond_click_sim.scorers.history.llmsrec_syn_retrieval import (
    CachedEmbeddings,
    token_usage,
)

from . import episodic, profiles, prompts
from .config import ADAPTATIONS, EMBEDDING_TEXT_VERSION, SimUSERConfig, TOKENS, VERSION
from .graph import TrainingGraph
from .state import (
    AgentSnapshot,
    digest,
    dumps,
    identity,
    load_checkpoint,
    save_checkpoint,
)


class SimUSERStageError(RuntimeError):
    pass


class SimUSERYesNoScorer(Scorer):
    def __init__(
        self,
        *,
        client=None,
        model="Qwen/Qwen3-8B",
        model_revision=None,
        embedding_client=None,
        embedding_model="text-embedding-3-small",
        embedding_identity="openai:text-embedding-3-small",
        config=None,
        training_events=None,
        catalog=None,
        candidate_group_column="candidate_group",
        cache_dir=None,
        extra_body=None,
        diagnostic_sink=None,
    ):
        self.client, self.model, self.model_revision = client, model, model_revision
        self.embedding_client, self.embedding_model = embedding_client, embedding_model
        self.embedding_identity = embedding_identity
        self.config = config or SimUSERConfig()
        self.training_events, self.input_catalog = training_events, catalog
        self.candidate_group_column = candidate_group_column
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.extra_body = deepcopy(extra_body)
        self.diagnostic_sink = diagnostic_sink
        self._lock = RLock()
        self._session = uuid4().hex
        self._fitted = False
        self.embeddings = None
        self._agents, self._diagnostics = {}, []
        self._counts, self._attempts = Counter(), Counter()
        self._usage = Counter()
        self.initialization_errors = {}
        self._group_cache = {}
        self._source_hashes = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        }

    @property
    def configuration(self):
        return {
            **self.config.to_dict(),
            "model": self.model,
            "model_revision": self.model_revision,
            "embedding_model": self.embedding_model,
            "embedding_identity": self.embedding_identity,
            "extra_body": deepcopy(self.extra_body),
            "variant": "zero_text_only",
            "implementation_sha256": self._source_hashes.copy(),
            "durable_generation_cache": self.model_revision is not None,
            "source": {
                "paper": "Bougie and Watanabe, SimUSER",
                "requested_version": "arXiv:2504.12722v1, 2025-04-17",
                "publication": "10.18653/v1/2025.acl-industry.5",
            },
            "adaptations": list(ADAPTATIONS),
        }

    @property
    def agent_snapshots(self):
        return dict(self._agents)

    def _emit(self, **record):
        with self._lock:
            self._counts[record.get("event", "unknown")] += 1
            if record.get("event") == "llm_call":
                usage = record.get("usage") or {}
                self._usage["calls_with_usage"] += int("total_tokens" in usage)
                self._usage["successful_calls"] += int(record.get("ok", False))
                self._usage["failed_calls"] += int(not record.get("ok", False))
                self._usage["elapsed_seconds"] += record["elapsed_seconds"]
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self._usage[field] += usage.get(field, 0)
            if self.diagnostic_sink is not None:
                self.diagnostic_sink(record)
            else:
                self._diagnostics.append(record)

    def drain_diagnostics(self):
        with self._lock:
            result, self._diagnostics = self._diagnostics, []
        return result

    def fit(self, X, y, *, profile_user_ids=None):
        """Validate permitted training evidence only; performs no inference."""
        self._fitted = False
        if self.input_catalog is None:
            raise ValueError("Provide the permitted static catalog")
        self.catalog = profiles.metadata(self.input_catalog, self.config)
        self.events = profiles.training_records(
            X, y, self.training_events, self.catalog
        )
        profiles.add_training_statistics(self.catalog, self.events)
        population = {}
        for record in self.events:
            population.setdefault(record["user"], []).append(record)
        users = (
            sorted(population)
            if profile_user_ids is None
            else sorted({identity(u) for u in profile_user_ids})
        )
        if not users or any(u not in population for u in users):
            raise ValueError("Every requested user needs observed training history")
        self.histories = {u: population[u] for u in users}
        # An item absent from the target history can only originate in another
        # user's history. Keep the first stable event per item once, rather than
        # rescan every population event separately for every target user.
        representatives = {}
        for event in self.events:
            representatives.setdefault(event["item"], event)
        self.prepared = {
            u: profiles.preparation(
                u, history, representatives.values(), self.catalog, self.config
            )
            for u, history in self.histories.items()
        }
        self.memories = {
            u: episodic.memory_records(history, self.catalog)
            for u, history in self.histories.items()
        }
        self.evidence_fingerprint = digest(
            {"catalog": self.catalog, "events": self.events}
        )
        self._base_key = digest(
            {
                "evidence": self.evidence_fingerprint,
                "config": self.configuration,
                "revision_scope": self.model_revision or self._session,
            }
        )
        self.graph = TrainingGraph(self.catalog, self.events)
        if self.embeddings is not None:
            self.embeddings.close()
        self.embeddings = None
        self._agents.clear()
        self._counts.clear()
        self._usage.clear()
        self._attempts.clear()
        self._group_cache.clear()
        self.initialization_errors.clear()
        self._memory_vectors = {}
        self._fitted = True
        return self

    def _path(self, kind, key):
        return None if self.cache_dir is None else self.cache_dir / kind / f"{key}.json"

    def _request(
        self,
        stage,
        payload,
        validator,
        context,
        *,
        checkpoint=False,
        trace_records=None,
    ):
        messages = prompts.render(stage, payload)
        options = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": TOKENS[stage],
        }
        if self.extra_body is not None:
            options["extra_body"] = deepcopy(self.extra_body)
        if self.config.provider_seed:
            options["seed"] = int(
                digest(
                    {"seed": self.config.seed, "stage": stage, "messages": messages}
                )[:8],
                16,
            ) % (2**31)
        key = digest({"base": self._base_key, "options": options})
        path = self._path("stages", key) if checkpoint else None
        if path is not None:
            cached = load_checkpoint(path, key)
            if cached is not None:
                result = validator(cached["response"])
                self._emit(event="stage_cache_hit", stage=stage, key=key, **context)
                return result
        byte_bound = (
            sum(len(m["content"].encode()) + 32 for m in messages) + TOKENS[stage]
        )
        if (
            self.config.context_tokens is not None
            and byte_bound > self.config.context_tokens
        ):
            raise SimUSERStageError(
                f"{stage}: conservative UTF-8 byte/token context bound {byte_bound} exceeds {self.config.context_tokens}; no evidence truncated"
            )
        record = {
            "event": "llm_call",
            "stage": stage,
            **context,
            "messages": messages,
            "options": {k: v for k, v in options.items() if k != "messages"},
            "prompt_utf8_bytes_plus_response_bound": byte_bound,
            "usage": None,
        }
        started = perf_counter()
        try:
            if self.client is None:
                raise ValueError("No LLM client configured")
            response = self.client.chat.completions.create(**options)
            text = _chat_completion_text(response).strip()
            record["response"] = text
            record["usage"] = token_usage(response)
            finish = getattr(response.choices[0], "finish_reason", None)
            record["finish_reason"] = finish
            if finish in {"length", "content_filter"}:
                raise ValueError(f"Incomplete response: {finish}")
            result = validator(text)
            if path is not None:
                save_checkpoint(path, key, {"response": text})
            record["ok"] = True
            return result
        except Exception as error:
            record.update(ok=False, error=repr(error))
            raise SimUSERStageError(f"{stage}: {error}") from error
        finally:
            record["elapsed_seconds"] = perf_counter() - started
            if trace_records is not None:
                trace_records.append(record)
            self._emit(**record)

    def _initialize_stage(self, stage, payload, validator, user, step=0):
        for attempt in range(1, self.config.initialization_attempts + 1):
            try:
                return self._request(
                    stage,
                    payload,
                    validator,
                    {
                        "user_id": user,
                        "step": step,
                        "attempt": attempt,
                        "phase": "initialization",
                    },
                    checkpoint=True,
                )
            except Exception as error:
                last_error = error
        raise SimUSERStageError(
            f"Initialization {stage} exhausted attempts: {last_error}"
        ) from last_error

    def _items(self, ids, prefix):
        return [
            {"label": f"{prefix}{i + 1}", **self.catalog[key]}
            for i, key in enumerate(ids)
        ]

    def _build_user(self, user):
        prepared = self.prepared[user]
        key = digest({"base": self._base_key, "user": user, "preparation": prepared})
        path = self._path("agents", key)
        cached = load_checkpoint(path, key) if path is not None else None
        if cached is not None:
            self._emit(event="agent_cache_hit", user_id=user, key=key)
            return AgentSnapshot(user, dumps(cached))
        history = prepared["sampled_history"]
        taste = self._initialize_stage(
            "taste", {"history": history}, prompts.taste, user
        )
        personas = self._initialize_stage(
            "personas", {"taste": taste, "history": history}, prompts.personas, user
        )
        totals, matched = [], []
        for p, persona in enumerate(personas):
            total, results = 0, []
            for b, subset in enumerate(prepared["subsets"]):
                # Deliberately no observed ratings, source membership or event IDs.
                items = self._items(subset["items"], "P")
                # Aggregate statistics are not needed for the contrast rating task.
                items = [
                    {k: item[k] for k in ("label", "title", "genres", "summary")}
                    for item in items
                ]
                labels = [r["label"] for r in items]
                values = self._initialize_stage(
                    "ratings",
                    {"persona": persona, "items": items},
                    lambda text: prompts.ratings(text, labels),
                    user,
                    step=p * self.config.contrast_subsets + b,
                )
                total += sum(
                    v if item in subset["target"] else -v
                    for item, v in zip(subset["items"], values, strict=True)
                )
                results.append(values)
            totals.append(total)
            matched.append(results)
        selected = max(range(5), key=lambda i: totals[i])
        payload = {
            "taste": taste,
            "persona": personas[selected],
            "personas": personas,
            "persona_scores": totals,
            "selected_persona_index": selected,
            "contrast_ratings": matched,
            "preparation": prepared,
            "habits": prepared["habits"],
            "memory_ids": [m["id"] for m in self.memories[user]],
            "memory_fingerprint": digest(self.memories[user]),
            "evidence_fingerprint": self.evidence_fingerprint,
            "configuration": self.configuration,
            "cache_key": key,
        }
        if path is not None:
            save_checkpoint(path, key, payload)
        self._emit(
            event="persona_selected",
            user_id=user,
            scores=totals,
            selected=selected,
            subsets=prepared["subsets"],
            sampled_event_ids=prepared["sampled_event_ids"],
        )
        return AgentSnapshot(user, dumps(payload))

    def build_agents(self):
        if not self._fitted:
            raise ValueError("Call fit before build_agents")
        if self.embeddings is None:
            self.embeddings = CachedEmbeddings(
                self.embedding_client,
                self.embedding_model,
                model_identity=self.embedding_identity,
                training_fingerprint=digest(
                    {"version": VERSION, "evidence": self.evidence_fingerprint}
                ),
                cache_path=None
                if self.cache_dir is None
                else self.cache_dir / "embeddings.sqlite",
            )
            # Override this instance's inherited title-sequence label. The
            # shared utility and LLMSRec's namespace/behavior stay unchanged.
            self.embeddings.namespace["serialization"] = EMBEDDING_TEXT_VERSION
        try:
            if self.graph.vectors is None:
                self.graph.vectors = self._initial_embeddings(
                    "graph_embeddings",
                    [
                        dumps(
                            {
                                k: self.catalog[i][k]
                                for k in ("title", "genres", "summary")
                            }
                        )
                        for i in self.graph.items
                    ],
                )
        except Exception as error:
            for user in self.histories:
                self.initialization_errors[user] = repr(error)
            self._emit(
                event="initialization_failed",
                stage="graph_embeddings",
                error=repr(error),
            )
            return self.initialization_metadata()
        for user in self.histories:
            if user in self._agents:
                continue
            try:
                vectors = self._initial_embeddings(
                    "episodic_embeddings",
                    [r["text"] for r in self.memories[user]],
                    user=user,
                )
                state = self._build_user(user)
                self._memory_vectors[user] = vectors
                self._agents[user] = state
                self.initialization_errors.pop(user, None)
            except Exception as error:
                self.initialization_errors[user] = repr(error)
                self._emit(
                    event="initialization_failed", user_id=user, error=repr(error)
                )
        return self.initialization_metadata()

    def _initial_embeddings(self, stage, texts, *, user=None):
        for attempt in range(1, self.config.initialization_attempts + 1):
            try:
                return self.embeddings.encode(texts)
            except Exception as error:
                last_error = error
                self._emit(
                    event="embedding_initialization_retry",
                    stage=stage,
                    user_id=user,
                    attempt=attempt,
                    error=repr(error),
                )
        raise SimUSERStageError(
            f"{stage} exhausted attempts: {last_error}"
        ) from last_error

    def initialization_metadata(self):
        return {
            "requested_users": len(self.histories),
            "initialized_users": len(self._agents),
            "unavailable_users": self.initialization_errors.copy(),
            "effective_evidence_fingerprint": self.evidence_fingerprint,
            "snapshot_fingerprints": {
                u: s.fingerprint for u, s in self._agents.items()
            },
            "diagnostic_counts": dict(self._counts),
            "llm_usage": {
                **dict(self._usage),
                "actual_calls": self._counts["llm_call"],
                "usage_complete": self._counts["llm_call"] > 0
                and self._usage["calls_with_usage"] == self._counts["llm_call"],
                "token_totals_are": "reported usage only; consult usage_complete",
                "elapsed_seconds_are": "sum of request durations, not parallel wall-clock duration",
            },
            "embedding": None
            if self.embeddings is None
            else self.embeddings.statistics(),
        }

    def validate_candidates(self, X):
        required = {"user_id", "item_id", self.candidate_group_column}
        if (
            X.empty
            or not X.index.is_unique
            or not required <= set(X)
            or X[list(required)].isna().any().any()
        ):
            raise ValueError(
                "Candidate rows require unique indices and user/item/group IDs"
            )
        if (
            X.user_id.nunique() != 1
            or X[self.candidate_group_column].nunique() != 1
            or X.item_id.duplicated().any()
        ):
            raise ValueError(
                "Score one user/group of distinct supplied items at a time"
            )
        user = identity(X.user_id.iloc[0])
        items = [identity(i) for i in X.item_id]
        if user not in self.histories or any(i not in self.catalog for i in items):
            raise ValueError(
                "Candidate user/item absent from prepared evidence/catalog"
            )
        return user, items

    def preflight(self, *, candidate_groups=0, max_candidates=20):
        if not self._fitted:
            raise ValueError("Call fit first")
        initialization = len(self.histories) * (2 + 5 * self.config.contrast_subsets)
        # A deliberately conservative byte bound, not a tokenizer measurement.
        # IDs/path descriptions are included; generated text has strict bounds.
        item_bytes = max(len(dumps(v).encode()) for v in self.catalog.values())
        id_bytes = max(len(i.encode()) for i in [*self.catalog, *self.histories])
        event_bytes = max(
            len(dumps(r).encode()) for rows in self.memories.values() for r in rows
        )
        feedback_bytes = max(
            (len(dumps(v).encode()) for v in self.graph.edges.values()), default=2
        )
        k1 = self.config.episodic_k + self.config.delta_k * self.config.max_refinements
        k2 = self.config.graph_k + self.config.delta_k * self.config.max_refinements
        graph_bytes = (
            max_candidates
            * min(k2, len(self.catalog) - 1)
            * (item_bytes + feedback_bytes + 8 * (4 * id_bytes + 200) + 2000)
        )
        decision_bound = (
            18000
            + 3 * k1 * (event_bytes + 200)
            + graph_bytes
            + max_candidates * (item_bytes + 2 * self.config.reason_chars * 4 + 1200)
        )
        init_bound = 20000 + min(
            self.config.history_sample, max(map(len, self.histories.values()))
        ) * (item_bytes + 200)
        bound = max(decision_bound, init_bound) + max(TOKENS.values())
        if (
            self.config.context_tokens is not None
            and bound > self.config.context_tokens
        ):
            raise ValueError(
                f"Conservative context bound {bound} exceeds configured {self.config.context_tokens}; increase serving capacity or explicitly revise configuration"
            )
        return {
            "users": len(self.histories),
            "population_users": len(self.graph.users),
            "training_events": len(self.events),
            "catalog_items": len(self.catalog),
            "candidate_groups": candidate_groups,
            "metadata_coverage": "complete",
            "uncached_successful_calls": {
                "initialization": initialization,
                "groups_min": 3 * candidate_groups,
                "groups_max": (3 + 2 * self.config.max_refinements) * candidate_groups,
                "additional_classification_or_ranking": 0,
            },
            "uncached_embedding_inputs_before_deduplication": {
                "catalog": len(self.catalog),
                "episodic_records": sum(map(len, self.memories.values())),
                "group_queries": 3 * candidate_groups,
                "note": "Excludes retries; embedding requests batch up to 64 inputs and reuse exact inputs.",
            },
            "generation_attempt_upper_bound": initialization
            * self.config.initialization_attempts
            + 5 * (3 + 2 * self.config.max_refinements) * candidate_groups,
            "context": {
                "conservative_utf8_byte_token_bound": bound,
                "configured_tokens": self.config.context_tokens,
                "capacity_verified": self.config.context_tokens is not None,
                "note": "Upper bound, not measured tokens; no model/tokenizer loaded. Exact request bounds also checked.",
            },
            "cache_coverage": "not_assumed; exact fingerprints checked during build/score",
            "evidence_fingerprint": self.evidence_fingerprint,
            "configuration": self.configuration,
        }

    def score(self, X):
        if not self._fitted:
            raise ValueError("Call fit and build_agents first")
        user, item_ids = self.validate_candidates(X)
        if user not in self._agents:
            raise SimUSERStageError(
                f"Initialization unavailable for {user}: {self.initialization_errors.get(user)}"
            )
        snapshot = self._agents[user]
        state = snapshot.to_dict()
        candidates = self._items(item_ids, "C")
        labels = [r["label"] for r in candidates]
        group_key = digest(
            {
                "base": self._base_key,
                "snapshot": snapshot.fingerprint,
                "group": identity(X[self.candidate_group_column].iloc[0]),
                "ordered_items": item_ids,
                "candidates": candidates,
            }
        )
        path = self._path("groups", group_key)
        with self._lock:
            self._attempts[group_key] += 1
            context = {
                "user_id": user,
                "group_key": group_key,
                "attempt": self._attempts[group_key],
                "phase": "score",
            }
            cached = self._group_cache.get(group_key)
        if cached is None and path is not None:
            cached = load_checkpoint(path, group_key)
        if cached is not None:
            parsed = prompts.response(
                dumps({"decisions": cached["decisions"]}),
                labels,
                set(cached["evidence_ids"]),
                reason_chars=self.config.reason_chars,
            )
            self._emit(
                event="group_cache_hit",
                **context,
                stop_reason=cached["stop_reason"],
                cached_inference=cached.get("inference"),
            )
            return pd.Series(
                [float(r["watch"]) for r in parsed], index=X.index, name="score"
            )
        profile = {
            "taste": state["taste"],
            "persona": state["persona"],
            "habits": state["habits"],
            "pickiness_sentence": f"You are {state['habits']['pickiness']} about movies",
        }
        original_query = prompts.TARGET + "\n" + dumps(candidates)
        request_records = []
        questions = self._request(
            "self_ask",
            {"profile": profile, "query": original_query},
            prompts.questions,
            context,
            trace_records=request_records,
        )
        query_vectors = self.embeddings.encode([original_query, *questions])
        episodic_ranks = episodic.ranked_queries(
            self.memories[user], self._memory_vectors[user], query_vectors
        )
        graph_ranks = {
            label: self.graph.ranked(user, item)
            for label, item in zip(labels, item_ids, strict=True)
        }
        evidence, previous_ids = None, None
        decisions, checks = None, None
        retrieval_records = []
        for round_index in range(self.config.max_refinements + 1):
            k1 = self.config.episodic_k + round_index * self.config.delta_k
            k2 = self.config.graph_k + round_index * self.config.delta_k
            evidence = {
                "episodic": episodic.retrieve(episodic_ranks, k1),
                "graph": self.graph.retrieve(user, graph_ranks, k2),
            }
            ids = {r["id"] for kind in evidence.values() for r in kind}
            # A new query-to-record mapping also counts as new graph evidence.
            signature = digest(evidence)
            if round_index and signature == previous_ids:
                stop = "memory_exhausted"
                break
            previous_ids = signature
            payload = {
                "profile": profile,
                "candidates": candidates,
                "evidence": evidence,
                "reason_max_chars": self.config.reason_chars,
            }
            if decisions is not None:
                payload.update(previous_decisions=decisions, previous_check=checks)
            self._emit(
                event="retrieval",
                **context,
                round=round_index,
                k1=k1,
                k2=k2,
                queries=[original_query, *questions],
                evidence=evidence,
            )
            retrieval_records.append(
                {"round": round_index, "k1": k1, "k2": k2, "evidence": evidence}
            )
            decisions = self._request(
                "watch" if round_index == 0 else "revise",
                payload,
                lambda text: prompts.response(
                    text, labels, ids, reason_chars=self.config.reason_chars
                ),
                {**context, "round": round_index},
                trace_records=request_records,
            )
            checks = self._request(
                "check",
                {**payload, "decisions": decisions},
                lambda text: prompts.response(
                    text, labels, ids, check=True, reason_chars=self.config.reason_chars
                ),
                {**context, "round": round_index},
                trace_records=request_records,
            )
            unresolved = any(
                r["contradiction"] or r["insufficient_evidence"] for r in checks
            )
            if not unresolved:
                stop = "consistent"
                break
            stop = "refinement_cap"
        result = {
            "decisions": decisions,
            "checks": checks,
            "stop_reason": stop,
            "unresolved": unresolved,
            "evidence_ids": sorted(ids),
            "snapshot": snapshot.fingerprint,
            "inference": {
                "queries": [original_query, *questions],
                "ordered_items": item_ids,
                "candidates": candidates,
                "retrieval_rounds": retrieval_records,
                "requests": request_records,
            },
        }
        if path is not None:
            save_checkpoint(path, group_key, result)
        with self._lock:
            self._group_cache[group_key] = result
            # Full inference provenance lives on disk. Bound the in-memory
            # cache so thousands of expanded groups do not retain all prompts.
            while len(self._group_cache) > 16:
                self._group_cache.pop(next(iter(self._group_cache)))
        self._emit(event="group_complete", **context, **result)
        return pd.Series(
            [float(r["watch"]) for r in decisions],
            index=X.index,
            dtype=float,
            name="score",
        )

    def close(self):
        if self.embeddings is not None:
            self.embeddings.close()
        if self._fitted:
            self.graph.item_channels.cache_clear()
            self.graph.user_projection.cache_clear()
