"""Frozen DyTA4Rec adaptation: three ranking branches, five calls per group."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
from threading import RLock
import time

import pandas as pd

from ..base import Scorer
from ..recagent.embeddings import CachedEmbeddings, LocalMPNet
from ..recagent.profiles import utc_time
from .config import (
    AUTHOR_REVISION,
    UPSTREAM_REVISION,
    SOURCE_ARTIFACT_SHA256,
    PROMPT_VERSION,
    VERSION,
    DyTA4RecConfig,
)
from .fusion import fuse
from . import memory, profiles, prompts
from .state import Snapshot, digest, read_cache, source_hashes, write_cache


class DyTA4RecScorer(Scorer):
    name = "dyta4rec_offline"

    def __init__(
        self,
        *,
        training_events,
        client=None,
        model="Qwen/Qwen3-8B",
        model_revision=None,
        config=None,
        candidate_group_column="candidate_group",
        sampled_column="is_sampled",
        embedding_provider=None,
        embedding_revision="main",
        extra_body=None,
        serving_metadata=None,
        cache_dir=None,
    ):
        self.config = config or DyTA4RecConfig()
        self.client, self.model, self.model_revision = client, model, model_revision
        self.training_events = training_events.copy(deep=True)
        self.candidate_group_column, self.sampled_column = (
            candidate_group_column,
            sampled_column,
        )
        self.embedding_provider = embedding_provider or LocalMPNet(
            revision=embedding_revision
        )
        self.embedding_revision = embedding_revision
        self.extra_body = (
            deepcopy(extra_body)
            if extra_body is not None
            else {"chat_template_kwargs": {"enable_thinking": False}}
        )
        self.serving_metadata = deepcopy(serving_metadata)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.embeddings = None
        self.agent_snapshots = {}
        self.initialization_errors = {}
        self.diagnostic_sink = None
        self._diagnostics, self._usage = [], []
        self._lock = RLock()
        self._attempts, self._calls, self._cache_hits = Counter(), Counter(), Counter()
        self._sources = source_hashes()
        self._users = None

    @property
    def configuration(self):
        return {
            "version": VERSION,
            "prompt_version": PROMPT_VERSION,
            **self.config.to_dict(),
            "model": self.model,
            "model_revision": self.model_revision,
            "model_revision_verified": self.model_revision is not None,
            "extra_body": deepcopy(self.extra_body),
            "serving": deepcopy(self.serving_metadata),
            "seed_request": "sent_to_provider; backend_determinism_not_guaranteed",
            "embedding_revision_requested": self.embedding_revision,
            "embedding": None if self.embeddings is None else self.embeddings.identity,
            "source_sha256": self._sources,
            "source": {
                "paper_doi": "10.1145/3746252.3760878",
                "author_revision": AUTHOR_REVISION,
                "upstream_revision": UPSTREAM_REVISION,
                "paper_run_correspondence_verified": False,
                "audited_artifact_sha256": SOURCE_ARTIFACT_SHA256,
                "memory_identity_with_dyta_verified": False,
            },
            "context_policy": "utf8_byte_token_upper_bound_plus_512_chat_overhead; reject_without_truncation",
        }

    def _inference_config(self):
        # A complete branch ensemble can also be fused as BC without new inference.
        return {
            k: v
            for k, v in self.configuration.items()
            if k not in {"fusion", "rrf_k", "bc_length_policy"}
        }

    def _emit(self, **record):
        with self._lock:
            if self.diagnostic_sink is not None:
                self.diagnostic_sink(record)
            else:
                self._diagnostics.append(record)

    def drain_diagnostics(self):
        with self._lock:
            records, self._diagnostics = self._diagnostics, []
            return records

    def _path(self, category, identity):
        return (
            None
            if self.cache_dir is None
            else self.cache_dir / category / f"{identity}.json"
        )

    def _bound(self, prompt, stage):
        # Conservative for the declared byte-based Qwen/GPT tokenizers. No model
        # or tokenizer is downloaded and no prompt evidence is silently removed.
        return (
            len((prompts.SYSTEM + prompt).encode("utf-8"))
            + 512
            + self.config.tokens(stage)
        )

    def _call(self, stage, prompt, validator, context):
        bound = self._bound(prompt, stage)
        if bound > self.config.context_tokens:
            raise ValueError(
                f"{stage} conservative token bound {bound} exceeds context {self.config.context_tokens}"
            )
        if self.client is None:
            raise RuntimeError("No response client supplied")
        record = {
            "event": "request",
            **context,
            "stage": stage,
            "prompt": prompt,
            "prompt_hash": digest(prompt),
            "token_upper_bound_including_response": bound,
        }
        start = time.monotonic()
        with self._lock:
            self._calls[stage] += 1
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompts.SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.tokens(stage),
                seed=self.config.seed,
                **({"extra_body": self.extra_body} if self.extra_body else {}),
            )
            usage = getattr(response, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            elif usage is not None and not isinstance(usage, dict):
                usage = {
                    k: getattr(usage, k)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if hasattr(usage, k)
                }
            record["usage"] = usage
            with self._lock:
                self._usage.append(usage)
            choice = response.choices[0]
            record["finish_reason"] = getattr(choice, "finish_reason", None)
            if record["finish_reason"] not in (None, "stop"):
                raise ValueError("Incomplete or refused generation")
            text = choice.message.content
            if not isinstance(text, str):
                raise ValueError("Missing generated text")
            record["response"] = text
            parsed = validator(text)
            record["parsed"] = parsed
            return parsed
        except Exception as error:
            record["error"] = repr(error)
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - start
            self._emit(**record)

    def _initialize_stage(
        self, stage, prompt, validator, *, user, snapshot_key, detail
    ):
        identity = digest(
            {
                "snapshot": snapshot_key,
                "stage": stage,
                "detail": detail,
                "prompt": prompt,
            }
        )
        path = self._path("stages", identity)
        cached = read_cache(path, identity)
        if cached is not None:
            # Validate responses again against the current schema and evidence.
            parsed = validator(cached["response"])
            self._cache_hits["initialization_stage"] += 1
            self._emit(
                event="stage_cache_hit",
                user_id=user,
                stage=stage,
                detail=detail,
                identity=identity,
            )
            return parsed
        last_error = None
        for attempt in range(1, self.config.max_initialization_attempts + 1):
            captured = {}

            def validate(text):
                captured["response"] = text
                return validator(text)

            try:
                result = self._call(
                    stage,
                    prompt,
                    validate,
                    {
                        "user_id": user,
                        "detail": detail,
                        "initialization_attempt": attempt,
                        "snapshot_key": snapshot_key,
                    },
                )
                write_cache(path, identity, captured)
                return result
            except Exception as error:
                last_error = error
        raise RuntimeError(
            f"Initialization stage {stage}/{detail} failed"
        ) from last_error

    def fit(self, X, y, *, profile_user_ids=None):
        self._users, self.population = profiles.prepare_evidence(
            X, y, self.training_events, self.config, sampled_column=self.sampled_column
        )
        requested = (
            list(self._users)
            if profile_user_ids is None
            else list(dict.fromkeys(map(str, profile_user_ids)))
        )
        missing = set(requested) - self._users.keys()
        if missing:
            raise ValueError(
                f"No training history for requested users: {sorted(missing)}"
            )
        self.requested_users = requested
        self.population_hash = digest(self.population)
        self.agent_snapshots.clear()
        self.initialization_errors.clear()
        return self

    def preflight(self, *, candidate_groups=0, candidate_frames=()):
        if self._users is None:
            raise RuntimeError("fit must precede preflight")
        sizes = {
            u: profiles.initialization_size(len(self._users[u]["events"]), self.config)
            for u in self.requested_users
        }
        bounds = {}
        for user in self.requested_users:
            events = self._users[user]["events"]
            blocks = [
                events[i : i + self.config.block_size]
                for i in range(0, len(events), self.config.block_size)
            ]
            bounds[user] = max(
                self._bound(
                    prompts.summary_prompt(
                        "memory_block",
                        {"events": [profiles.event_view(e) for e in block]},
                        max_chars=self.config.text_max_chars,
                    ),
                    "memory_block",
                )
                for block in blocks
            )
        # Future generated text is unknown: report its declared worst-case bound,
        # and check the exact serialized prompt before every request.
        merge_bound = self._bound(
            prompts.summary_prompt(
                "memory_merge",
                {
                    "summaries": [
                        {"ref": f"S{i}", "text": "\x00" * self.config.text_max_chars}
                        for i in range(self.config.merge_fan_in)
                    ]
                },
                max_chars=self.config.text_max_chars,
            ),
            "memory_merge",
        )
        group_bounds = []
        for frame in candidate_frames:
            for group_id, group in frame.groupby(
                self.candidate_group_column, sort=False
            ):
                user = str(group.user_id.iloc[0])
                evidence = self._users[user]
                recent = evidence["events"][-self.config.history_items :]
                observation = prompts.dumps(
                    {
                        "task": "predict_interaction_occurrence",
                        "candidates": profiles.candidate_view(
                            group,
                            self.population,
                            self.config,
                            group_column=self.candidate_group_column,
                        ),
                    }
                )
                sequence = prompts.ranking_prompt(
                    "sequence_ranking",
                    prompts.sequential_inputs(
                        recent, self.config.icl_examples, observation
                    ),
                    flag=True,
                )
                profile_inputs = prompts.dumps(
                    {
                        "recent": [profiles.event_view(e) for e in recent],
                        "statistics": evidence["statistics"],
                        "demographics": evidence["demographics"],
                        "observation": observation,
                    }
                )
                # Generated JSON characters need at most six bytes each. Include
                # generous schema/reference overhead; actual prompts are checked
                # before requests, so this pessimistic estimate never truncates.
                generated_chars = (
                    3 + min(self.config.retrieval_k, sizes[user]["summary_records"])
                ) * self.config.text_max_chars
                generated_chars += (
                    len(prompts.PROFILE_FIELDS) * self.config.profile_field_max_chars
                )
                group_bounds.append(
                    {
                        "candidate_group": str(group_id),
                        "user_id": user,
                        "sequence_token_upper_bound": self._bound(
                            sequence, "sequence_ranking"
                        ),
                        "profile_generated_text_upper_bound": 2
                        * len(profile_inputs.encode("utf-8"))
                        + 6 * generated_chars
                        + 8192
                        + self.config.ranking_max_tokens,
                    }
                )
        return {
            "configuration": self.configuration,
            "users": len(sizes),
            "per_user": sizes,
            "training_events": sum(len(self._users[u]["events"]) for u in sizes),
            "population_training_events": sum(
                v["training_interaction_count"] for v in self.population.values()
            ),
            "temporal_window": self.config.history_items,
            "icl_examples": self.config.icl_examples,
            "candidate_groups": candidate_groups,
            "uncached_initialization_llm_calls": sum(
                s["llm_calls"] for s in sizes.values()
            ),
            "uncached_group_llm_calls": 5 * candidate_groups,
            "summary_embeddings": sum(s["summary_records"] for s in sizes.values()),
            "query_embeddings": candidate_groups,
            "block_token_upper_bounds": bounds,
            "merge_worst_case_token_bound": merge_bound,
            "group_prompt_token_bounds": group_bounds,
            "generated_bound_policy": "six_JSON_bytes_per_generated_character; pessimistic_estimate_not_actual_length",
            "snapshot_resolution": "requires_local_embedding_revision; no models_loaded_by_preflight",
            "required_users_without_in_memory_snapshot": [
                u for u in sizes if u not in self.agent_snapshots
            ],
            "context_check": "exact_prompt_bound_before_each_call; overflow_is_failure; no_truncation",
        }

    def build_agents(self):
        if self._users is None:
            raise RuntimeError("fit must precede build_agents")
        try:
            if self.embeddings is None:
                self.embeddings = CachedEmbeddings(
                    self.embedding_provider,
                    None if self.cache_dir is None else self.cache_dir / "embeddings",
                )
        except Exception as error:
            for user in self.requested_users:
                self.initialization_errors[user] = repr(error)
            self._emit(event="embedding_initialization_failed", error=repr(error))
            return self
        for user in self.requested_users:
            try:
                self.agent_snapshots[user] = self._build_user(user)
                self.initialization_errors.pop(user, None)
            except Exception as error:
                self.initialization_errors[user] = repr(error)
                self._emit(
                    event="initialization_failed", user_id=user, error=repr(error)
                )
        return self

    def _build_user(self, user):
        evidence = self._users[user]
        key = digest(
            {
                "user": user,
                "evidence": evidence,
                "population": self.population_hash,
                "configuration": self._inference_config(),
            }
        )
        path = self._path("snapshots", key)
        cached = read_cache(path, key)
        if cached is not None:
            self._cache_hits["snapshot"] += 1
            self._emit(event="snapshot_cache_hit", user_id=user, snapshot_key=key)
            return Snapshot.create(cached)
        records = []

        def request(stage, prompt, validator, detail):
            return self._initialize_stage(
                stage, prompt, validator, user=user, snapshot_key=key, detail=detail
            )

        def summarize(stage, inputs, support, allowed):
            record_id = f"S{len(records) + 1}"
            prompt = prompts.summary_prompt(
                stage, inputs, max_chars=self.config.text_max_chars
            )
            summary = request(
                stage,
                prompt,
                lambda t: prompts.parse_summary(t, allowed, self.config.text_max_chars),
                record_id,
            )

            def importance_request(stage, text, validator):
                return request(
                    stage, prompts.render(stage, text, {}), validator, record_id
                )

            importance = memory.adapter(
                [],
                config=self.config,
                embeddings=self.embeddings,
                request=importance_request,
            ).importance(summary["text"])
            now = max(e["timestamp"] for e in support)
            record = {
                "id": record_id,
                **summary,
                "source_event_refs": [e["ref"] for e in support],
                "source_event_ids": [e["event_id"] for e in support],
                "importance": importance,
                "created_at": now,
                "accessed_at": now,
                "embedding": self.embeddings.embed(summary["text"]),
                "stage": stage,
            }
            records.append(record)
            return record

        events = evidence["events"]
        by_ref = {e["ref"]: e for e in events}
        level = [
            summarize(
                "memory_block",
                {"events": [profiles.event_view(e) for e in block]},
                block,
                [e["ref"] for e in block],
            )
            for start in range(0, len(events), self.config.block_size)
            for block in [events[start : start + self.config.block_size]]
        ]
        while len(level) > 1:
            level = [
                summarize(
                    "memory_merge",
                    {"summaries": [{"ref": r["id"], "text": r["text"]} for r in chunk]},
                    [by_ref[ref] for r in chunk for ref in r["source_event_refs"]],
                    [r["id"] for r in chunk],
                )
                for start in range(0, len(level), self.config.merge_fan_in)
                for chunk in [level[start : start + self.config.merge_fan_in]]
            ]
        root = {"ref": level[0]["id"], "text": level[0]["text"]}
        profile = request(
            "long_profile",
            prompts.profile_prompt(
                root, evidence["statistics"], evidence["demographics"], self.config
            ),
            lambda t: prompts.parse_profile(
                t, [root["ref"]], self.config.profile_field_max_chars
            ),
            "profile",
        )
        recent = events[-self.config.history_items :]
        short = {
            stage: request(
                stage,
                prompts.recent_prompt(stage, recent, self.config),
                lambda t: prompts.parse_summary(
                    t, [e["ref"] for e in recent], self.config.text_max_chars
                ),
                stage,
            )
            for stage in ("recent_behavior", "recent_interests")
        }
        data = {
            "snapshot_key": key,
            "user_id": user,
            "configuration": self._inference_config(),
            "evidence": evidence,
            "population_hash": self.population_hash,
            "recent": recent,
            "records": records,
            "root_id": root["ref"],
            "long_profile": profile,
            **short,
            "reference_time": evidence["cutoff"] + 1,
            "completed": True,
        }
        write_cache(path, key, data)
        snapshot = Snapshot.create(data)
        self._emit(
            event="snapshot_completed",
            user_id=user,
            snapshot_key=key,
            identity=snapshot.identity,
            summary_records=len(records),
        )
        return snapshot

    def validate_candidates(self, X):
        profiles.candidate_view(
            X, self.population, self.config, group_column=self.candidate_group_column
        )

    def score(self, X):
        if self._users is None:
            raise RuntimeError("DyTA4Rec is not fitted")
        self.validate_candidates(X)
        result = pd.Series(index=X.index, dtype=float, name="score")
        for group_id, group in X.groupby(self.candidate_group_column, sort=False):
            result.loc[group.index] = self._score_group(str(group_id), group)
        return result

    def _score_group(self, group_id, group):
        user = str(group.user_id.iloc[0])
        if user not in self.agent_snapshots:
            raise RuntimeError(
                f"Missing initialized snapshot for {user}: {self.initialization_errors.get(user, 'build_agents required')}"
            )
        snapshot = self.agent_snapshots[user]
        state = snapshot.to_dict()
        candidates = profiles.candidate_view(
            group,
            self.population,
            self.config,
            group_column=self.candidate_group_column,
        )
        labels = [c["label"] for c in candidates]
        observation = prompts.dumps(
            {"task": "predict_interaction_occurrence", "candidates": candidates}
        )
        key = digest(
            {
                "snapshot": snapshot.identity,
                "ordered_candidate_ids": group.item_id.map(str).tolist(),
                "observation": observation,
                "configuration": self._inference_config(),
            }
        )
        path = self._path("groups", key)
        context = {
            "user_id": user,
            "candidate_group": group_id,
            "snapshot": snapshot.identity,
            "group_key": key,
        }
        with self._lock:
            self._attempts[key] += 1
            context["group_attempt"] = self._attempts[key]
        cached = read_cache(path, key)
        cache_hit = cached is not None
        if cached is None:
            recent = state["recent"]

            def call(stage, prompt, validator):
                return self._call(stage, prompt, validator, context)

            try:
                cluster = call(
                    "cluster_analysis",
                    prompts.analysis_prompt("cluster_analysis", recent, self.config),
                    lambda t: prompts.parse_analysis(t, self.config.text_max_chars),
                )
                cluster_ranking = call(
                    "cluster_ranking",
                    prompts.ranking_prompt(
                        "cluster_ranking",
                        {
                            "analysis": cluster["analysis"],
                            "candidate_observation": observation,
                        },
                    ),
                    lambda t: prompts.parse_ranking(t, labels),
                )
                sequence = call(
                    "sequence_ranking",
                    prompts.ranking_prompt(
                        "sequence_ranking",
                        prompts.sequential_inputs(
                            recent, self.config.icl_examples, observation
                        ),
                        flag=True,
                    ),
                    lambda t: prompts.parse_ranking(t, labels, flag=True),
                )
                profile_analysis = call(
                    "profile_analysis",
                    prompts.analysis_prompt("profile_analysis", recent, self.config),
                    lambda t: prompts.parse_analysis(t, self.config.text_max_chars),
                )
                retrieved, terms = memory.retrieve(
                    state["records"],
                    observation,
                    now=state["reference_time"],
                    config=self.config,
                    embeddings=self.embeddings,
                )
                profile_ranking = call(
                    "profile_ranking",
                    prompts.ranking_prompt(
                        "profile_ranking",
                        {
                            "long_term_profile": state["long_profile"],
                            "observed_statistics": state["evidence"]["statistics"],
                            "recorded_demographics": state["evidence"]["demographics"],
                            "recent_behavior": state["recent_behavior"],
                            "recent_interests": state["recent_interests"],
                            "recent_history_and_short_term_memory": [
                                profiles.event_view(e) for e in recent
                            ],
                            "retrieved_long_term_memory": retrieved,
                            "temporal_analysis": profile_analysis["analysis"],
                            "reference_time_utc": utc_time(state["reference_time"]),
                            "candidate_observation": observation,
                        },
                    ),
                    lambda t: prompts.parse_ranking(t, labels),
                )
                cached = {
                    "branches": {
                        "sequence": sequence["ranking"],
                        "profile": profile_ranking["ranking"],
                        "cluster": cluster_ranking["ranking"],
                    },
                    "cluster_flag": cluster["pattern_flag"],
                    "sequence_flag": sequence["pattern_flag"],
                    "cluster_analysis": cluster,
                    "profile_analysis": profile_analysis,
                    "retrieval": terms,
                    "query": observation,
                    "query_hash": digest(observation),
                }
                write_cache(path, key, cached)
            except Exception as error:
                self._emit(event="group_failed", **context, error=repr(error))
                raise
        else:
            with self._lock:
                self._cache_hits["group"] += 1
        fused = fuse(
            cached["branches"],
            labels=labels,
            cluster_flag=cached["cluster_flag"],
            sequence_flag=cached["sequence_flag"],
            strategy=self.config.fusion,
            rrf_k=self.config.rrf_k,
        )
        self._emit(
            event="group_completed",
            **context,
            **cached,
            fusion=fused,
            cache_hit=cache_hit,
        )
        return fused["scores"]

    def initialization_metadata(self):
        return {
            "requested_users": len(self.requested_users),
            "completed_users": len(self.agent_snapshots),
            "errors": dict(self.initialization_errors),
            "snapshot_identities": {
                u: s.identity for u, s in self.agent_snapshots.items()
            },
            "requests_by_stage": dict(self._calls),
            "cache_hits": dict(self._cache_hits),
            "reported_token_totals": {
                key: sum((u or {}).get(key, 0) or 0 for u in self._usage)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "usage_complete": len(self._usage) == sum(self._calls.values())
            and all(
                u
                and all(
                    u.get(k) is not None
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
                for u in self._usage
            ),
            "embedding": None if self.embeddings is None else self.embeddings.stats(),
        }
