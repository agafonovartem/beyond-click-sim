"""RecAgent memory port, using the audited author-code defaults.

All times are explicit Unix seconds. The caller owns transactions and RNG state.
This module never consults a wall clock or a process-global random generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Callable

import numpy as np

from . import prompts
from .embeddings import CachedEmbeddings
from .state import MemoryConfig, MemoryRecord, MemoryState, Support


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or denominator == 0
    ):
        raise ValueError("Invalid vectors for cosine similarity")
    return float(np.clip(np.dot(a, b) / denominator, -1, 1))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def parse_importance(text: str) -> int:
    if re.fullmatch(r"(?:[1-9]|10)", text.strip()) is None:
        raise ValueError("Importance must be one integer from 1 to 10")
    return int(text.strip())


def parse_reflection(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    match = re.fullmatch(r"(.+?)\s*\[(\d+(?:\s*,\s*\d+)*)\]\s*", text.strip(), re.S)
    if match is None or not match[1].strip() or re.search(r"[\[\]\n]", match[1]):
        raise ValueError("Reflection requires an insight and bracketed evidence IDs")
    ids = list(dict.fromkeys(str(int(s.strip())) for s in match[2].split(",")))
    if not set(ids) <= allowed:
        raise ValueError("Reflection cites a memory outside the retrieved evidence")
    return match[1].strip(), ids


@dataclass
class Promotion:
    text: str
    importance: float
    insight: str


class RecAgentMemory:
    def __init__(
        self,
        state: MemoryState,
        *,
        config: MemoryConfig,
        embeddings: CachedEmbeddings,
        request: Callable,
        rng: random.Random,
        emit: Callable = lambda **kwargs: None,
    ):
        self.state, self.config, self.embeddings = state, config, embeddings
        self.request, self.rng, self.emit = request, rng, emit

    @staticmethod
    def _time(now: float):
        if not math.isfinite(now):
            raise ValueError("Memory requires finite historical time")

    def importance(self, text: str) -> float:
        score = self.request(
            "importance",
            prompts.IMPORTANCE.format(observation=text),
            validator=parse_importance,
        )
        return score / 10 * self.config.importance_weight

    def observe(self, observation: str, *, now: float) -> None:
        self._time(now)
        self.state.sensory.append(observation)
        if len(self.state.sensory) < self.config.sensory_capacity:
            return
        text = self.request(
            "sensory",
            prompts.SENSORY.format(
                observations="\n".join(
                    f"[{i}] {s}" for i, s in enumerate(self.state.sensory)
                )
            ),
        )
        importance = self.importance(text)
        self.state.sensory.clear()
        retained = importance > self.config.sensory_threshold
        self.emit(event="sensory", retained=retained, importance=importance, now=now)
        if retained:
            identity = f"s{self.state.next_short_id}"
            self.state.next_short_id += 1
            promotions = self.reinforce(text, identity=identity, now=now)
            self.state.short_term.append(
                MemoryRecord(
                    identity, text, self.embeddings.embed(text), importance, now, now
                )
            )
            self._evict()
            self.save(promotions, now=now)

    def reinforce(self, text: str, *, identity: str, now: float) -> list[Promotion]:
        """A retrieved memory enhances STM but is never inserted as a new STM entry."""
        self._time(now)
        embedding = self.embeddings.embed(text)
        for index, record in enumerate(self.state.short_term):
            similarity = cosine(embedding, record.embedding)
            if index == len(self.state.short_term) - 1:
                similarity += self.config.last_record_bonus
            probability = sigmoid(similarity)
            if (
                probability >= self.config.enhancement_probability
                and self.rng.random() <= probability
            ):
                record.supports.append(Support(identity, text))
                self.emit(
                    event="reinforce",
                    memory_id=record.identity,
                    source_id=identity,
                    count=len(record.supports),
                    probability=probability,
                    now=now,
                )
        promotions = []
        remaining = []
        for record in self.state.short_term:
            if len(record.supports) >= self.config.enhancement_threshold:
                evidence = [record.text, *(s.text for s in record.supports[:-1]), text]
                insight = self.request(
                    "promotion", prompts.PROMOTION.format(memories="; ".join(evidence))
                )
                promotions.append(Promotion(record.text, record.importance, insight))
                self.emit(event="promote", memory_id=record.identity, now=now)
            else:
                remaining.append(record)
        self.state.short_term = remaining
        return promotions

    def _evict(self) -> None:
        if len(self.state.short_term) <= self.config.stm_capacity:
            return
        # Protect the newest entry. Stable IDs fix the source's text-key collisions.
        victim = min(
            self.state.short_term[:-1], key=lambda r: (r.importance, len(r.supports))
        )
        self.state.short_term.remove(victim)
        for record in self.state.short_term:
            # Source removes one corresponding enhancement occurrence, not all.
            for i, support in enumerate(record.supports):
                if support.identity == victim.identity:
                    record.supports.pop(i)
                    break
        self.emit(event="evict", memory_id=victim.identity)

    def add_long_term(
        self, text: str, importance: float, *, now: float, reflecting: bool = False
    ) -> MemoryRecord:
        self._time(now)
        if not math.isfinite(importance) or not 0 <= importance <= 1:
            raise ValueError("Invalid long-term importance")
        identity = str(len(self.state.long_term))
        record = MemoryRecord(
            identity,
            text,
            self.embeddings.embed(f"[{identity}] {text}"),
            importance,
            now,
            now,
        )
        self.state.long_term.append(record)
        if not reflecting:
            self.state.aggregate_importance += importance
        self.emit(event="store", memory_id=identity, importance=importance, now=now)
        return record

    def retrieve(self, query: str, *, now: float) -> list[MemoryRecord]:
        self._time(now)
        stream = self.state.long_term
        if not stream:
            self.emit(event="retrieve", memory_ids=[], now=now)
            return []
        query_vector = np.asarray(self.embeddings.embed(query), dtype=np.float32)
        matrix = np.asarray([r.embedding for r in stream], dtype=np.float32)
        # FAISS IndexFlatL2 returns squared L2. Tombstones keep their old vectors.
        distances = np.sum((matrix - query_vector) ** 2, axis=1)
        if not np.isfinite(distances).all():
            raise ValueError("Non-finite retrieval distance")
        nearest = np.argsort(distances, kind="stable")[: self.config.relevance_k]
        relevance = {int(i): 1 - float(distances[i]) / math.sqrt(2) for i in nearest}
        scored = []
        for i, record in enumerate(stream):
            hours = (now - record.accessed_at) / 3600
            if hours < 0:
                raise ValueError("Retrieval time precedes a memory's access time")
            score = (
                (1 - self.config.decay_rate) ** hours
                + record.importance
                + relevance.get(i, 0.0)
            )
            if record.status == "active":
                scored.append((score, record))
        selected = [
            r for _, r in sorted(scored, key=lambda v: -v[0])[: self.config.retrieval_k]
        ]
        for record in selected:
            record.accessed_at = now
        self.emit(event="retrieve", memory_ids=[r.identity for r in selected], now=now)
        return selected

    def read(self, observation: str, *, now: float) -> str:
        retrieved = self.retrieve(observation, now=now)
        promotions = []
        for record in retrieved:
            promotions.extend(
                self.reinforce(record.display, identity=f"l{record.identity}", now=now)
            )
        # Capture the retrieved content before save_context's reflection/forgetting.
        content = [r.display for r in retrieved] + [
            r.text for r in self.state.short_term
        ]
        self.save(promotions, now=now)
        return "\n".join(content) or "No retained memories."

    def save(self, promotions: list[Promotion], *, now: float) -> None:
        insights = [(p.insight, self.importance(p.insight)) for p in promotions]
        for promotion in promotions:
            self.add_long_term(promotion.text, promotion.importance, now=now)
        for text, importance in insights:
            self.add_long_term(text, importance, now=now)
        if self.state.aggregate_importance > self.config.reflection_threshold:
            self.reflect(now=now)
            self.state.aggregate_importance = 0.0
        self.forget(now=now)

    def reflect(self, *, now: float) -> None:
        recent = self.state.long_term[-self.config.reflection_window :]
        if not any(r.status == "active" for r in self.state.long_term):
            self.emit(event="reflection_skipped_empty", now=now)
            return
        question = self.request(
            "reflection_question",
            prompts.REFLECTION_QUESTION.format(
                memories="\n".join(r.display for r in recent)
            ),
        )
        retrieved = self.retrieve(question, now=now)
        text, ids = self.request(
            "reflection_insight",
            prompts.REFLECTION_INSIGHT.format(
                question=question, memories="\n".join(r.display for r in retrieved)
            ),
            validator=lambda s: parse_reflection(s, {r.identity for r in retrieved}),
        )
        embedding = self.embeddings.embed(text)
        parents = [self.state.long_term[int(i)] for i in ids]
        merged = []
        for record in parents:
            if (
                sigmoid(cosine(embedding, self.embeddings.embed(record.display)))
                >= self.config.merge_threshold
            ):
                record.status, record.importance, record.accessed_at = (
                    "merged",
                    1.0,
                    now,
                )
                merged.append(record.identity)
        # Source averages AFTER setting merged parents' importance to 1.
        importance = sum(r.importance for r in parents) / len(parents)
        self.add_long_term(text, importance, now=now, reflecting=True)
        self.emit(event="reflect", evidence_ids=ids, merged_ids=merged, now=now)

    def forgetting_probabilities(self, *, now: float) -> list[float]:
        weights = []
        for record in self.state.long_term:
            hours = (now - record.accessed_at) / 3600
            if hours < 0:
                raise ValueError("Forgetting time precedes a memory's access time")
            recency = (1 - self.config.decay_rate) ** hours
            retention = (
                max(
                    recency**self.config.forgetting_exponent,
                    self.config.forgetting_floor,
                )
                * (record.importance + recency)
                / 2
            )
            weights.append(max(0.0, 1 - retention))
        total = sum(weights)
        return [w / total for w in weights] if total else [0.0] * len(weights)

    def forget(self, *, now: float) -> None:
        self._time(now)
        probabilities = self.forgetting_probabilities(now=now)
        for record, probability in zip(
            self.state.long_term, probabilities, strict=True
        ):
            if (now - record.accessed_at) / 3600 <= self.config.forgetting_grace_hours:
                continue
            if self.rng.random() < probability:
                record.status, record.importance, record.accessed_at = (
                    "forgotten",
                    1.0,
                    now,
                )
                self.emit(
                    event="forget",
                    memory_id=record.identity,
                    probability=probability,
                    now=now,
                )
