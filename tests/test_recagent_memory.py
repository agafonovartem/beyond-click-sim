from dataclasses import replace
import math
import random
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from beyond_click_sim.scorers.recagent.embeddings import CachedEmbeddings, vector
from beyond_click_sim.scorers.recagent import embeddings as embedding_module
from beyond_click_sim.scorers.recagent.memory import (
    RecAgentMemory,
    parse_importance,
    parse_reflection,
)
from beyond_click_sim.scorers.recagent.state import (
    MemoryConfig,
    MemoryRecord,
    MemoryState,
    Support,
)


class FakeEmbeddings:
    def __init__(self, values=None, revision="fixture-v1"):
        self.values, self.revision = values or {}, revision
        self.calls = []

    def metadata(self):
        return {
            "model": "fake",
            "revision": self.revision,
            "dimensions": 2,
            "normalize_embeddings": False,
        }

    def embed(self, text):
        self.calls.append(text)
        return self.values.get(text, [1.0, 0.0])


class FixedRNG:
    def __init__(self, draw=0.0):
        self.draw, self.calls = draw, 0

    def random(self):
        self.calls += 1
        return self.draw


class MemoryRequests:
    def __init__(self, importance="8", **responses):
        self.responses = {
            "sensory": "Recorded movie interaction.",
            "importance": importance,
            "promotion": "The user may be curious about comedy.",
            "reflection_question": "What interests recur?",
            "reflection_insight": None,
            **responses,
        }
        self.calls = []

    def __call__(self, stage, prompt, validator=None):
        self.calls.append((stage, prompt))
        text = self.responses[stage]
        if stage == "reflection_insight" and text is None:
            identity = re.search(r"^\[(\d+)\]", prompt, re.M)[1]
            text = f"Comedies recur [{identity}]"
        return text if validator is None else validator(text)


def memory(*, config=None, rng=None, provider=None, request=None):
    return RecAgentMemory(
        MemoryState(),
        config=config or MemoryConfig(),
        embeddings=CachedEmbeddings(provider or FakeEmbeddings()),
        request=request or MemoryRequests(),
        rng=rng or FixedRNG(),
    )


def short(identity, text="same", importance=0.72, supports=None, embedding=None):
    return MemoryRecord(
        identity,
        text,
        embedding or [1.0, 0.0],
        importance,
        0,
        0,
        supports=supports or [],
    )


@pytest.mark.parametrize(
    "score,retained", [(1, False), (6, False), (7, True), (10, True)]
)
def test_sensory_integer_boundary(score, retained):
    m = memory(request=MemoryRequests(importance=str(score)))
    m.observe("A recorded rating", now=42)
    assert bool(m.state.short_term) == retained
    assert not m.state.sensory
    if retained:
        r = m.state.short_term[0]
        assert r.importance == pytest.approx(score * 0.09)
        assert r.created_at == r.accessed_at == 42


@pytest.mark.parametrize("bad", ["0", "11", "7.0", "important: 8", "", "7\n8"])
def test_importance_rejects_invalid_outputs(bad):
    with pytest.raises(ValueError):
        parse_importance(bad)


def test_promotion_after_three_supports_includes_original_and_insight():
    m = memory()
    m.state.short_term.append(short("s0", "first movie"))
    for i in range(2):
        assert m.reinforce(f"support{i}", identity=f"s{i + 1}", now=100) == []
    promotions = m.reinforce("support2", identity="s3", now=100)
    assert len(promotions) == 1
    assert not m.state.short_term
    m.save(promotions, now=100)
    assert [r.text for r in m.state.long_term] == [
        "first movie",
        "The user may be curious about comedy.",
    ]
    assert all(r.created_at == r.accessed_at == 100 for r in m.state.long_term)
    prompt = next(p for stage, p in m.request.calls if stage == "promotion")
    assert "first movie; support0; support1; support2" in prompt


def test_enhancement_has_last_record_bonus_and_stochastic_gate():
    for draw, count in [(0.7, 1), (0.99, 0)]:
        m = memory(rng=FixedRNG(draw))
        # cosine .8 is below sigmoid threshold; .8+.1 passes for final record.
        m.state.short_term = [
            short("s0", embedding=[0.8, 0.6]),
            short("s1", embedding=[0.8, 0.6]),
        ]
        m.reinforce("query", identity="l0", now=0)
        assert len(m.state.short_term[0].supports) == 0
        assert len(m.state.short_term[1].supports) == count
        assert m.rng.calls == 1


def test_eviction_uses_identity_and_keeps_newest_duplicate():
    m = memory(config=replace(MemoryConfig(), stm_capacity=2))
    m.state.short_term = [
        short("s0", importance=0.5),
        short(
            "s1",
            importance=0.5,
            supports=[Support("s0", "same"), Support("s0", "same")],
        ),
        short("s2", importance=0.1),
    ]
    m._evict()
    assert [r.identity for r in m.state.short_term] == ["s1", "s2"]
    assert len(m.state.short_term[0].supports) == 1


def test_retrieval_matches_squared_l2_recency_importance_and_tombstone_slots():
    cfg = replace(MemoryConfig(), retrieval_k=2, relevance_k=1)
    m = memory(config=cfg)
    a = m.add_long_term("closest but deleted", 0.5, now=0)
    b = m.add_long_term("far but important", 0.9, now=0)
    c = m.add_long_term("less important", 0.1, now=0)
    a.status = "forgotten"
    b.embedding, c.embedding = [3.0, 0.0], [1.1, 0.0]
    # Tombstone consumes the only relevance slot; all active records still compete.
    assert [r.identity for r in m.retrieve("query", now=3600)] == ["1", "2"]
    assert b.accessed_at == c.accessed_at == 3600
    assert a.accessed_at == 0
    # With 100 relevance slots, c's near-zero distance wins over b's squared distance 4.
    m.config = replace(cfg, relevance_k=100)
    assert [r.identity for r in m.retrieve("query", now=3600)] == ["2", "1"]


def test_retrieval_scores_all_records_even_outside_relevance_top_k():
    m = memory(config=replace(MemoryConfig(), relevance_k=1, retrieval_k=1))
    a = m.add_long_term("old and unimportant", 0.01, now=0)
    b = m.add_long_term("recent and important", 0.9, now=1_000_000)
    b.embedding = [2, 0]
    assert m.retrieve("query", now=1_000_000) == [b]
    assert a.accessed_at == 0


def test_read_enhances_without_inserting_retrieved_records_and_can_promote():
    m = memory()
    m.add_long_term("related", 0.8, now=0)
    m.state.short_term = [
        short("s0", supports=[Support("x", "one"), Support("y", "two")])
    ]
    context = m.read("query", now=1)
    assert "related" in context
    assert not m.state.short_term
    assert len(m.state.long_term) == 3
    assert m.state.long_term[1].created_at == 1


def test_reflection_strict_threshold_window_merge_and_parent_importance():
    req = MemoryRequests(reflection_insight="An inferred pattern [50]")
    m = memory(request=req)
    for i in range(51):
        m.add_long_term(f"record{i}", 0.0, now=0)
    # Make record50 the only active record so it is retrieved.
    for r in m.state.long_term[:-1]:
        r.status = "merged"
    m.state.long_term[-1].importance = 0.8
    m.state.aggregate_importance = 10
    m.save([], now=0)
    assert not req.calls
    m.state.aggregate_importance = 10.01
    m.save([], now=0)
    assert m.state.aggregate_importance == 0
    assert m.state.long_term[50].status == "merged"
    assert m.state.long_term[-1].importance == 1.0
    assert m.state.long_term[-1].text == "An inferred pattern"
    prompt = next(p for s, p in req.calls if s == "reflection_question")
    assert prompt.count("[MERGE]") == 49  # 50-record window, not all 51.
    assert [s for s, _ in req.calls] == ["reflection_question", "reflection_insight"]


@pytest.mark.parametrize(
    "text",
    [
        "No IDs",
        "Claim [8]",
        "Claim [-1]",
        "Claim [0, 8]",
        "Claim [99] plus another [0]",
    ],
)
def test_reflection_requires_actual_retrieved_evidence(text):
    with pytest.raises(ValueError):
        parse_reflection(text, {"0"})


def test_forgetting_formula_normalization_grace_and_tombstones():
    m = memory()
    m.add_long_term("old", 0.7, now=0)
    m.add_long_term("fresh", 0.2, now=90_000)
    r = 0.99**25
    q0 = 1 - max(r**1.5, 0.01) * (0.7 + r) / 2
    q1 = 1 - (0.2 + 1) / 2
    assert m.forgetting_probabilities(now=90_000) == pytest.approx(
        [q0 / (q0 + q1), q1 / (q0 + q1)]
    )
    m.forget(now=90_000)
    assert m.state.long_term[0].status == "forgotten"
    assert m.state.long_term[0].accessed_at == 90_000
    assert m.state.long_term[0].importance == 1
    assert m.state.long_term[1].status == "active"


def test_empty_degenerate_and_exact_grace_boundary():
    m = memory()
    assert m.retrieve("query", now=0) == []
    assert m.forgetting_probabilities(now=0) == []
    m.forget(now=0)
    m.add_long_term("perfect", 1, now=0)
    assert m.forgetting_probabilities(now=0) == [0]
    m.forget(now=86400)
    assert m.state.long_term[0].status == "active"
    with pytest.raises(ValueError, match="precedes"):
        m.retrieve("query", now=-1)


@pytest.mark.parametrize("v", [[0, 0], [math.nan, 1], [math.inf, 1], [1], [[1, 0]]])
def test_invalid_embeddings_fail_explicitly(v):
    with pytest.raises(ValueError):
        vector(v, 2)


def test_numeric_embedding_cache_and_revision_invalidation(tmp_path):
    first = FakeEmbeddings()
    cache = CachedEmbeddings(first, tmp_path)
    assert cache.embed("text") == cache.embed("text") == [1, 0]
    assert first.calls == ["text"]
    second = FakeEmbeddings()
    CachedEmbeddings(second, tmp_path).embed("text")
    assert second.calls == []
    changed = FakeEmbeddings(revision="v2")
    CachedEmbeddings(changed, tmp_path).embed("text")
    assert changed.calls == ["text"]
    assert all(
        np.load(p, allow_pickle=False).shape == (2,) for p in tmp_path.glob("*.npy")
    )


def test_real_local_rng_repeatable():
    def run():
        m = memory(rng=random.Random(2023))
        for i in range(40):
            m.observe(f"movie{i}", now=i * 86400)
        return m.state

    assert run() == run()


def test_mpnet_resolves_local_revision_before_lazy_cpu_load(monkeypatch):
    calls = []
    revision = "a" * 40

    def snapshot(model, **kwargs):
        calls.append(("snapshot", model, kwargs))
        assert kwargs == {"revision": "pinned-request", "local_files_only": True}
        return f"/cache/snapshots/{revision}"

    class FakeSentenceTransformer:
        def __init__(self, path, **kwargs):
            calls.append(("load", path, kwargs))
            assert kwargs == {"device": "cpu", "local_files_only": True}

        def encode(self, text, **kwargs):
            calls.append(("encode", text, kwargs))
            assert self.max_seq_length == 384
            assert kwargs["normalize_embeddings"] is False
            assert kwargs["precision"] == "float32"
            return np.ones(768, dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot)
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setattr(embedding_module, "version", lambda name: "test-version")
    provider = embedding_module.LocalMPNet(revision="pinned-request")
    identity = provider.metadata()
    assert identity["revision"] == revision
    assert identity["libraries"]["sentence-transformers"] == "test-version"
    assert [c[0] for c in calls] == ["snapshot"]
    assert len(provider.embed("movie")) == 768
    assert [c[0] for c in calls] == ["snapshot", "load", "encode"]


def test_mpnet_missing_local_snapshot_never_tries_network(monkeypatch):
    def missing(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        raise FileNotFoundError("not provisioned")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=missing)
    )
    with pytest.raises(FileNotFoundError):
        embedding_module.LocalMPNet().metadata()
