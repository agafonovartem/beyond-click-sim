from copy import deepcopy
import math
import random

import pytest

from beyond_click_sim.scorers.dyta4rec.config import DyTA4RecConfig
from beyond_click_sim.scorers.dyta4rec.memory import adapter, retrieve
from beyond_click_sim.scorers.recagent.embeddings import CachedEmbeddings
from beyond_click_sim.scorers.recagent.memory import RecAgentMemory
from beyond_click_sim.scorers.recagent.state import (
    MemoryConfig,
    MemoryRecord,
    MemoryState,
)
from test_dyta4rec_scorer import FakeEmbeddings, scorer


def record(i, embedding, importance=0.5, accessed=0):
    return {
        "id": f"S{i}",
        "text": f"summary {i}",
        "embedding": embedding,
        "importance": importance,
        "created_at": accessed,
        "accessed_at": accessed,
        "source_event_refs": [f"E{i}"],
    }


class ConstantEmbedding(FakeEmbeddings):
    def embed(self, text):
        self.calls.append(text)
        return [1, 0, 0]


@pytest.mark.parametrize("importance", ["1", "10", "8"])
def test_importance_is_same_upstream_port_prompt_and_scale(importance):
    calls = []

    def request(stage, text, validator):
        calls.append((stage, text))
        return validator(importance)

    mem = adapter([], config=DyTA4RecConfig(), embeddings=None, request=request)
    actual = mem.importance("a historical summary")
    direct = RecAgentMemory(
        MemoryState(),
        config=MemoryConfig(),
        embeddings=None,
        request=request,
        rng=random.Random(0),
    )
    assert (
        actual
        == direct.importance("a historical summary")
        == int(importance) / 10 * 0.9
    )
    assert calls[0] == calls[1]


@pytest.mark.parametrize("bad", ["0", "11", "8.0", "Importance: 8", "", "yes"])
def test_importance_range_is_strict(bad):
    with pytest.raises(ValueError):
        adapter(
            [],
            config=DyTA4RecConfig(),
            embeddings=None,
            request=lambda stage, text, validator: validator(bad),
        ).importance("summary")


def test_retrieval_matches_direct_port_old_evidence_wins_and_state_is_frozen():
    config = DyTA4RecConfig()
    embeddings = CachedEmbeddings(ConstantEmbedding())
    records = [record(1, [1, 0, 0], 0.9, 0), record(2, [8, 0, 0], 0.1, 3600)]
    before = deepcopy(records)
    selected, terms = retrieve(
        records, "candidate observation", now=7200, config=config, embeddings=embeddings
    )
    direct = RecAgentMemory(
        MemoryState(
            long_term=[
                MemoryRecord(
                    r["id"],
                    r["text"],
                    r["embedding"],
                    r["importance"],
                    r["created_at"],
                    r["accessed_at"],
                )
                for r in records
            ]
        ),
        config=MemoryConfig(),
        embeddings=embeddings,
        request=None,
        rng=random.Random(0),
    ).retrieve("candidate observation", now=7200)
    assert [r["ref"] for r in selected] == [r.identity for r in direct] == ["S1", "S2"]
    assert terms[0]["recency"] == pytest.approx(0.99**2)
    assert terms[0]["retrieval_score"] == pytest.approx(0.99**2 + 0.9 + 1)
    assert terms[1]["semantic_contribution"] == pytest.approx(1 - 49 / math.sqrt(2))
    assert terms[1]["semantic_contribution"] < 0
    assert records == before


def test_semantic_pool_100_stable_ties_and_top_five():
    config = DyTA4RecConfig()
    embeddings = CachedEmbeddings(ConstantEmbedding())
    # Equal-distance 101st record gets zero semantic contribution, not 1-d²/sqrt(2).
    records = [record(i, [3, 0, 0]) for i in range(101)]
    selected, terms = retrieve(
        records, "query", now=1, config=config, embeddings=embeddings
    )
    assert [r["ref"] for r in selected] == ["S100", "S0", "S1", "S2", "S3"]
    assert terms[0]["semantic_contribution"] == 0
    assert all(t["semantic_contribution"] < 0 for t in terms[1:])
    assert embeddings.provider.calls == ["query"]


def test_snapshot_embeddings_and_encoder_provenance():
    s = scorer().build_agents()
    state = s.agent_snapshots["u"].to_dict()
    assert state["configuration"]["embedding"] == s.embeddings.identity
    assert s.embeddings.identity["revision"] == "fake-v1"
    assert s.embeddings.identity["max_seq_length"] == 384
    assert state["configuration"]["memory_mode"] == "yulan_retrieval_two_level_v1"
