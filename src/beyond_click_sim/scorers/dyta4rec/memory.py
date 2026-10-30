"""Narrow adapter: importance and retrieval only; never the RecAgent lifecycle."""

import math
import random

import numpy as np

from ..recagent.memory import RecAgentMemory
from ..recagent.state import MemoryConfig, MemoryRecord, MemoryState, Support


def adapter(records, *, config, embeddings, request):
    # Construct fresh records, so inherited access-time updates are disposable.
    state = MemoryState(
        long_term=[
            MemoryRecord(
                r["id"],
                r["text"],
                list(r["embedding"]),
                r["importance"],
                r["created_at"],
                r["accessed_at"],
                [Support(ref, ref) for ref in r["source_event_refs"]],
            )
            for r in records
        ]
    )
    return RecAgentMemory(
        state,
        config=MemoryConfig(
            importance_weight=config.importance_weight,
            retrieval_k=config.retrieval_k,
            relevance_k=config.relevance_k,
            decay_rate=config.decay_rate,
        ),
        embeddings=embeddings,
        request=request,
        rng=random.Random(config.seed),
    )


def retrieve(records, query, *, now, config, embeddings):
    memory = adapter(
        records,
        config=config,
        embeddings=embeddings,
        request=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Retrieval must not generate text")
        ),
    )
    query_vector = np.asarray(embeddings.embed(query), dtype=np.float32)
    matrix = np.asarray([r["embedding"] for r in records], dtype=np.float32)
    distances = np.sum((matrix - query_vector) ** 2, axis=1)
    nearest = set(np.argsort(distances, kind="stable")[: config.relevance_k].tolist())
    selected = memory.retrieve(query, now=now)
    positions = {r["id"]: i for i, r in enumerate(records)}
    diagnostics = []
    for record in selected:
        i = positions[record.identity]
        original = records[i]
        recency = (1 - config.decay_rate) ** ((now - original["accessed_at"]) / 3600)
        relevance = 1 - float(distances[i]) / math.sqrt(2) if i in nearest else 0.0
        diagnostics.append(
            {
                "id": record.identity,
                "squared_l2_distance": float(distances[i]),
                "semantic_contribution": relevance,
                "importance": record.importance,
                "recency": recency,
                "retrieval_score": recency + record.importance + relevance,
                "source_event_refs": original["source_event_refs"],
                "created_at": original["created_at"],
                "snapshot_accessed_at": original["accessed_at"],
            }
        )
    return [{"ref": r.identity, "text": r.text} for r in selected], diagnostics
