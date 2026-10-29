"""Small, isolated embedding/cache utilities for LLMSRec-Syn."""

from __future__ import annotations

from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path
import sqlite3
from threading import RLock
from time import perf_counter
from typing import Any

import numpy as np


SERIALIZATION_VERSION = "llmsrec_syn_title_sequence_json_v1"


def scalar(value: Any) -> str | int:
    """Use typed scalar IDs; reject ambiguous/missing identifiers."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    raise ValueError(f"Expected a nonempty string or integer identifier: {value!r}")


def identity(value: Any) -> str:
    value = scalar(value)
    return f"{type(value).__name__}:{json.dumps(value, ensure_ascii=False)}"


def digest(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def serialize_titles(titles: list[str]) -> str:
    # The manuscript does not specify a byte-level embedding input. Preserve
    # order and repeated titles, without IDs, ratings or demonstration labels.
    return json.dumps(titles, ensure_ascii=False)


def token_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    values = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = (
            usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
        )
        if isinstance(value, Integral):
            values[field] = int(value)
    return values or None


class CachedEmbeddings:
    """OpenAI-compatible embeddings with exact-input, model and split keys.

    No client construction or network calls occur at import/initialization.
    The lock protects the cache and deduplicates simultaneous misses. Each
    request batches multiple prefixes; LLM generation remains concurrent.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        model_identity: str,
        training_fingerprint: str,
        cache_path: Path | None = None,
        batch_size: int = 64,
    ) -> None:
        if not model or not model_identity or not training_fingerprint:
            raise ValueError(
                "Embedding model, identity and training fingerprint are required"
            )
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size < 1
        ):
            raise ValueError("embedding batch_size must be positive")
        self.client = client
        self.model = model
        self.namespace = {
            "model": model,
            "model_identity": model_identity,
            "training_fingerprint": training_fingerprint,
            "serialization": SERIALIZATION_VERSION,
        }
        self.batch_size = int(batch_size)
        self._lock = RLock()
        self._memory: dict[str, np.ndarray] = {}
        self._dimension: int | None = None
        self._requests: list[dict[str, Any]] = []
        self._hits = 0
        self._connection = None
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                cache_path, check_same_thread=False, timeout=60
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
            )
            self._connection.commit()

    def _vector(self, raw: Any) -> np.ndarray:
        vector = np.asarray(raw, dtype=np.float64)
        if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
            raise ValueError("Embedding must be a nonempty finite vector")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Embedding must have a finite nonzero norm")
        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError("Embedding dimensions differ within one model identity")
        self._dimension = len(vector)
        return vector / norm

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed an empty sequence of inputs")
        with self._lock:
            keys = [digest({**self.namespace, "text": text}) for text in texts]
            missing: dict[str, str] = {}
            for key, text in zip(keys, texts, strict=True):
                if key not in self._memory and self._connection is not None:
                    row = self._connection.execute(
                        "SELECT vector FROM embeddings WHERE key = ?", (key,)
                    ).fetchone()
                    if row is not None:
                        self._memory[key] = self._vector(json.loads(row[0]))
                if key in self._memory:
                    self._hits += 1
                else:
                    missing[key] = text
            entries = list(missing.items())
            for start in range(0, len(entries), self.batch_size):
                batch = entries[start : start + self.batch_size]
                record: dict[str, Any] = {
                    "input_count": len(batch),
                    "completed": False,
                    "usage": None,
                }
                started = perf_counter()
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=[text for _, text in batch],
                    )
                    record["usage"] = token_usage(response)
                    data = sorted(response.data, key=lambda entry: entry.index)
                    if [entry.index for entry in data] != list(range(len(batch))):
                        raise ValueError(
                            "Embedding response indices must cover the complete batch"
                        )
                    vectors = [self._vector(entry.embedding) for entry in data]
                    for (key, _), vector, entry in zip(
                        batch, vectors, data, strict=True
                    ):
                        if self._connection is not None:
                            self._connection.execute(
                                "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                                (
                                    key,
                                    json.dumps(
                                        np.asarray(
                                            entry.embedding, dtype=float
                                        ).tolist()
                                    ),
                                ),
                            )
                        self._memory[key] = vector
                    if self._connection is not None:
                        self._connection.commit()
                    record["completed"] = True
                except Exception as error:
                    record["error"] = repr(error)
                    raise
                finally:
                    record["elapsed_seconds"] = perf_counter() - started
                    self._requests.append(record)
            result = np.stack([self._memory[key] for key in keys])
            # Target-specific cutoffs can create many distinct prefixes. The
            # SQLite cache persists them without retaining every vector in RAM.
            while len(self._memory) > 4096:
                self._memory.pop(next(iter(self._memory)))
            return result

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            usages = [r["usage"] for r in self._requests]
            return {
                "namespace": dict(self.namespace),
                "cache_hits": self._hits,
                "scorer_issued_requests": len(self._requests),
                "embedded_inputs": sum(
                    r["input_count"] for r in self._requests if r["completed"]
                ),
                "requests": [dict(r) for r in self._requests],
                "usage_complete": bool(usages)
                and all(u is not None and "total_tokens" in u for u in usages),
                "reported_total_tokens": sum(
                    u["total_tokens"] for u in usages if u and "total_tokens" in u
                )
                if any(u and "total_tokens" in u for u in usages)
                else None,
            }

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
