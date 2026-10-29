"""Lazy local MPNet and a numeric, content-addressed embedding cache."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import tempfile
from threading import RLock
from typing import Protocol

import numpy as np

from .state import digest


class EmbeddingProvider(Protocol):
    def metadata(self) -> dict: ...
    def embed(self, text: str) -> list[float]: ...


class LocalMPNet:
    """Never downloads weights. Provision the optional extra and snapshot first."""

    def __init__(self, *, revision: str = "main"):
        self.revision = revision
        self._path = None
        self._model = None

    def metadata(self) -> dict:
        if self._path is None:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as error:
                raise RuntimeError(
                    "Install the recagent optional dependencies first"
                ) from error
            self._path = Path(
                snapshot_download(
                    "sentence-transformers/all-mpnet-base-v2",
                    revision=self.revision,
                    local_files_only=True,
                )
            )
        if not re.fullmatch(r"[0-9a-f]{40}", self._path.name):
            raise ValueError(
                "MPNet cache must resolve to an immutable snapshot revision"
            )
        try:
            libraries = {
                name: version(name)
                for name in ("sentence-transformers", "transformers", "torch", "numpy")
            }
        except PackageNotFoundError as error:
            raise RuntimeError(
                "Install the recagent optional dependencies first"
            ) from error
        return {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "revision": self._path.name,
            "dimensions": 768,
            "device": "cpu",
            "normalize_embeddings": False,
            "precision": "float32",
            "batch_size": 1,
            "max_seq_length": 384,
            "long_text_policy": "encoder_default_truncation_384",
            "libraries": libraries,
        }

    def embed(self, text: str) -> list[float]:
        self.metadata()
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Install beyond-click-sim[recagent] first"
                ) from error
            self._model = SentenceTransformer(
                str(self._path), device="cpu", local_files_only=True
            )
            self._model.max_seq_length = 384
        return self._model.encode(
            text,
            batch_size=1,
            normalize_embeddings=False,
            precision="float32",
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()


def vector(value, dimensions: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if (
        result.shape != (dimensions,)
        or not np.isfinite(result).all()
        or np.linalg.norm(result) == 0
    ):
        raise ValueError(
            "Embedding must have the declared dimension, finite values and nonzero norm"
        )
    return result


class CachedEmbeddings:
    def __init__(self, provider: EmbeddingProvider, directory: Path | None = None):
        self.provider = provider
        self.directory = directory
        self.identity = provider.metadata()
        if not self.identity.get("revision") or not self.identity.get("dimensions"):
            raise ValueError(
                "Embedding provider must report actual revision and dimensions"
            )
        # Also reject non-JSON provenance before creating persistent initialization.
        self.key = digest(self.identity)
        self._vectors = {}
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> list[float]:
        key = digest({"provider": self.key, "text": text})
        with self._lock:
            if key in self._vectors:
                self.hits += 1
                return self._vectors[key].copy()
            path = None if self.directory is None else self.directory / f"{key}.npy"
            if path is not None and path.exists():
                with path.open("rb") as file:
                    result = vector(
                        np.load(file, allow_pickle=False), self.identity["dimensions"]
                    )
                self.hits += 1
            else:
                result = vector(self.provider.embed(text), self.identity["dimensions"])
                self.misses += 1
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        dir=path.parent, delete=False
                    ) as file:
                        temporary = Path(file.name)
                        try:
                            np.save(file, result, allow_pickle=False)
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            raise
                    try:
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
            self._vectors[key] = result.tolist()
            return self._vectors[key].copy()

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "provider": json.loads(json.dumps(self.identity)),
            }
