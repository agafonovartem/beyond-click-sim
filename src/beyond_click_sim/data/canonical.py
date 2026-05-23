from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "canonical-v1"


@dataclass(frozen=True)
class CanonicalDataset:
    """Descriptor for materialized canonical dataset tables."""

    name: str
    version: str
    root: Path
    users_path: Path
    items_path: Path
    interactions_path: Path
    manifest_path: Path

    def load_users(self, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.users_path, columns=columns)

    def load_items(self, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.items_path, columns=columns)

    def load_interactions(self, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.interactions_path, columns=columns)

    def load_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # Streaming fingerprint to track raw files; this does not load the full file.
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        **manifest,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
