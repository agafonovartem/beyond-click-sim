from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ManifestTable:
    """One materialized table entry in a canonical dataset manifest."""

    path: str
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "rows": int(self.rows)}


@dataclass(frozen=True)
class CanonicalManifest:
    """Common manifest shape shared by all canonical dataset adapters."""

    dataset: str
    adapter: str
    version: str
    raw_sources: list[dict[str, Any]]
    tables: dict[str, ManifestTable]
    id_policy: dict[str, str]
    caveats: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    extra_sections: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        common_keys = {
            "dataset",
            "adapter",
            "version",
            "schema_version",
            "raw_sources",
            "tables",
            "id_policy",
            "caveats",
        }
        extra_conflicts = common_keys & set(self.extra_sections)
        if extra_conflicts:
            raise ValueError(
                f"Manifest extra_sections override common keys: {sorted(extra_conflicts)}"
            )

        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "adapter": self.adapter,
            "version": self.version,
            "schema_version": self.schema_version,
            "raw_sources": self.raw_sources,
            "tables": {
                name: table.to_dict() for name, table in self.tables.items()
            },
            "id_policy": self.id_policy,
            "caveats": self.caveats,
        }
        payload.update(self.extra_sections)
        return payload


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


def write_manifest(path: Path, manifest: CanonicalManifest | dict[str, Any]) -> None:
    manifest_dict = manifest.to_dict() if isinstance(manifest, CanonicalManifest) else manifest
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        **manifest_dict,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
