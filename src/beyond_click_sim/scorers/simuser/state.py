"""Immutable JSON snapshots and checksummed atomic checkpoints (no pickle)."""

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

from beyond_click_sim.scorers.history.llmsrec_syn_retrieval import digest, identity


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(dumps(value) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, key: str, payload):
    atomic_json(path, {"key": key, "sha256": digest(payload), "payload": payload})


def load_checkpoint(path: Path, key: str):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("key") != key or data.get("sha256") != digest(data.get("payload")):
        raise ValueError("SimUSER checkpoint identity/checksum mismatch")
    return data["payload"]


@dataclass(frozen=True)
class AgentSnapshot:
    """Nested values are serialized so callers cannot mutate fitted state."""

    user_id: str
    payload_json: str

    @property
    def fingerprint(self):
        return digest({"user_id": self.user_id, "payload": self.to_dict()})

    def to_dict(self):
        return json.loads(self.payload_json)


__all__ = [
    "AgentSnapshot",
    "atomic_json",
    "digest",
    "dumps",
    "identity",
    "load_checkpoint",
    "save_checkpoint",
]
