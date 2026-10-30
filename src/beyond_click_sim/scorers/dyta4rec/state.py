"""Immutable JSON snapshots and content-addressed, non-executable checkpoints."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from ..recagent.state import atomic_json, digest


@dataclass(frozen=True)
class Snapshot:
    identity: str
    payload: str

    @classmethod
    def create(cls, data):
        return cls(
            digest(data),
            json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False),
        )

    def to_dict(self):
        # Every caller gets an isolated value, including on concurrent retrieval.
        return json.loads(self.payload)


def source_hashes():
    root = Path(__file__).resolve().parent
    paths = [*root.glob("*.py"), *(root.parent / "recagent").glob("*.py")]
    return {
        str(p.relative_to(root.parent)): sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def read_cache(path, identity):
    if path is None or not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("identity") != identity or digest(value["data"]) != value["checksum"]:
        raise ValueError(f"Invalid DyTA4Rec cache: {path}")
    return value["data"]


def write_cache(path, identity, data):
    if path is not None:
        atomic_json(
            path, {"identity": identity, "checksum": digest(data), "data": data}
        )
