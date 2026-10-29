"""Serializable state for the offline RecAgent port (no executable snapshots)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any


PORT_VERSION = "recagent-offline-v1"
AUTHOR_REVISION = "fff935570079502d5a17509b0baf590becd1eaab"


def digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    ) as file:
        tmp = Path(file.name)
        try:
            json.dump(value, file, sort_keys=True, ensure_ascii=False, allow_nan=False)
            file.flush()
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def random_state(value: Any) -> Any:
    return (
        tuple(random_state(v) for v in value)
        if isinstance(value, (tuple, list))
        else value
    )


@dataclass(frozen=True)
class MemoryConfig:
    sensory_capacity: int = 1
    importance_weight: float = 0.9
    sensory_threshold: float = 0.62
    stm_capacity: int = 10
    enhancement_threshold: int = 3
    enhancement_probability: float = 0.7
    last_record_bonus: float = 0.1
    retrieval_k: int = 5
    relevance_k: int = 100
    decay_rate: float = 0.01
    reflection_threshold: float = 10.0
    reflection_window: int = 50
    merge_threshold: float = 0.72
    forgetting_exponent: float = 1.5
    forgetting_floor: float = 0.01
    forgetting_grace_hours: float = 24.0

    def __post_init__(self):
        for name in (
            "sensory_capacity",
            "stm_capacity",
            "enhancement_threshold",
            "retrieval_k",
            "relevance_k",
            "reflection_window",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid memory setting {name}")
        for name in (
            "importance_weight",
            "sensory_threshold",
            "enhancement_probability",
            "merge_threshold",
            "forgetting_floor",
        ):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")
        if not 0 < self.decay_rate < 1:
            raise ValueError("decay_rate must be between 0 and 1")


@dataclass
class Support:
    identity: str
    text: str


@dataclass
class MemoryRecord:
    identity: str
    text: str
    embedding: list[float]
    importance: float
    created_at: float
    accessed_at: float
    supports: list[Support] = field(default_factory=list)
    status: str = "active"

    @property
    def display(self) -> str:
        if self.status != "active":
            return "[MERGE]" if self.status == "merged" else "[FORGET]"
        return f"[{self.identity}] {self.text}"


@dataclass
class MemoryState:
    sensory: list[str] = field(default_factory=list)
    short_term: list[MemoryRecord] = field(default_factory=list)
    long_term: list[MemoryRecord] = field(default_factory=list)
    aggregate_importance: float = 0.0
    next_short_id: int = 0


@dataclass
class AgentSnapshot:
    user_id: str
    demographics: dict[str, str]
    interests: str = "No interests inferred yet."
    memory: MemoryState = field(default_factory=MemoryState)
    recent: list[str] = field(default_factory=list)
    profile_blocks: int = 0
    replay_events: int = 0
    clock: float = 0.0
    rng_state: Any = None
    completed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AgentSnapshot:
        data = dict(data)
        memory = dict(data.pop("memory"))
        for key in ("short_term", "long_term"):
            memory[key] = [
                MemoryRecord(**{**r, "supports": [Support(**s) for s in r["supports"]]})
                for r in memory[key]
            ]
        return cls(**data, memory=MemoryState(**memory))

    def rng(self) -> random.Random:
        result = random.Random()
        result.setstate(random_state(self.rng_state))
        return result
