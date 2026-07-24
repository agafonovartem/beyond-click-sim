"""Durable per-candidate-group checkpoints for paid KuaiRand LLM runs.

The runner evaluates one listwise candidate group with one API request.  Keeping
only the final ``predictions.parquet`` therefore makes an interrupted run
expensive to resume: every successful request before the interruption would be
repeated.  This module stores each successful group independently and atomically.

Checkpoint keys include the exact prompt and request configuration.  A change to
the task, cell, user, prompt, model, or request parameters is consequently a
cache miss rather than a silent reuse of incompatible predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointCorruptionError(ValueError):
    """Raised when a completed checkpoint exists but cannot be trusted."""


@dataclass(frozen=True)
class CachedGroupScores:
    """Validated scores loaded from one completed group checkpoint."""

    key: str
    item_ids: tuple[Any, ...]
    scores: tuple[float, ...]
    usage: dict[str, int] | None
    created_at_utc: str


def build_group_checkpoint_key(
    *,
    task_identity: Mapping[str, Any] | str,
    cell: Mapping[str, Any],
    user_id: Any,
    candidate_group: Any,
    candidate_item_ids: Sequence[Any],
    prompt_messages: Sequence[Mapping[str, Any]],
    model: str,
    request_config: Mapping[str, Any],
) -> str:
    """Return a content-addressed key for one listwise inference request.

    Raw item IDs are included even though the prompt uses opaque candidate
    labels.  This prevents accidental reuse when two candidate groups happen to
    have identical public text.
    """

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "task_identity": task_identity,
        "cell": cell,
        "user_id": user_id,
        "candidate_group": candidate_group,
        "candidate_item_ids": list(candidate_item_ids),
        "prompt_messages": list(prompt_messages),
        "model": model,
        "request_config": request_config,
    }
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class GroupCheckpointStore:
    """Content-addressed, atomic store for completed listwise groups.

    Files are sharded by the first two key characters so a full three-seed run
    does not put tens of thousands of files in one directory.  Writes use a
    temporary file in the destination directory followed by ``os.replace``;
    readers therefore see either the previous complete record or the new
    complete record, never a partially written JSON document.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()

    def path_for_key(self, key: str) -> Path:
        _validate_key(key)
        return self.root / key[:2] / f"{key}.json"

    def load(
        self,
        key: str,
        *,
        expected_item_ids: Sequence[Any],
    ) -> CachedGroupScores | None:
        """Load and validate a completed group, returning ``None`` on a miss."""

        path = self.path_for_key(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointCorruptionError(
                f"Cannot parse checkpoint {path}"
            ) from error
        return _parse_checkpoint(
            payload,
            path=path,
            expected_key=key,
            expected_item_ids=expected_item_ids,
        )

    def save(
        self,
        key: str,
        *,
        item_ids: Sequence[Any],
        scores: Sequence[float],
        usage: Mapping[str, int] | None = None,
    ) -> CachedGroupScores:
        """Atomically persist one successful group and return its parsed record."""

        path = self.path_for_key(key)
        normalized_ids = [_json_safe(value) for value in item_ids]
        normalized_scores = [_finite_float(value) for value in scores]
        if len(normalized_ids) != len(normalized_scores):
            raise ValueError(
                "item_ids and scores must have the same number of values"
            )
        normalized_usage = _normalize_usage(usage)
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "key": key,
            "item_ids": normalized_ids,
            "scores": normalized_scores,
            "usage": normalized_usage,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{key}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(_canonical_json(payload))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

        loaded = self.load(key, expected_item_ids=item_ids)
        if loaded is None:  # pragma: no cover - os.replace made the file visible.
            raise RuntimeError(f"Checkpoint disappeared immediately after save: {path}")
        return loaded


def _parse_checkpoint(
    payload: Any,
    *,
    path: Path,
    expected_key: str,
    expected_item_ids: Sequence[Any],
) -> CachedGroupScores:
    if not isinstance(payload, dict):
        raise CheckpointCorruptionError(
            f"Checkpoint {path} is not a JSON object"
        )
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCorruptionError(
            f"Checkpoint {path} has an unsupported format version"
        )
    if payload.get("key") != expected_key:
        raise CheckpointCorruptionError(
            f"Checkpoint key mismatch in {path}"
        )

    actual_ids = payload.get("item_ids")
    expected_ids = [_json_safe(value) for value in expected_item_ids]
    if actual_ids != expected_ids:
        raise CheckpointCorruptionError(
            f"Candidate item order mismatch in {path}"
        )

    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, list) or len(raw_scores) != len(expected_ids):
        raise CheckpointCorruptionError(
            f"Invalid score vector in {path}"
        )
    try:
        scores = tuple(_finite_float(value) for value in raw_scores)
    except (TypeError, ValueError) as error:
        raise CheckpointCorruptionError(
            f"Invalid score value in {path}"
        ) from error

    raw_usage = payload.get("usage")
    try:
        usage = _normalize_usage(raw_usage)
    except (TypeError, ValueError) as error:
        raise CheckpointCorruptionError(
            f"Invalid token usage in {path}"
        ) from error
    created_at = payload.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at:
        raise CheckpointCorruptionError(
            f"Missing creation timestamp in {path}"
        )
    return CachedGroupScores(
        key=expected_key,
        item_ids=tuple(actual_ids),
        scores=scores,
        usage=usage,
        created_at_utc=created_at,
    )


def _normalize_usage(
    usage: Mapping[str, int] | None,
) -> dict[str, int] | None:
    if usage is None:
        return None
    if not isinstance(usage, Mapping):
        raise TypeError("usage must be a mapping or None")
    normalized: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field, 0)
        if isinstance(value, bool):
            raise ValueError(f"usage[{field!r}] must be a non-negative integer")
        integer = int(value)
        if integer != value or integer < 0:
            raise ValueError(f"usage[{field!r}] must be a non-negative integer")
        normalized[field] = integer
    return normalized


def _finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("checkpoint scores must be finite")
    return number


def _validate_key(key: str) -> None:
    if (
        len(key) != 64
        or any(character not in "0123456789abcdef" for character in key)
    ):
        raise ValueError("checkpoint key must be a lowercase SHA-256 hex digest")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Value is not JSON-serializable for checkpointing: {value!r}")
