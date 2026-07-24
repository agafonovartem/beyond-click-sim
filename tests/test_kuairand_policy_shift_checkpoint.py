from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.ood.kuairand_policy_shift.checkpoint import (
    CheckpointCorruptionError,
    GroupCheckpointStore,
    build_group_checkpoint_key,
)


def _key(**overrides: object) -> str:
    values: dict[str, object] = {
        "task_identity": {
            "dataset": "kuairand-pure",
            "source_seed": 42,
            "candidate_seed": 43,
        },
        "cell": {
            "history_policy": "standard",
            "target_policy": "random",
            "history_control": "observed",
        },
        "user_id": 17,
        "candidate_group": "test_random_u17_c3",
        "candidate_item_ids": [101, 202, 303],
        "prompt_messages": [
            {"role": "system", "content": "Rank the candidates."},
            {"role": "user", "content": "C1: 猫\nC2: 音乐\nC3: 新闻"},
        ],
        "model": "gpt-5.4-mini",
        "request_config": {
            "max_completion_tokens": 16,
            "reasoning_effort": "none",
        },
    }
    values.update(overrides)
    return build_group_checkpoint_key(**values)  # type: ignore[arg-type]


def test_group_checkpoint_key_is_canonical_and_content_addressed() -> None:
    first = _key()
    reordered = _key(
        task_identity={
            "candidate_seed": 43,
            "source_seed": 42,
            "dataset": "kuairand-pure",
        },
        request_config={
            "reasoning_effort": "none",
            "max_completion_tokens": 16,
        },
    )

    assert first == reordered
    assert len(first) == 64
    assert _key(model="gpt-5.4") != first
    assert _key(user_id=18) != first
    assert _key(candidate_item_ids=[101, 303, 202]) != first
    assert _key(
        cell={
            "history_policy": "random",
            "target_policy": "random",
            "history_control": "observed",
        }
    ) != first
    assert _key(
        request_config={
            "max_completion_tokens": 24,
            "reasoning_effort": "none",
        }
    ) != first
    assert _key(
        prompt_messages=[
            {"role": "system", "content": "Rank the candidates."},
            {"role": "user", "content": "changed prompt"},
        ]
    ) != first


def test_atomic_group_checkpoint_round_trip_and_cache_miss(tmp_path) -> None:
    store = GroupCheckpointStore(tmp_path / "groups")
    key = _key()

    assert store.load(key, expected_item_ids=[101, 202, 303]) is None

    saved = store.save(
        key,
        item_ids=[101, 202, 303],
        scores=[3, 2, 1],
        usage={
            "prompt_tokens": 123,
            "completion_tokens": 6,
            "total_tokens": 129,
        },
    )
    loaded = store.load(key, expected_item_ids=[101, 202, 303])

    assert loaded == saved
    assert loaded is not None
    assert loaded.scores == (3.0, 2.0, 1.0)
    assert loaded.usage == {
        "prompt_tokens": 123,
        "completion_tokens": 6,
        "total_tokens": 129,
    }
    assert store.path_for_key(key).parent.name == key[:2]
    assert list((tmp_path / "groups").rglob("*.tmp")) == []


def test_checkpoint_rejects_candidate_order_mismatch(tmp_path) -> None:
    store = GroupCheckpointStore(tmp_path)
    key = _key()
    store.save(
        key,
        item_ids=[101, 202, 303],
        scores=[3, 2, 1],
    )

    with pytest.raises(
        CheckpointCorruptionError,
        match="Candidate item order mismatch",
    ):
        store.load(key, expected_item_ids=[101, 303, 202])


def test_checkpoint_rejects_corrupt_or_nonfinite_completed_record(tmp_path) -> None:
    store = GroupCheckpointStore(tmp_path)
    key = _key()
    path = store.path_for_key(key)
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(CheckpointCorruptionError, match="Cannot parse"):
        store.load(key, expected_item_ids=[101, 202, 303])

    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "key": key,
                "item_ids": [101, 202, 303],
                "scores": [3, float("nan"), 1],
                "usage": None,
                "created_at_utc": "2026-07-24T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointCorruptionError, match="Invalid score value"):
        store.load(key, expected_item_ids=[101, 202, 303])


def test_unfinished_temporary_file_is_never_treated_as_checkpoint(tmp_path) -> None:
    store = GroupCheckpointStore(tmp_path)
    key = _key()
    final_path = store.path_for_key(key)
    final_path.parent.mkdir(parents=True)
    (final_path.parent / f".{key}.interrupted.tmp").write_text(
        '{"partial":',
        encoding="utf-8",
    )

    assert store.load(key, expected_item_ids=[101, 202, 303]) is None
