from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    CappedUserInteractionCandidateSampler,
    ColdStartTask,
    ColdStartTaskBuilder,
    ColdUserHoldoutSplitter,
    MinUserInteractionsFilter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interactions(
    n_users: int = 5,
    interactions_per_user: int = 6,
) -> pd.DataFrame:
    """Toy interaction frame; each (user, timestamp) pair gets a unique item."""
    rows = []
    for u in range(1, n_users + 1):
        for t in range(1, interactions_per_user + 1):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "item_id": f"u{u}i{t}",
                    "timestamp": t,
                    "target_interact": 1,
                }
            )
    return pd.DataFrame(rows)


def _write_toy_canonical_dataset(
    tmp_path: Path,
    *,
    with_item_summaries: bool = False,
) -> CanonicalDataset:
    """Five users, each with 6 timestamped interactions on their own unique items."""
    root = tmp_path / "toy-cold-start"
    root.mkdir()

    users = pd.DataFrame(
        {"user_id": [f"u{u}" for u in range(1, 6)], "age": [25, 31, 19, 42, 28]}
    )
    items = pd.DataFrame(
        {
            "item_id": [f"u{u}i{t}" for u in range(1, 6) for t in range(1, 7)],
            "title": [
                f"User{u} Item{t}" for u in range(1, 6) for t in range(1, 7)
            ],
        }
    )
    if with_item_summaries:
        items["summary"] = [f"Summary for item {i}." for i in range(len(items))]
    interactions = _make_interactions(n_users=5, interactions_per_user=6)
    interactions["rating"] = 4  # used in history_context_columns tests

    users_path = root / "users.parquet"
    items_path = root / "items.parquet"
    interactions_path = root / "interactions.parquet"
    manifest_path = root / "manifest.json"

    users.to_parquet(users_path, index=False)
    items.to_parquet(items_path, index=False)
    interactions.to_parquet(interactions_path, index=False)
    manifest: dict[str, object] = {"dataset": "toy"}
    if with_item_summaries:
        manifest["item_enrichment"] = {
            "movie_summaries": {
                "enabled": True,
                "column": "summary",
                "source": {"sha256": "fixture-sha256"},
            }
        }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    return CanonicalDataset(
        name="toy",
        version="v1",
        root=root,
        users_path=users_path,
        items_path=items_path,
        interactions_path=interactions_path,
        manifest_path=manifest_path,
    )


def _splitter_all_cold(k: int = 3, seed: int = 0) -> ColdUserHoldoutSplitter:
    """Splitter with no warm users — useful for small two-user fixtures."""
    return ColdUserHoldoutSplitter(
        k=k,
        train_fraction=0.0,
        val_fraction=0.5,
        test_fraction=0.5,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# ColdUserHoldoutSplitter — user partitioning
# ---------------------------------------------------------------------------


def test_splitter_warm_cold_sets_cover_all_users_without_overlap() -> None:
    interactions = _make_interactions()
    split = ColdUserHoldoutSplitter(k=3, seed=0).split(interactions)

    all_cold = split.val_cold_user_ids | split.test_cold_user_ids
    all_users = set(interactions["user_id"])
    assert split.warm_user_ids | all_cold == all_users
    assert split.warm_user_ids.isdisjoint(all_cold)
    assert split.val_cold_user_ids.isdisjoint(split.test_cold_user_ids)


def test_splitter_no_cold_interaction_in_train() -> None:
    interactions = _make_interactions()
    split = ColdUserHoldoutSplitter(k=3, seed=0).split(interactions)

    all_cold = split.val_cold_user_ids | split.test_cold_user_ids
    assert split.train["user_id"].isin(all_cold).sum() == 0


def test_splitter_warm_users_absent_from_history_and_eval() -> None:
    interactions = _make_interactions()
    split = ColdUserHoldoutSplitter(k=3, seed=0).split(interactions)

    eval_rows = pd.concat([split.val, split.test])
    assert split.online_session_history["user_id"].isin(split.warm_user_ids).sum() == 0
    assert eval_rows["user_id"].isin(split.warm_user_ids).sum() == 0


# ---------------------------------------------------------------------------
# ColdUserHoldoutSplitter — temporal ordering (the critical property)
# ---------------------------------------------------------------------------


def test_splitter_history_timestamps_precede_eval_timestamps_for_every_cold_user() -> None:
    interactions = _make_interactions(n_users=10, interactions_per_user=8)
    split = ColdUserHoldoutSplitter(k=3, seed=0).split(interactions)

    eval_rows = pd.concat([split.val, split.test])
    all_cold = split.val_cold_user_ids | split.test_cold_user_ids

    for user_id in all_cold:
        history = split.online_session_history[
            split.online_session_history["user_id"] == user_id
        ]
        evl = eval_rows[eval_rows["user_id"] == user_id]
        assert len(history) > 0
        assert len(evl) > 0
        assert history["timestamp"].max() <= evl["timestamp"].min(), (
            f"Temporal violation for {user_id!r}: "
            f"history max={history['timestamp'].max()}, eval min={evl['timestamp'].min()}"
        )


def test_splitter_history_size_is_exactly_k_when_users_have_enough_interactions() -> None:
    # Every user has 8 interactions, k=3 → every cold user must have exactly 3
    # history items. Asserting <= 3 would pass even if history were empty.
    interactions = _make_interactions(n_users=10, interactions_per_user=8)
    split = ColdUserHoldoutSplitter(k=3, seed=0).split(interactions)

    history_sizes = split.online_session_history.groupby("user_id").size()
    assert (history_sizes == 3).all()


def test_splitter_history_contains_exactly_k_earliest_items() -> None:
    # Two cold users with known timestamps; we can assert exact item membership.
    u1 = pd.DataFrame(
        {
            "user_id": ["u1"] * 6,
            "item_id": ["iA", "iB", "iC", "iD", "iE", "iF"],
            "timestamp": [10, 20, 30, 40, 50, 60],
            "target_interact": [1] * 6,
        }
    )
    u2 = pd.DataFrame(
        {
            "user_id": ["u2"] * 6,
            "item_id": ["iG", "iH", "iI", "iJ", "iK", "iL"],
            "timestamp": [10, 20, 30, 40, 50, 60],
            "target_interact": [1] * 6,
        }
    )
    interactions = pd.concat([u1, u2], ignore_index=True)
    split = _splitter_all_cold(k=3).split(interactions)

    eval_rows = pd.concat([split.val, split.test])
    expected = {
        "u1": ({"iA", "iB", "iC"}, {"iD", "iE", "iF"}),
        "u2": ({"iG", "iH", "iI"}, {"iJ", "iK", "iL"}),
    }
    for user_id, (exp_history, exp_eval) in expected.items():
        actual_history = set(
            split.online_session_history[
                split.online_session_history["user_id"] == user_id
            ]["item_id"]
        )
        actual_eval = set(eval_rows[eval_rows["user_id"] == user_id]["item_id"])
        assert actual_history == exp_history, f"{user_id} history mismatch"
        assert actual_eval == exp_eval, f"{user_id} eval mismatch"


def test_splitter_breaks_timestamp_ties_by_item_id() -> None:
    # u1: two interactions at timestamp=1 (iA and iC); k=2.
    # After sorting by (timestamp, item_id): iA(1), iC(1), iD(2), iB(3).
    # History = {iA, iC}; eval = {iD, iB}.
    u1 = pd.DataFrame(
        {
            "user_id": ["u1"] * 4,
            "item_id": ["iC", "iA", "iD", "iB"],  # intentionally shuffled
            "timestamp": [1, 1, 2, 3],
            "target_interact": [1] * 4,
        }
    )
    u2 = pd.DataFrame(
        {
            "user_id": ["u2"] * 4,
            "item_id": ["iE", "iF", "iG", "iH"],
            "timestamp": [1, 2, 3, 4],
            "target_interact": [1] * 4,
        }
    )
    interactions = pd.concat([u1, u2], ignore_index=True)
    split = _splitter_all_cold(k=2).split(interactions)

    eval_rows = pd.concat([split.val, split.test])
    u1_history = set(
        split.online_session_history[
            split.online_session_history["user_id"] == "u1"
        ]["item_id"]
    )
    u1_eval = set(eval_rows[eval_rows["user_id"] == "u1"]["item_id"])
    assert u1_history == {"iA", "iC"}
    assert u1_eval == {"iD", "iB"}


def test_splitter_breaks_k_boundary_ties_by_item_id() -> None:
    # Three items share timestamp=1; with k=2 the cut falls inside the tie group.
    # Sorted by item_id: iA(t=1) < iB(t=1) < iC(t=1). Positions 0,1 → history;
    # position 2 → eval. This tests that tie-breaking determines the correct side
    # of the boundary, not just the order of all-history ties.
    u1 = pd.DataFrame(
        {
            "user_id": ["u1"] * 4,
            "item_id": ["iC", "iA", "iB", "iD"],  # intentionally shuffled
            "timestamp": [1, 1, 1, 2],
            "target_interact": [1] * 4,
        }
    )
    u2 = pd.DataFrame(
        {
            "user_id": ["u2"] * 4,
            "item_id": ["iE", "iF", "iG", "iH"],
            "timestamp": [1, 2, 3, 4],
            "target_interact": [1] * 4,
        }
    )
    interactions = pd.concat([u1, u2], ignore_index=True)
    split = _splitter_all_cold(k=2).split(interactions)

    eval_rows = pd.concat([split.val, split.test])
    u1_history = set(
        split.online_session_history[
            split.online_session_history["user_id"] == "u1"
        ]["item_id"]
    )
    u1_eval = set(eval_rows[eval_rows["user_id"] == "u1"]["item_id"])
    assert u1_history == {"iA", "iB"}, f"Got {u1_history}"
    assert u1_eval == {"iC", "iD"}, f"Got {u1_eval}"


# ---------------------------------------------------------------------------
# ColdUserHoldoutSplitter — drop behaviour
# ---------------------------------------------------------------------------


def test_splitter_drops_users_with_zero_postk_interactions() -> None:
    # u1: exactly k=3 interactions → 0 post-k rows → dropped.
    # u2: 5 interactions → 2 post-k rows → kept.
    interactions = pd.DataFrame(
        {
            "user_id": ["u1"] * 3 + ["u2"] * 5,
            "item_id": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "timestamp": [1, 2, 3, 1, 2, 3, 4, 5],
            "target_interact": [1] * 8,
        }
    )
    split = _splitter_all_cold(k=3).split(interactions)

    all_cold = split.val_cold_user_ids | split.test_cold_user_ids
    assert "u1" not in all_cold
    assert "u2" in all_cold
    assert split.dropped_val_cold_count + split.dropped_test_cold_count == 1


# ---------------------------------------------------------------------------
# ColdUserHoldoutSplitter — validation errors
# ---------------------------------------------------------------------------


def test_splitter_raises_without_timestamp_column() -> None:
    interactions = pd.DataFrame(
        {"user_id": ["u1", "u2"], "item_id": ["i1", "i2"], "target_interact": [1, 1]}
    )
    with pytest.raises(ValueError, match="timestamp"):
        ColdUserHoldoutSplitter(k=1).split(interactions)


def test_splitter_raises_if_k_is_zero() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        ColdUserHoldoutSplitter(k=0)


def test_splitter_raises_if_fractions_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ColdUserHoldoutSplitter(
            k=1, train_fraction=0.5, val_fraction=0.3, test_fraction=0.3
        )


# ---------------------------------------------------------------------------
# ColdUserHoldoutSplitter — determinism
# ---------------------------------------------------------------------------


def test_splitter_is_deterministic() -> None:
    interactions = _make_interactions(n_users=20, interactions_per_user=5)
    split_a = ColdUserHoldoutSplitter(k=2, seed=7).split(interactions)
    split_b = ColdUserHoldoutSplitter(k=2, seed=7).split(interactions)

    assert split_a.warm_user_ids == split_b.warm_user_ids
    assert split_a.val_cold_user_ids == split_b.val_cold_user_ids
    assert split_a.test_cold_user_ids == split_b.test_cold_user_ids


# ---------------------------------------------------------------------------
# ColdStartTaskBuilder
# ---------------------------------------------------------------------------


def test_cold_start_task_builder_column_schema(tmp_path: Path) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)

    task = ColdStartTaskBuilder(
        name="toy-cold-start",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=ColdUserHoldoutSplitter(k=3, seed=0),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1, total_items=4, seed=0
        ),
    ).build(dataset)

    expected_columns = [
        "user_id",
        "item_id",
        "user_age",
        "item_title",
        "target",
        "sampled",
        "candidate_group",
    ]
    assert list(task.train.columns) == expected_columns
    assert list(task.val.columns) == expected_columns
    assert list(task.test.columns) == expected_columns
    # online_session_history must share the same schema as train so scorers can
    # call fit() on either frame interchangeably.
    assert list(task.online_session_history.columns) == expected_columns


def test_cold_start_task_builder_returns_cold_start_task(tmp_path: Path) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)
    task = ColdStartTaskBuilder(
        name="toy",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=ColdUserHoldoutSplitter(k=3, seed=0),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1, total_items=4, seed=0
        ),
    ).build(dataset)
    assert isinstance(task, ColdStartTask)
    assert task.k == 3


def test_cold_start_task_builder_propagates_movie_summaries(tmp_path: Path) -> None:
    dataset = _write_toy_canonical_dataset(
        tmp_path,
        with_item_summaries=True,
    )
    task = ColdStartTaskBuilder(
        name="toy",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=ColdUserHoldoutSplitter(k=3, seed=0),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1,
            total_items=4,
            seed=0,
        ),
    ).build(dataset)

    for frame in (
        task.train,
        task.val,
        task.test,
        task.online_session_history,
    ):
        assert "item_summary" in frame.columns
    assert task.manifest["item_enrichment"] == {
        "movie_summaries": {
            "enabled": True,
            "canonical_column": "summary",
            "task_column": "item_summary",
            "source_sha256": "fixture-sha256",
        }
    }


def test_cold_start_task_builder_cold_users_absent_from_train(tmp_path: Path) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)
    task = ColdStartTaskBuilder(
        name="toy",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=ColdUserHoldoutSplitter(k=3, seed=0),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1, total_items=4, seed=0
        ),
    ).build(dataset)

    cold_users = set(task.val["user_id"]) | set(task.test["user_id"])
    assert task.train["user_id"].isin(cold_users).sum() == 0


def test_cold_start_task_builder_all_observed_items_excluded_from_negatives(
    tmp_path: Path,
) -> None:
    # Negatives must exclude ALL items a cold user has ever interacted with:
    # both pre-k history items and post-k positive items. The bug this guards
    # against: if all_cold_interactions were set to only history (instead of
    # concat(history, val, test)), post-k items could leak into negatives.
    dataset = _write_toy_canonical_dataset(tmp_path)
    task = ColdStartTaskBuilder(
        name="toy",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=ColdUserHoldoutSplitter(k=3, seed=0),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1, total_items=4, seed=0
        ),
    ).build(dataset)

    history_pairs = set(
        zip(
            task.online_session_history["user_id"],
            task.online_session_history["item_id"],
            strict=True,
        )
    )
    eval_rows = pd.concat([task.val, task.test])
    positive_pairs = set(
        zip(
            eval_rows.loc[~eval_rows["sampled"].astype(bool), "user_id"],
            eval_rows.loc[~eval_rows["sampled"].astype(bool), "item_id"],
            strict=True,
        )
    )
    sampled_pairs = set(
        zip(
            eval_rows.loc[eval_rows["sampled"].astype(bool), "user_id"],
            eval_rows.loc[eval_rows["sampled"].astype(bool), "item_id"],
            strict=True,
        )
    )
    all_observed_pairs = history_pairs | positive_pairs
    assert sampled_pairs.isdisjoint(all_observed_pairs)


def test_cold_start_task_builder_history_context_present_in_history_null_in_candidates(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)
    task = ColdStartTaskBuilder(
        name="toy",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=ColdUserHoldoutSplitter(k=3, seed=0),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1, total_items=4, seed=0
        ),
        history_context_columns=("rating",),
    ).build(dataset)

    # History and train rows carry the real rating; candidate rows must not.
    assert task.online_session_history["rating"].notna().all()
    assert task.train["rating"].notna().all()
    assert task.val["rating"].isna().all()
    assert task.test["rating"].isna().all()


def test_cold_start_task_builder_manifest_records_k_and_user_counts(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)
    task = ColdStartTaskBuilder(
        name="toy",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=ColdUserHoldoutSplitter(k=3, seed=0),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1, total_items=4, seed=0
        ),
    ).build(dataset)

    assert task.manifest["k"] == 3
    users = task.manifest["users"]
    assert {"warm", "val_cold", "test_cold", "dropped_val_cold", "dropped_test_cold"} <= users.keys()
    # All 5 toy users pass MinUserInteractionsFilter(4) and have post-k rows.
    assert users["warm"] + users["val_cold"] + users["test_cold"] == 5
    assert users["dropped_val_cold"] == 0
    assert users["dropped_test_cold"] == 0
    assert task.manifest["rows"]["online_session_history"] == len(task.online_session_history)


def test_cold_start_task_builder_online_session_history_has_exactly_k_rows_per_cold_user(
    tmp_path: Path,
) -> None:
    # Each toy user has 6 interactions; with k=3 every cold user must have
    # exactly 3 history rows. Also verifies the splitter's temporal result is
    # correctly wired through the builder (not mixed up with train or eval).
    dataset = _write_toy_canonical_dataset(tmp_path)
    task = ColdStartTaskBuilder(
        name="toy",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=ColdUserHoldoutSplitter(k=3, seed=0),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1, total_items=4, seed=0
        ),
    ).build(dataset)

    cold_users = set(task.val["user_id"]) | set(task.test["user_id"])
    history_sizes = task.online_session_history.groupby("user_id").size()
    assert set(history_sizes.index) == cold_users
    assert (history_sizes == 3).all()
