from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    ColdStartTask,
    ColdStartTaskBuilder,
    ColdUserHoldoutSplitter,
    MinUserInteractionsFilter,
    PerTargetCandidateSampler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interactions(
    n_users: int = 5,
    interactions_per_user: int = 6,
    n_items: int = 20,
) -> pd.DataFrame:
    """Toy interaction frame with a shared item catalog.

    Users stagger across the shared catalog so warm users cover items that
    cold users have not observed (needed for warm-only negative sampling).
    """
    rows = []
    for u in range(1, n_users + 1):
        for t in range(1, interactions_per_user + 1):
            item_idx = ((u - 1) + (t - 1)) % n_items + 1
            rows.append(
                {
                    "user_id": f"u{u}",
                    "item_id": f"i{item_idx}",
                    "timestamp": t,
                    "target_interact": 1,
                }
            )
    return pd.DataFrame(rows)


def _write_toy_canonical_dataset(
    tmp_path: Path,
    *,
    with_item_summaries: bool = False,
    n_users: int = 5,
    interactions_per_user: int = 6,
    n_items: int = 20,
) -> CanonicalDataset:
    """Users share a catalog so warm-item coverage can be enforced."""
    root = tmp_path / "toy-cold-start"
    root.mkdir()

    users = pd.DataFrame(
        {
            "user_id": [f"u{u}" for u in range(1, n_users + 1)],
            "age": [20 + u for u in range(n_users)],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": [f"i{t}" for t in range(1, n_items + 1)],
            "title": [f"Item{t}" for t in range(1, n_items + 1)],
        }
    )
    if with_item_summaries:
        items["summary"] = [f"Summary for item {i}." for i in range(len(items))]
    interactions = _make_interactions(
        n_users=n_users,
        interactions_per_user=interactions_per_user,
        n_items=n_items,
    )
    interactions["rating"] = 4  # used in history_context_columns tests
    interactions["target_like_ge4"] = 1  # all toy interactions are liked

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


def _builder(
    *,
    name: str = "toy",
    k: int = 3,
    pool_k: int = 3,
    targets_per_user: int = 2,
    max_eval_users: int | None = None,
    negative_ratio: int = 1,
    sampler_seed: int = 0,
    strategy: str = "uniform",
    splitter: ColdUserHoldoutSplitter | None = None,
    history_context_columns: tuple[str, ...] = (),
) -> ColdStartTaskBuilder:
    return ColdStartTaskBuilder(
        name=name,
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=splitter or ColdUserHoldoutSplitter(k=k, seed=0),
        sampler=PerTargetCandidateSampler(
            negative_ratio=negative_ratio, strategy=strategy, seed=sampler_seed
        ),
        targets_per_user=targets_per_user,
        max_eval_users=max_eval_users,
        pool_k=pool_k,
        split_seed=0,
        history_context_columns=history_context_columns,
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

    task = _builder(name="toy-cold-start").build(dataset)

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
    task = _builder().build(dataset)
    assert isinstance(task, ColdStartTask)
    assert task.k == 3


def test_cold_start_task_builder_propagates_movie_summaries(tmp_path: Path) -> None:
    dataset = _write_toy_canonical_dataset(
        tmp_path,
        with_item_summaries=True,
    )
    task = _builder().build(dataset)

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
    task = _builder().build(dataset)

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
    task = _builder().build(dataset)

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
    task = _builder(history_context_columns=("rating",)).build(dataset)

    # History and train rows carry the real rating; candidate rows must not.
    assert task.online_session_history["rating"].notna().all()
    assert task.train["rating"].notna().all()
    assert task.val["rating"].isna().all()
    assert task.test["rating"].isna().all()


def test_cold_start_task_builder_manifest_records_k_and_user_counts(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)
    task = _builder().build(dataset)

    assert task.manifest["k"] == 3
    assert task.manifest["targets_per_user"] == 2
    assert task.manifest["target_source_column"] == "target_like_ge4"
    assert task.manifest["split_seed"] == 0
    assert task.manifest["pool_k"] == 3
    users = task.manifest["users"]
    assert {
        "warm",
        "val_cold",
        "test_cold",
        "evaluated_val",
        "evaluated_test",
        "dropped_val_cold",
        "dropped_test_cold",
        "dropped_zero_qualifying_targets_val",
        "dropped_zero_qualifying_targets_test",
        "dropped_nonwarm_history_val",
        "dropped_nonwarm_history_test",
    } <= users.keys()
    # All 5 toy users pass MinUserInteractionsFilter(4) and have post-k rows.
    assert users["warm"] + users["val_cold"] + users["test_cold"] == 5
    assert users["dropped_val_cold"] == 0
    assert users["dropped_test_cold"] == 0
    # Shared catalog makes history warm for every cold user under this fixture.
    assert users["evaluated_val"] == users["val_cold"]
    assert users["evaluated_test"] == users["test_cold"]
    assert users["dropped_zero_qualifying_targets_val"] == 0
    assert users["dropped_zero_qualifying_targets_test"] == 0
    assert users["dropped_nonwarm_history_val"] == 0
    assert users["dropped_nonwarm_history_test"] == 0
    coverage = task.manifest["warm_item_coverage"]
    assert coverage["enforced"] is True
    assert coverage["policy"] == "strict"
    assert coverage["warm_items"] > 0
    assert task.manifest["targets_per_user_distribution"]["val"]["min"] >= 1
    assert task.manifest["targets_per_user_distribution"]["test"]["min"] >= 1
    assert task.manifest["rows"]["online_session_history"] == len(
        task.online_session_history
    )


def test_cold_start_task_builder_online_session_history_has_exactly_k_rows_per_cold_user(
    tmp_path: Path,
) -> None:
    # Each toy user has 6 interactions; with k=3 every cold user must have
    # exactly 3 history rows. Also verifies the splitter's temporal result is
    # correctly wired through the builder (not mixed up with train or eval).
    dataset = _write_toy_canonical_dataset(tmp_path)
    task = _builder().build(dataset)

    cold_users = set(task.val["user_id"]) | set(task.test["user_id"])
    history_sizes = task.online_session_history.groupby("user_id").size()
    assert set(history_sizes.index) == cold_users
    assert (history_sizes == 3).all()


def test_cold_start_task_builder_candidate_groups_are_target_keyed(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)
    task = _builder(negative_ratio=2).build(dataset)

    for frame in (task.val, task.test):
        for group_id, group in frame.groupby("candidate_group"):
            assert len(group) == 3  # 1 positive + 2 negatives
            assert (group["target"] == 1).sum() == 1
            positive = group.loc[group["target"] == 1].iloc[0]
            assert group_id == (
                f"candidate:user:{positive['user_id']}:item:{positive['item_id']}"
            )


# ---------------------------------------------------------------------------
# Target frame semantics
# ---------------------------------------------------------------------------


def _write_rated_toy_dataset(tmp_path: Path, *, n_users: int = 4) -> CanonicalDataset:
    """Users with mixed ratings so target_like_ge4 selection is non-trivial.

    Pattern users share items i1..i6:
      u1: ratings [5,2,5,2,5,2] -> qualifying t1,t3,t5
      u2: ratings all 2         -> zero qualifying (dropped at frame stage)
      u3: ratings all 4         -> all qualify
      u4: ratings [3,3,5,5,5,5] -> qualifying t3..t6
    Extra pattern users (n_users > 4) repeat the u1 pattern.

    Filler users w1..w20 each interact only with one catalog item (4 times, to
    pass MinUserInteractionsFilter). That warms i1..i20 when the filler is in
    train; if a filler is cold they still have unused warm items for negatives.
    """
    root = tmp_path / "toy-rated"
    root.mkdir()

    rating_patterns = {
        "u1": [5, 2, 5, 2, 5, 2],
        "u2": [2, 2, 2, 2, 2, 2],
        "u3": [4, 4, 4, 4, 4, 4],
        "u4": [3, 3, 5, 5, 5, 5],
    }
    rows = []
    for u in range(1, n_users + 1):
        user_id = f"u{u}"
        ratings = rating_patterns.get(user_id, [5, 2, 5, 2, 5, 2])
        for t, rating in enumerate(ratings, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": f"i{t}",
                    "timestamp": t,
                    "rating": rating,
                    "target_interact": 1,
                    "target_like_ge4": int(rating >= 4),
                }
            )
    # Per-item warm densifiers: each covers one catalog item.
    for t in range(1, 21):
        for repeat in range(1, 5):
            rows.append(
                {
                    "user_id": f"w{t}",
                    "item_id": f"i{t}",
                    "timestamp": repeat,
                    "rating": 4,
                    "target_interact": 1,
                    "target_like_ge4": 1,
                }
            )
    interactions = pd.DataFrame(rows)

    user_ids = [f"u{u}" for u in range(1, n_users + 1)] + [
        f"w{t}" for t in range(1, 21)
    ]
    users = pd.DataFrame({"user_id": user_ids})
    items = pd.DataFrame({"item_id": [f"i{t}" for t in range(1, 21)]})

    users.to_parquet(root / "users.parquet", index=False)
    items.to_parquet(root / "items.parquet", index=False)
    interactions.to_parquet(root / "interactions.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"dataset": "toy-rated"}), encoding="utf-8"
    )
    return CanonicalDataset(
        name="toy-rated",
        version="v1",
        root=root,
        users_path=root / "users.parquet",
        items_path=root / "items.parquet",
        interactions_path=root / "interactions.parquet",
        manifest_path=root / "manifest.json",
    )


def _positive_items_by_user(task: ColdStartTask) -> dict[str, list[str]]:
    eval_rows = pd.concat([task.val, task.test])
    positives = eval_rows[eval_rows["target"] == 1]
    return {
        user_id: group["item_id"].tolist()
        for user_id, group in positives.groupby("user_id", sort=False)
    }


def _pattern_cold_splitter(k: int) -> ColdUserHoldoutSplitter:
    """Keep pattern users u1..u4 cold while densifiers keep i1..i6 (+ pool) warm.

    With 4 pattern + 20 densifier users, train_fraction=0.6 and seed=96 places
    all four pattern users in the cold set and keeps densifiers for i1..i6 warm
    (verified against stable_sample_values).
    """

    return ColdUserHoldoutSplitter(
        k=k,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=96,
    )


def test_target_frame_selects_first_c_qualifying_items_in_temporal_order(
    tmp_path: Path,
) -> None:
    dataset = _write_rated_toy_dataset(tmp_path)
    task = _builder(
        k=1,
        pool_k=1,
        targets_per_user=2,
        splitter=_pattern_cold_splitter(k=1),
    ).build(dataset)

    positives = _positive_items_by_user(task)
    # u1: post-k = t2..t6, qualifying t3,t5 -> first two
    assert positives["u1"] == ["i3", "i5"]
    # u3: all qualify -> first two post-k items
    assert positives["u3"] == ["i2", "i3"]
    # u4: qualifying t3..t6 -> first two
    assert positives["u4"] == ["i3", "i4"]
    # u2 has zero qualifying items and is dropped entirely
    assert "u2" not in positives


def test_target_frame_counts_zero_qualifying_user_drops(tmp_path: Path) -> None:
    dataset = _write_rated_toy_dataset(tmp_path)
    task = _builder(
        k=1,
        pool_k=1,
        targets_per_user=2,
        splitter=_pattern_cold_splitter(k=1),
    ).build(dataset)

    users = task.manifest["users"]
    total_zero_qualifying = (
        users["dropped_zero_qualifying_targets_val"]
        + users["dropped_zero_qualifying_targets_test"]
    )
    assert total_zero_qualifying == 1  # exactly u2
    # splitter-level drops remain zero: u2 has post-k rows, just none qualifying
    assert users["dropped_val_cold"] == 0
    assert users["dropped_test_cold"] == 0


def test_target_frame_identical_across_m_strategy_and_sampling_seed(
    tmp_path: Path,
) -> None:
    dataset = _write_rated_toy_dataset(tmp_path)
    splitter = _pattern_cold_splitter(k=1)
    base = _builder(k=1, pool_k=1, splitter=splitter).build(dataset)
    different_m = _builder(
        k=1, pool_k=1, negative_ratio=3, splitter=_pattern_cold_splitter(k=1)
    ).build(dataset)
    different_seed = _builder(
        k=1, pool_k=1, sampler_seed=2, splitter=_pattern_cold_splitter(k=1)
    ).build(dataset)
    different_strategy = _builder(
        k=1,
        pool_k=1,
        strategy="popularity_matched",
        splitter=_pattern_cold_splitter(k=1),
    ).build(dataset)

    for other in (different_m, different_seed, different_strategy):
        assert other.manifest["target_frame_sha256"] == base.manifest[
            "target_frame_sha256"
        ]
        assert _positive_items_by_user(other) == _positive_items_by_user(base)

    # negatives may differ, targets must not
    base_negatives = set(
        pd.concat([base.val, base.test])
        .loc[lambda frame: frame["target"] == 0, "item_id"]
    )
    seed_negatives = set(
        pd.concat([different_seed.val, different_seed.test])
        .loc[lambda frame: frame["target"] == 0, "item_id"]
    )
    assert base_negatives != seed_negatives


def test_eval_users_fixed_across_k_targets_move(tmp_path: Path) -> None:
    dataset = _write_rated_toy_dataset(tmp_path, n_users=10)
    splitter_k1 = ColdUserHoldoutSplitter(
        k=1, train_fraction=0.5, val_fraction=0.25, test_fraction=0.25, seed=0
    )
    splitter_k3 = ColdUserHoldoutSplitter(
        k=3, train_fraction=0.5, val_fraction=0.25, test_fraction=0.25, seed=0
    )
    task_k1 = _builder(
        k=1, pool_k=3, max_eval_users=10, splitter=splitter_k1
    ).build(dataset)
    task_k3 = _builder(
        k=3, pool_k=3, max_eval_users=10, splitter=splitter_k3
    ).build(dataset)

    users_k1 = task_k1.manifest["users"]
    users_k3 = task_k3.manifest["users"]
    assert (
        users_k1["evaluated_val"] == users_k3["evaluated_val"]
        and users_k1["evaluated_test"] == users_k3["evaluated_test"]
    )
    assert set(task_k1.val["user_id"]) == set(task_k3.val["user_id"])
    assert set(task_k1.test["user_id"]) == set(task_k3.test["user_id"])
    # targets themselves move with k
    assert (
        task_k1.manifest["target_frame_sha256"]
        != task_k3.manifest["target_frame_sha256"]
    )


def test_builder_rejects_pool_k_smaller_than_k() -> None:
    with pytest.raises(ValueError, match="pool_k"):
        _builder(k=3, pool_k=1)


def test_item_rating_stats_computed_from_warm_train_only(tmp_path: Path) -> None:
    """Cold users' ratings must never enter item_rating_mean: the enrichment
    uses split.train (warm users) only. Cold users rate every item 5, warm
    users rate every item 1 — any leak shifts the means upward."""
    root = tmp_path / "toy-item-stats"
    root.mkdir()

    shared_items = [f"i{t}" for t in range(1, 7)]
    negative_pool_items = [f"i{t}" for t in range(7, 21)]
    catalog_items = shared_items + negative_pool_items
    splitter = ColdUserHoldoutSplitter(
        k=2, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, seed=96
    )

    rows = []
    # Pattern users: only i1..i6 so densifiers can supply the negative pool.
    for u in range(1, 5):
        for t, item_id in enumerate(shared_items, start=1):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "item_id": item_id,
                    "timestamp": t,
                    "rating": 1,
                    "target_like_ge4": 1,
                }
            )
    # Per-item densifiers cover the full catalog without observing every item.
    for t, item_id in enumerate(catalog_items, start=1):
        for repeat in range(1, 5):
            rows.append(
                {
                    "user_id": f"w{t}",
                    "item_id": item_id,
                    "timestamp": repeat,
                    "rating": 1,
                    "target_like_ge4": 1,
                }
            )
    interactions = pd.DataFrame(rows)

    # Warm users rate 1, cold users rate 5 (split first so we know who is cold).
    split = splitter.split(interactions)
    cold_users = split.val_cold_user_ids | split.test_cold_user_ids
    interactions.loc[interactions["user_id"].isin(cold_users), "rating"] = 5

    users = pd.DataFrame(
        {
            "user_id": [f"u{u}" for u in range(1, 5)]
            + [f"w{t}" for t in range(1, len(catalog_items) + 1)]
        }
    )
    items_df = pd.DataFrame({"item_id": catalog_items})
    users.to_parquet(root / "users.parquet", index=False)
    items_df.to_parquet(root / "items.parquet", index=False)
    interactions.to_parquet(root / "interactions.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"dataset": "toy-item-stats"}), encoding="utf-8"
    )
    dataset = CanonicalDataset(
        name="toy-item-stats",
        version="v1",
        root=root,
        users_path=root / "users.parquet",
        items_path=root / "items.parquet",
        interactions_path=root / "interactions.parquet",
        manifest_path=root / "manifest.json",
    )

    from beyond_click_sim.tasks import TrainItemRatingStatistics

    task = ColdStartTaskBuilder(
        name="toy-item-stats",
        dataset_filter=MinUserInteractionsFilter(min_interactions=1),
        splitter=splitter,
        sampler=PerTargetCandidateSampler(negative_ratio=1, seed=0),
        targets_per_user=1,
        max_eval_users=None,
        pool_k=2,
        split_seed=0,
        item_feature_builder=TrainItemRatingStatistics(value_column="rating"),
    ).build(dataset)

    warm_means = (
        interactions[~interactions["user_id"].isin(cold_users)]
        .groupby("item_id")["rating"]
        .mean()
    )
    assert (warm_means == 1.0).all()
    test_features = task.test.drop_duplicates("item_id").set_index("item_id")
    assert len(test_features) > 0
    for item_id in test_features.index:
        mean = test_features.loc[item_id, "item_rating_mean"]
        assert item_id in warm_means.index, f"{item_id} is not a warm-train item"
        assert mean == 1.0, f"{item_id}: item_rating_mean leaked cold-user ratings"


def test_item_stats_enrichment_preserves_eval_candidates_and_order(
    tmp_path: Path,
) -> None:
    from beyond_click_sim.tasks import TrainItemRatingStatistics

    dataset = _write_rated_toy_dataset(tmp_path, n_users=10)
    plain = _builder(
        name="plain",
        k=3,
        pool_k=3,
        targets_per_user=2,
        negative_ratio=1,
        sampler_seed=0,
        strategy="uniform",
    ).build(dataset)
    enriched = ColdStartTaskBuilder(
        name="item-stats",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=ColdUserHoldoutSplitter(k=3, seed=0),
        sampler=PerTargetCandidateSampler(
            negative_ratio=1, strategy="uniform", seed=0
        ),
        targets_per_user=2,
        max_eval_users=None,
        pool_k=3,
        split_seed=0,
        item_feature_builder=TrainItemRatingStatistics(value_column="rating"),
    ).build(dataset)

    identity_columns = [
        "user_id",
        "item_id",
        "candidate_group",
        "sampled",
        "target",
    ]
    for split_name in ("val", "test"):
        pd.testing.assert_frame_equal(
            getattr(plain, split_name)[identity_columns].reset_index(drop=True),
            getattr(enriched, split_name)[identity_columns].reset_index(drop=True),
        )


def test_strict_warm_item_coverage_for_history_targets_and_negatives(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path, n_users=8, n_items=30)
    task = _builder(k=2, pool_k=2, negative_ratio=2).build(dataset)

    warm_items = set(task.train["item_id"])
    history_items = set(task.online_session_history["item_id"])
    eval_rows = pd.concat([task.val, task.test], ignore_index=True)
    positive_items = set(eval_rows.loc[eval_rows["target"] == 1, "item_id"])
    negative_items = set(eval_rows.loc[eval_rows["target"] == 0, "item_id"])

    assert history_items <= warm_items
    assert positive_items <= warm_items
    assert negative_items <= warm_items
    assert task.manifest["warm_item_coverage"]["enforced"] is True
    assert task.manifest["warm_item_coverage"]["policy"] == "strict"


def test_nonwarm_history_users_are_dropped(tmp_path: Path) -> None:
    """A cold user whose profile contains a train-unseen item is not evaluated."""
    root = tmp_path / "toy-nonwarm-history"
    root.mkdir()

    rows = []
    # Per-item densifiers warm i1..i10.
    for t in range(1, 11):
        for repeat in range(1, 5):
            rows.append(
                {
                    "user_id": f"w{t}",
                    "item_id": f"i{t}",
                    "timestamp": repeat,
                    "rating": 4,
                    "target_like_ge4": 1,
                }
            )
    rows.extend(
        [
            {
                "user_id": "c1",
                "item_id": "x1",  # never in warm train
                "timestamp": 1,
                "rating": 4,
                "target_like_ge4": 1,
            },
            {
                "user_id": "c1",
                "item_id": "i2",
                "timestamp": 2,
                "rating": 4,
                "target_like_ge4": 1,
            },
            {
                "user_id": "c1",
                "item_id": "i3",
                "timestamp": 3,
                "rating": 4,
                "target_like_ge4": 1,
            },
            {
                "user_id": "c1",
                "item_id": "i4",
                "timestamp": 4,
                "rating": 4,
                "target_like_ge4": 1,
            },
        ]
    )
    # Extra cold-eligible user with fully warm history so the task is non-empty.
    for t in range(1, 7):
        rows.append(
            {
                "user_id": "c2",
                "item_id": f"i{t}",
                "timestamp": t,
                "rating": 4,
                "target_like_ge4": 1,
            }
        )
    interactions = pd.DataFrame(rows)
    users = pd.DataFrame(
        {"user_id": [f"w{t}" for t in range(1, 11)] + ["c1", "c2"]}
    )
    items = pd.DataFrame(
        {"item_id": [f"i{t}" for t in range(1, 11)] + ["x1"]}
    )
    users.to_parquet(root / "users.parquet", index=False)
    items.to_parquet(root / "items.parquet", index=False)
    interactions.to_parquet(root / "interactions.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"dataset": "toy-nonwarm-history"}), encoding="utf-8"
    )
    dataset = CanonicalDataset(
        name="toy-nonwarm-history",
        version="v1",
        root=root,
        users_path=root / "users.parquet",
        items_path=root / "items.parquet",
        interactions_path=root / "interactions.parquet",
        manifest_path=root / "manifest.json",
    )

    splitter = None
    cold = set()
    for seed in range(50):
        candidate = ColdUserHoldoutSplitter(
            k=2,
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=seed,
        )
        split = candidate.split(interactions)
        cold = split.val_cold_user_ids | split.test_cold_user_ids
        warm_items = set(split.train["item_id"])
        if "c1" in cold and "x1" not in warm_items and "c2" in cold:
            # c2 needs warm history items present in train.
            if {"i1", "i2"} <= warm_items:
                splitter = candidate
                break
    assert splitter is not None, "could not find a split with c1 cold and warm i1/i2"

    task = ColdStartTaskBuilder(
        name="toy-nonwarm-history",
        dataset_filter=MinUserInteractionsFilter(min_interactions=1),
        splitter=splitter,
        sampler=PerTargetCandidateSampler(negative_ratio=1, seed=0),
        targets_per_user=1,
        max_eval_users=None,
        pool_k=2,
        split_seed=0,
    ).build(dataset)

    evaluated = set(task.val["user_id"]) | set(task.test["user_id"])
    assert "c1" not in evaluated
    assert (
        task.manifest["users"]["dropped_nonwarm_history_val"]
        + task.manifest["users"]["dropped_nonwarm_history_test"]
        >= 1
    )
