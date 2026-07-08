from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    MinUserInteractionsFilter,
    PolicyRankingTaskBuilder,
    PostSplitUserSampler,
    PopularityPolicy,
    RandomFractionSplitter,
    RandomPolicy,
)


def _write_toy_dataset(root: Path, *, history_context_column: str | None = None) -> CanonicalDataset:
    root.mkdir(parents=True, exist_ok=True)
    users = pd.DataFrame({"user_id": [f"u{i}" for i in range(8)]})
    items = pd.DataFrame({"item_id": [f"i{i}" for i in range(12)]})
    rows = []
    for u in range(8):
        for offset in range(6):
            row = {
                "user_id": f"u{u}",
                "item_id": f"i{(u * 3 + offset) % 12}",
                "target_interact": 1,
            }
            if history_context_column is not None:
                row[history_context_column] = float(offset + 1)
            rows.append(row)
    interactions = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "item_id"])
    users_path = root / "users.parquet"
    items_path = root / "items.parquet"
    interactions_path = root / "interactions.parquet"
    manifest_path = root / "manifest.json"
    users.to_parquet(users_path, index=False)
    items.to_parquet(items_path, index=False)
    interactions.to_parquet(interactions_path, index=False)
    manifest_path.write_text(json.dumps({"dataset": "toy"}), encoding="utf-8")
    return CanonicalDataset(
        name="toy",
        version="v1",
        root=root,
        users_path=users_path,
        items_path=items_path,
        interactions_path=interactions_path,
        manifest_path=manifest_path,
    )


def _make_builder(*, policies=None, history_context_columns=(), eval_sampler=None):
    if policies is None:
        policies = [RandomPolicy(k=3, seed=0), PopularityPolicy(k=3, seed=0)]
    return PolicyRankingTaskBuilder(
        name="toy_policy_ranking_seed0",
        dataset_filter=MinUserInteractionsFilter(min_interactions=3),
        splitter=RandomFractionSplitter(
            train_fraction=0.7,
            val_fraction=0.0,
            test_fraction=0.3,
            seed=0,
        ),
        policies=policies,
        history_context_columns=history_context_columns,
        eval_sampler=eval_sampler,
    )


def test_val_is_always_empty(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds")
    task = _make_builder().build(dataset)
    assert len(task.val) == 0


def test_rejects_row_capped_eval_sampler() -> None:
    with pytest.raises(ValueError, match="row-capped"):
        _make_builder(
            eval_sampler=PostSplitUserSampler(
                n_users=2,
                seed=0,
                max_rows_per_user=1,
            )
        )


def test_test_has_policy_and_rank_columns(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds")
    task = _make_builder().build(dataset)
    assert "policy" in task.test.columns
    assert "rank" in task.test.columns


def test_train_has_no_policy_or_rank_columns(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds")
    task = _make_builder().build(dataset)
    assert "policy" not in task.train.columns
    assert "rank" not in task.train.columns


def test_manifest_protocol_is_policy_ranking(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds")
    task = _make_builder().build(dataset)
    assert task.manifest["protocol"] == "policy_ranking"


def test_manifest_contains_two_policies(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds")
    task = _make_builder().build(dataset)
    assert len(task.manifest["policies"]) == 2


def test_manifest_contains_policy_recommendation_metrics(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds")
    task = _make_builder().build(dataset)
    metrics = task.manifest["policy_recommendation_metrics"]
    assert metrics["protocol"] == "held_out_recommendation_eval"
    assert metrics["aggregation_headline"] == "macro_by_user_group_mean"
    assert metrics["ks"] == [1, 3, 5, 10]
    policy_entries = metrics["policies"]
    assert len(policy_entries) == 2
    for entry in policy_entries:
        assert "mean_hit_rate" in entry
        assert "ranking" in entry


def test_target_values_are_binary(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds")
    task = _make_builder().build(dataset)
    assert set(task.test["target"].unique()).issubset({0, 1})


def test_recommended_items_not_in_train(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds")
    task = _make_builder().build(dataset)
    train_pairs = set(zip(task.train["user_id"], task.train["item_id"]))
    for _, row in task.test.iterrows():
        assert (row["user_id"], row["item_id"]) not in train_pairs


def test_history_context_cols_masked_to_na_in_test(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds", history_context_column="rating")
    task = _make_builder(history_context_columns=("rating",)).build(dataset)
    assert task.test["rating"].isna().all()
    # Train should have real rating values.
    assert task.train["rating"].notna().any()


def test_both_policy_names_appear_in_test(tmp_path):
    dataset = _write_toy_dataset(tmp_path / "ds")
    task = _make_builder().build(dataset)
    assert "RandomPolicy" in task.test["policy"].unique()
    assert "PopularityPolicy" in task.test["policy"].unique()


def test_empty_policies_raises():
    with pytest.raises(ValueError):
        PolicyRankingTaskBuilder(
            name="bad",
            dataset_filter=MinUserInteractionsFilter(min_interactions=3),
            splitter=RandomFractionSplitter(0.7, 0.0, 0.3, seed=0),
            policies=[],
        )
