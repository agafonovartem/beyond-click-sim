from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    MinUserInteractionsFilter,
    PostSplitUserSampler,
    RandomFractionSplitter,
    RegressionPredictionTaskBuilder,
)


def test_regression_task_builder_builds_observed_only_eval_user_task(
    tmp_path: Path,
) -> None:
    dataset = _write_regression_canonical_dataset(tmp_path)

    task = RegressionPredictionTaskBuilder(
        name="toy_rating_eval_users2_seed0",
        dataset_filter=MinUserInteractionsFilter(min_interactions=10),
        splitter=RandomFractionSplitter(
            train_fraction=0.7,
            val_fraction=0.1,
            test_fraction=0.2,
            seed=0,
        ),
        target_source_column="target_rating",
        history_context_columns=("rating",),
        eval_sampler=PostSplitUserSampler(n_users=2, seed=0),
    ).build(dataset)

    expected_columns = [
        "user_id",
        "item_id",
        "user_segment",
        "item_title",
        "rating",
        "target",
    ]
    assert list(task.train.columns) == expected_columns
    assert list(task.val.columns) == expected_columns
    assert list(task.test.columns) == expected_columns
    assert "candidate_group" not in task.test.columns
    assert "sampled" not in task.test.columns

    assert task.schema.candidate_group_column is None
    assert task.schema.sampled_column is None
    assert task.schema.history_context_columns == ("rating",)
    assert task.schema.feature_columns == ("user_segment", "item_title")

    assert task.train["user_id"].nunique() == 4
    assert task.val["user_id"].nunique() == 2
    assert task.test["user_id"].nunique() == 2
    assert len(task.train) == 24
    assert len(task.val) == 4
    assert len(task.test) == 4

    assert task.train["rating"].notna().all()
    assert task.val["rating"].isna().all()
    assert task.test["rating"].isna().all()
    assert task.train["target"].between(1, 5).all()
    assert task.val["target"].between(1, 5).all()
    assert task.test["target"].between(1, 5).all()

    manifest = task.manifest
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["protocol"] == "regression"
    assert manifest["target_source_column"] == "target_rating"
    assert manifest["sampler"] is None
    assert manifest["eval_sampler"] == {
        "class": "PostSplitUserSampler",
        "n_users": 2,
        "seed": 0,
        "user_column": "user_id",
        "require_train_history": True,
    }
    assert manifest["candidate_group_column"] is None
    assert manifest["sampled_column"] is None
    assert manifest["eval_user_selection"]["kind"] == "post_split"
    assert manifest["eval_user_selection"]["test"]["eligible_users"] == 4
    assert manifest["eval_user_selection"]["test"]["selected_users"] == 2
    assert manifest["rows"] == {"train": 24, "val": 4, "test": 4}
    assert manifest["users"] == {"filtered": 4, "train": 4, "val": 2, "test": 2}


def _write_regression_canonical_dataset(tmp_path: Path) -> CanonicalDataset:
    root = tmp_path / "toy-regression-canonical"
    root.mkdir()

    users = pd.DataFrame(
        {
            "user_id": [f"u{idx}" for idx in range(1, 5)],
            "segment": ["a", "b", "c", "d"],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": [f"i{idx}" for idx in range(1, 11)],
            "title": [f"Item {idx}" for idx in range(1, 11)],
        }
    )
    interactions = pd.DataFrame(
        [
            {
                "interaction_id": f"r{user_idx}-{item_idx}",
                "user_id": f"u{user_idx}",
                "item_id": f"i{item_idx}",
                "rating": ((user_idx + item_idx) % 5) + 1,
                "target_rating": ((user_idx + item_idx) % 5) + 1,
            }
            for user_idx in range(1, 5)
            for item_idx in range(1, 11)
        ]
    )

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
