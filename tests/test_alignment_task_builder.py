from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    AlignmentInteractionTaskBuilder,
    MinUserInteractionsFilter,
    NonInteractionCandidateSampler,
    RandomFractionSplitter,
)


def test_alignment_interaction_task_builder_creates_candidate_sets(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)

    task = AlignmentInteractionTaskBuilder(
        name="toy-interaction-alignment",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=RandomFractionSplitter(
            train_fraction=0.75,
            val_fraction=0.0,
            test_fraction=0.25,
            seed=0,
        ),
        sampler=NonInteractionCandidateSampler(negative_ratio=2, seed=0),
    ).build(dataset)

    expected_columns = [
        "user_id",
        "item_id",
        "user_age",
        "user_segment",
        "item_title",
        "item_genre",
        "target",
        "sampled",
        "candidate_group",
    ]
    assert list(task.train.columns) == expected_columns
    assert list(task.val.columns) == expected_columns
    assert list(task.test.columns) == expected_columns

    assert len(task.train) == 6
    assert task.val.empty
    assert len(task.test) == 6

    assert set(task.train["user_id"]) == {"u1", "u2"}
    assert task.train["target"].eq(1).all()
    assert task.train["sampled"].eq(False).all()
    assert task.train["candidate_group"].isna().all()

    group_sizes = task.test.groupby("candidate_group").size()
    assert group_sizes.tolist() == [3, 3]

    targets_per_group = task.test.groupby("candidate_group")["target"].sum()
    assert targets_per_group.tolist() == [1, 1]

    sampled_per_group = task.test.groupby("candidate_group")["sampled"].sum()
    assert sampled_per_group.tolist() == [2, 2]

    observed_items = {
        user_id: set(rows["item_id"])
        for user_id, rows in dataset.load_interactions().groupby("user_id")
    }
    negatives = task.test[task.test["sampled"]]
    for row in negatives.itertuples(index=False):
        assert row.item_id not in observed_items[row.user_id]

    assert task.schema.target_column == "target"
    assert task.schema.sampled_column == "sampled"
    assert task.schema.candidate_group_column == "candidate_group"
    assert task.schema.history_context_columns == ()
    assert task.schema.feature_columns == (
        "user_age",
        "user_segment",
        "item_title",
        "item_genre",
    )
    assert task.manifest["target_source_column"] == "target_interact"
    assert task.manifest["history_context_columns"] == []
    assert task.manifest["filter"]["min_interactions"] == 4
    assert task.manifest["splitter"]["train_fraction"] == 0.75
    assert task.manifest["splitter"]["test_fraction"] == 0.25
    assert task.manifest["sampler"]["negative_ratio"] == 2
    assert task.manifest["rows"] == {"train": 6, "val": 0, "test": 6}


def test_alignment_interaction_task_builder_keeps_history_context_train_only(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)

    task = AlignmentInteractionTaskBuilder(
        name="toy-interaction-alignment-with-history-ratings",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=RandomFractionSplitter(
            train_fraction=0.75,
            val_fraction=0.0,
            test_fraction=0.25,
            seed=0,
        ),
        sampler=NonInteractionCandidateSampler(negative_ratio=2, seed=0),
        history_context_columns=("rating",),
    ).build(dataset)

    expected_columns = [
        "user_id",
        "item_id",
        "user_age",
        "user_segment",
        "item_title",
        "item_genre",
        "rating",
        "target",
        "sampled",
        "candidate_group",
    ]
    assert list(task.train.columns) == expected_columns
    assert list(task.val.columns) == expected_columns
    assert list(task.test.columns) == expected_columns

    assert task.train["rating"].notna().all()
    assert task.val["rating"].isna().all()
    assert task.test["rating"].isna().all()
    assert task.schema.history_context_columns == ("rating",)
    assert task.manifest["history_context_columns"] == ["rating"]


def test_non_interaction_sampler_is_stable_per_candidate_group() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": ["r1", "r2"],
            "user_id": ["u1", "u1"],
            "item_id": ["i1", "i2"],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 9)]})
    sampler = NonInteractionCandidateSampler(negative_ratio=2, seed=0)

    first = sampler.sample(positives, interactions=interactions, items=items)
    second = sampler.sample(
        positives.iloc[::-1].reset_index(drop=True),
        interactions=interactions,
        items=items,
    )

    def negatives_by_group(frame: pd.DataFrame) -> dict[str, set[str]]:
        negatives = frame[frame["sampled"]]
        return {
            group_id: set(rows["item_id"])
            for group_id, rows in negatives.groupby("candidate_group")
        }

    assert negatives_by_group(first) == negatives_by_group(second)


def _write_toy_canonical_dataset(tmp_path: Path) -> CanonicalDataset:
    root = tmp_path / "toy-canonical"
    root.mkdir()

    users = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "age": [25, 31, 19],
            "segment": ["a", "b", "c"],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": [f"i{idx}" for idx in range(1, 9)],
            "title": [f"Item {idx}" for idx in range(1, 9)],
            "genre": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
        }
    )
    interactions = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(1, 11)],
            "user_id": [
                "u1",
                "u1",
                "u1",
                "u1",
                "u2",
                "u2",
                "u2",
                "u2",
                "u3",
                "u3",
            ],
            "item_id": [
                "i1",
                "i2",
                "i3",
                "i4",
                "i1",
                "i5",
                "i6",
                "i7",
                "i1",
                "i2",
            ],
            "rating": [5, 4, 3, 2, 5, 4, 3, 2, 5, 4],
            "target_interact": [1] * 10,
        }
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
