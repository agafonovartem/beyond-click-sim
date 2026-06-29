from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    AlignmentInteractionTaskBuilder,
    CappedUserInteractionCandidateSampler,
    MinUserInteractionsFilter,
    NonInteractionCandidateSampler,
    RandomFractionSplitter,
    SampleUsersFilter,
    SequentialDatasetFilter,
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


def test_alignment_interaction_task_builder_serializes_sequential_filter(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)

    task = AlignmentInteractionTaskBuilder(
        name="toy-interaction-alignment-sampled-users",
        dataset_filter=SequentialDatasetFilter(
            [
                MinUserInteractionsFilter(min_interactions=4),
                SampleUsersFilter(n_users=1, seed=0),
            ]
        ),
        splitter=RandomFractionSplitter(
            train_fraction=0.75,
            val_fraction=0.0,
            test_fraction=0.25,
            seed=0,
        ),
        sampler=NonInteractionCandidateSampler(negative_ratio=1, seed=0),
    ).build(dataset)

    assert task.manifest["users"] == 1
    assert json.loads(json.dumps(task.manifest)) == task.manifest
    assert task.manifest["filter"] == {
        "class": "SequentialDatasetFilter",
        "filters": [
            {
                "class": "MinUserInteractionsFilter",
                "min_interactions": 4,
                "user_column": "user_id",
            },
            {
                "class": "SampleUsersFilter",
                "n_users": 1,
                "seed": 0,
                "user_column": "user_id",
            },
        ],
    }


def test_alignment_interaction_task_builder_keeps_val_test_negatives_disjoint(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)

    task = AlignmentInteractionTaskBuilder(
        name="toy-interaction-alignment-disjoint-val-test-negatives",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=RandomFractionSplitter(
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            seed=0,
        ),
        sampler=NonInteractionCandidateSampler(negative_ratio=2, seed=0),
    ).build(dataset)

    assert _sampled_pairs(task.val).isdisjoint(_sampled_pairs(task.test))


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


def test_non_interaction_sampler_respects_excluded_pairs() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": ["r1"],
            "user_id": ["u1"],
            "item_id": ["i1"],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 9)]})
    sampler = NonInteractionCandidateSampler(negative_ratio=2, seed=0)
    excluded_pairs = {("u1", f"i{idx}") for idx in range(2, 7)}

    sampled = sampler.sample(
        positives,
        interactions=interactions,
        items=items,
        excluded_pairs=excluded_pairs,
    )

    assert _sampled_pairs(sampled) == {("u1", "i7"), ("u1", "i8")}


def test_capped_user_candidate_sampler_respects_excluded_pairs() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": ["r1", "r2"],
            "user_id": ["u1", "u1"],
            "item_id": ["i1", "i2"],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 9)]})
    sampler = CappedUserInteractionCandidateSampler(
        negative_ratio=1,
        total_items=4,
        seed=0,
    )
    excluded_pairs = {("u1", f"i{idx}") for idx in range(3, 7)}

    sampled = sampler.sample(
        positives,
        interactions=interactions,
        items=items,
        excluded_pairs=excluded_pairs,
    )

    assert _sampled_pairs(sampled) == {("u1", "i7"), ("u1", "i8")}


def test_non_interaction_sampler_shuffles_rows_within_candidate_group() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": ["r1"],
            "user_id": ["u1"],
            "item_id": ["i1"],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 20)]})
    sampler = NonInteractionCandidateSampler(negative_ratio=9, seed=0)

    sampled = sampler.sample(positives, interactions=interactions, items=items)

    assert sampled["candidate_group"].nunique() == 1
    assert sampled["target"].tolist() != [1] + [0] * 9


def test_capped_user_candidate_sampler_chunks_all_user_positives() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(1, 8)],
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2", "u2"],
            "item_id": ["i1", "i2", "i3", "i4", "i5", "i6", "i7"],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 30)]})
    sampler = CappedUserInteractionCandidateSampler(
        negative_ratio=1,
        total_items=4,
        seed=0,
    )

    sampled = sampler.sample(positives, interactions=interactions, items=items)

    groups_per_user = (
        sampled[["user_id", "candidate_group"]]
        .drop_duplicates()
        .groupby("user_id")
        .size()
        .to_dict()
    )
    assert groups_per_user == {"u1": 3, "u2": 1}

    output_positives = sampled[sampled["target"].eq(1)]
    assert {
        user_id: set(rows["item_id"])
        for user_id, rows in output_positives.groupby("user_id")
    } == {
        "u1": {"i1", "i2", "i3", "i4", "i5"},
        "u2": {"i6", "i7"},
    }

    positives_per_group = sampled.groupby("candidate_group")["target"].sum()
    sampled_per_group = sampled.groupby("candidate_group")["sampled"].sum()
    assert (sampled.groupby("candidate_group").size() <= 4).all()
    assert sampled_per_group.eq(positives_per_group).all()

    observed_items = {
        user_id: set(rows["item_id"])
        for user_id, rows in interactions.groupby("user_id")
    }
    negatives = sampled[sampled["sampled"]]
    for row in negatives.itertuples(index=False):
        assert row.item_id not in observed_items[row.user_id]


def test_capped_user_candidate_sampler_agent4rec_like_ratio_uses_all_positives() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(1, 11)],
            "user_id": ["u1"] * 10,
            "item_id": [f"i{idx}" for idx in range(1, 11)],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 101)]})
    sampler = CappedUserInteractionCandidateSampler(
        negative_ratio=9,
        total_items=20,
        seed=0,
    )

    sampled = sampler.sample(positives, interactions=interactions, items=items)

    assert sampled["candidate_group"].nunique() == 5
    assert sampled.groupby("candidate_group").size().to_dict() == {
        group_id: 20
        for group_id in sampled["candidate_group"].unique()
    }
    assert sampled.groupby("candidate_group")["target"].sum().to_dict() == {
        group_id: 2
        for group_id in sampled["candidate_group"].unique()
    }
    assert sampled.groupby("candidate_group")["sampled"].sum().to_dict() == {
        group_id: 18
        for group_id in sampled["candidate_group"].unique()
    }
    assert set(sampled[sampled["target"].eq(1)]["item_id"]) == set(positives["item_id"])


def test_capped_user_candidate_sampler_limits_eval_users_before_chunking() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(1, 16)],
            "user_id": ["u1"] * 5 + ["u2"] * 5 + ["u3"] * 5,
            "item_id": [f"i{idx}" for idx in range(1, 16)],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 80)]})
    sampler = CappedUserInteractionCandidateSampler(
        negative_ratio=1,
        total_items=4,
        max_eval_users=2,
        seed=0,
    )

    sampled = sampler.sample(positives, interactions=interactions, items=items)
    sampled_reversed = sampler.sample(
        positives.iloc[::-1].reset_index(drop=True),
        interactions=interactions,
        items=items,
    )

    selected_users = set(sampled["user_id"])
    assert len(selected_users) == 2
    assert selected_users == set(sampled_reversed["user_id"])

    output_positives = sampled[sampled["target"].eq(1)]
    for user_id in selected_users:
        expected_items = set(positives[positives["user_id"].eq(user_id)]["item_id"])
        actual_items = set(
            output_positives[output_positives["user_id"].eq(user_id)]["item_id"]
        )
        assert actual_items == expected_items

    groups_per_user = (
        sampled[["user_id", "candidate_group"]]
        .drop_duplicates()
        .groupby("user_id")
        .size()
    )
    assert groups_per_user.eq(3).all()


def test_capped_user_candidate_sampler_limits_candidate_groups_per_user() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(1, 18)],
            "user_id": ["u1"] * 10 + ["u2"] * 7,
            "item_id": [f"i{idx}" for idx in range(1, 18)],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 80)]})
    sampler = CappedUserInteractionCandidateSampler(
        negative_ratio=1,
        total_items=4,
        max_candidate_groups_per_user=2,
        seed=0,
    )

    sampled = sampler.sample(positives, interactions=interactions, items=items)
    sampled_reversed = sampler.sample(
        positives.iloc[::-1].reset_index(drop=True),
        interactions=interactions,
        items=items,
    )

    groups_per_user = (
        sampled[["user_id", "candidate_group"]]
        .drop_duplicates()
        .groupby("user_id")
        .size()
        .to_dict()
    )
    assert groups_per_user == {"u1": 2, "u2": 2}
    assert sampled["target"].sum() == 8
    assert (
        sampled[["user_id", "candidate_group", "item_id", "target", "sampled"]]
        .sort_values(["user_id", "candidate_group", "item_id"])
        .reset_index(drop=True)
        .equals(
            sampled_reversed[
                ["user_id", "candidate_group", "item_id", "target", "sampled"]
            ]
            .sort_values(["user_id", "candidate_group", "item_id"])
            .reset_index(drop=True)
        )
    )


def test_capped_user_candidate_sampler_is_stable_to_positive_row_order() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(1, 7)],
            "user_id": ["u1"] * 6,
            "item_id": [f"i{idx}" for idx in range(1, 7)],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 50)]})
    sampler = CappedUserInteractionCandidateSampler(
        negative_ratio=1,
        total_items=4,
        seed=0,
    )

    first = sampler.sample(positives, interactions=interactions, items=items)
    second = sampler.sample(
        positives.iloc[::-1].reset_index(drop=True),
        interactions=interactions,
        items=items,
    )

    def rows_by_group(frame: pd.DataFrame) -> dict[str, set[tuple[str, int, bool]]]:
        return {
            group_id: set(
                zip(
                    rows["item_id"],
                    rows["target"],
                    rows["sampled"],
                    strict=True,
                )
            )
            for group_id, rows in frame.groupby("candidate_group")
        }

    assert rows_by_group(first) == rows_by_group(second)


def test_capped_user_candidate_sampler_shuffles_rows_within_candidate_group() -> None:
    positives = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(1, 3)],
            "user_id": ["u1"] * 2,
            "item_id": ["i1", "i2"],
        }
    )
    interactions = positives[["interaction_id", "user_id", "item_id"]].copy()
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 50)]})
    sampler = CappedUserInteractionCandidateSampler(
        negative_ratio=9,
        total_items=20,
        seed=0,
    )

    sampled = sampler.sample(positives, interactions=interactions, items=items)

    assert sampled["candidate_group"].nunique() == 1
    assert sampled["target"].tolist() != [1, 1] + [0] * 18


def test_alignment_task_manifest_records_candidate_group_cap(tmp_path: Path) -> None:
    dataset = _write_toy_canonical_dataset(tmp_path)

    task = AlignmentInteractionTaskBuilder(
        name="toy-interaction-cg1",
        dataset_filter=MinUserInteractionsFilter(min_interactions=4),
        splitter=RandomFractionSplitter(
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            seed=0,
        ),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1,
            total_items=4,
            max_eval_users=2,
            max_candidate_groups_per_user=1,
            seed=0,
        ),
    ).build(dataset)

    assert task.manifest["sampler"]["max_eval_users"] == 2
    assert task.manifest["sampler"]["max_candidate_groups_per_user"] == 1


def _sampled_pairs(frame: pd.DataFrame) -> set[tuple[object, object]]:
    sampled = frame[frame["sampled"].astype(bool)]
    return set(zip(sampled["user_id"], sampled["item_id"], strict=True))


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
