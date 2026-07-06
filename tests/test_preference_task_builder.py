from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    CappedObservedPreferenceCandidateSampler,
    MinUserInteractionsFilter,
    PreferencePredictionTaskBuilder,
    SplitFrames,
    Splitter,
)


def test_preference_prediction_task_builder_creates_observed_candidate_sets(
    tmp_path: Path,
) -> None:
    dataset = _write_toy_preference_dataset(tmp_path)

    task = PreferencePredictionTaskBuilder(
        name="toy-preference",
        target_source_column="target_like_ge4",
        dataset_filter=MinUserInteractionsFilter(min_interactions=1),
        splitter=_InteractionIdSplitter(
            train_ids=("r1", "r2", "r3", "r4"),
            val_ids=("r5", "r6"),
            test_ids=("r7", "r8", "r9", "r10"),
        ),
        sampler=CappedObservedPreferenceCandidateSampler(
            negative_ratio=1,
            total_items=4,
            seed=0,
            target_source_column="target_like_ge4",
        ),
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
        "candidate_group",
    ]
    assert list(task.train.columns) == expected_columns
    assert list(task.val.columns) == expected_columns
    assert list(task.test.columns) == expected_columns

    assert len(task.train) == 4
    assert task.train["target"].tolist() == [1, 0, 1, 0]
    assert task.train["candidate_group"].isna().all()
    assert task.train["rating"].notna().all()
    assert task.val["rating"].isna().all()
    assert task.test["rating"].isna().all()

    assert len(task.val) == 2
    assert len(task.test) == 4
    assert task.val["candidate_group"].nunique() == 1
    assert task.test["candidate_group"].nunique() == 1
    assert task.val.groupby("candidate_group")["target"].sum().tolist() == [1]
    assert task.test.groupby("candidate_group")["target"].sum().tolist() == [2]
    assert set(task.test["item_id"]) == {"i7", "i8", "i9", "i10"}

    assert "sampled" not in task.train.columns
    assert task.schema.sampled_column is None
    assert task.schema.candidate_group_column == "candidate_group"
    assert task.schema.history_context_columns == ("rating",)
    assert task.manifest["target_source_column"] == "target_like_ge4"
    assert task.manifest["sampled_column"] is None
    assert task.manifest["sampler"]["negative_ratio"] == 1
    assert task.manifest["sampler"]["total_items"] == 4
    assert task.manifest["rows"] == {"train": 4, "val": 2, "test": 4}


def test_capped_observed_preference_sampler_chunks_with_observed_negatives() -> None:
    heldout = pd.DataFrame(
        {
            "user_id": ["u1"] * 11,
            "item_id": [f"i{idx}" for idx in range(1, 12)],
            "target_like_ge4": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        }
    )
    sampler = CappedObservedPreferenceCandidateSampler(
        negative_ratio=1,
        total_items=4,
        seed=0,
        target_source_column="target_like_ge4",
    )

    sampled = sampler.sample(
        heldout,
        interactions=heldout,
        items=pd.DataFrame({"item_id": heldout["item_id"]}),
    )

    assert sampled["candidate_group"].nunique() == 3
    assert (sampled.groupby("candidate_group").size() <= 4).all()
    positives_per_group = sampled.groupby("candidate_group")["target"].sum()
    negatives_per_group = sampled.groupby("candidate_group")["target"].apply(
        lambda values: int(values.eq(0).sum())
    )
    assert negatives_per_group.eq(positives_per_group).all()
    assert set(sampled[sampled["target"].eq(1)]["item_id"]) == {
        "i1",
        "i2",
        "i3",
        "i4",
        "i5",
    }
    assert set(sampled[sampled["target"].eq(0)]["item_id"]).issubset(
        {f"i{idx}" for idx in range(6, 12)}
    )
    assert "sampled" not in sampled.columns


def test_capped_observed_preference_sampler_skips_incomplete_ratio_groups() -> None:
    heldout = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "target_like_ge4": [1, 1, 0],
        }
    )
    sampler = CappedObservedPreferenceCandidateSampler(
        negative_ratio=1,
        total_items=4,
        seed=0,
        target_source_column="target_like_ge4",
    )

    sampled = sampler.sample(
        heldout,
        interactions=heldout,
        items=pd.DataFrame({"item_id": heldout["item_id"]}),
    )

    assert sampled.empty
    assert list(sampled.columns) == [
        "user_id",
        "item_id",
        "target",
        "candidate_group",
    ]


def test_capped_observed_preference_sampler_is_stable_to_row_order() -> None:
    heldout = pd.DataFrame(
        {
            "user_id": ["u1"] * 8,
            "item_id": [f"i{idx}" for idx in range(1, 9)],
            "target_like_ge4": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    sampler = CappedObservedPreferenceCandidateSampler(
        negative_ratio=1,
        total_items=4,
        seed=0,
        target_source_column="target_like_ge4",
    )

    first = sampler.sample(
        heldout,
        interactions=heldout,
        items=pd.DataFrame({"item_id": heldout["item_id"]}),
    )
    second = sampler.sample(
        heldout.iloc[::-1].reset_index(drop=True),
        interactions=heldout,
        items=pd.DataFrame({"item_id": heldout["item_id"]}),
    )

    def rows_by_group(frame: pd.DataFrame) -> dict[str, set[tuple[str, int]]]:
        return {
            group_id: set(zip(rows["item_id"], rows["target"], strict=True))
            for group_id, rows in frame.groupby("candidate_group")
        }

    assert rows_by_group(first) == rows_by_group(second)


class _InteractionIdSplitter(Splitter):
    def __init__(
        self,
        *,
        train_ids: tuple[str, ...],
        val_ids: tuple[str, ...],
        test_ids: tuple[str, ...],
    ) -> None:
        super().__init__(seed=0)
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

    def split(self, interactions: pd.DataFrame) -> SplitFrames:
        return SplitFrames(
            train=self._select(interactions, self.train_ids),
            val=self._select(interactions, self.val_ids),
            test=self._select(interactions, self.test_ids),
        )

    @staticmethod
    def _select(interactions: pd.DataFrame, ids: tuple[str, ...]) -> pd.DataFrame:
        return interactions[
            interactions["interaction_id"].isin(ids)
        ].reset_index(drop=True)


def _write_toy_preference_dataset(tmp_path: Path) -> CanonicalDataset:
    root = tmp_path / "toy-preference-canonical"
    root.mkdir()

    users = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "age": [25, 31],
            "segment": ["a", "b"],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": [f"i{idx}" for idx in range(1, 11)],
            "title": [f"Item {idx}" for idx in range(1, 11)],
            "genre": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5"],
        }
    )
    interactions = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(1, 11)],
            "user_id": ["u1"] * 10,
            "item_id": [f"i{idx}" for idx in range(1, 11)],
            "rating": [5, 2, 4, 1, 5, 2, 5, 4, 2, 1],
            "target_interact": [1] * 10,
            "target_like_ge4": [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
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
