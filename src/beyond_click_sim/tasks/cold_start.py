from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks._sampling import stable_sample_values
from beyond_click_sim.tasks.base import (
    CandidateSampler,
    DatasetFilter,
    ItemFeatureBuilder,
    Task,
    TaskBuilder,
    TaskSchema,
)


@dataclass(frozen=True)
class ColdStartSplitFrames:
    """User-partitioned cold-start split with temporal online session history.

    `val` and `test` contain raw post-k positive interaction rows — no negatives.
    Candidate sampling is the task builder's responsibility.
    `online_session_history` contains the first k temporal interactions per cold
    user; these rows are not in train, val, or test.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    online_session_history: pd.DataFrame
    warm_user_ids: frozenset
    val_cold_user_ids: frozenset
    test_cold_user_ids: frozenset
    dropped_val_cold_count: int = 0
    dropped_test_cold_count: int = 0


class ColdUserHoldoutSplitter:
    """Partition users into warm and cold sets; temporally split cold users at position k.

    No cold user or any of their interactions appears in train. The first k
    temporal interactions per cold user (sorted by timestamp, ties broken by
    item_id) form `online_session_history`. Post-k interactions are the
    evaluation target in val/test. Cold users with zero post-k interactions are
    dropped from evaluation entirely.

    Requires a timestamp column — raises ValueError for datasets without one
    (e.g. Steam, which has no wall-clock timestamps in iteration 1).
    """

    def __init__(
        self,
        k: int,
        train_fraction: float = 0.7,
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
        seed: int = 0,
        timestamp_column: str = "timestamp",
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> None:
        if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-9:
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must sum to 1.0, "
                f"got {train_fraction + val_fraction + test_fraction}."
            )
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")
        self.k = k
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.timestamp_column = timestamp_column
        self.user_column = user_column
        self.item_column = item_column

    def split(self, interactions: pd.DataFrame) -> ColdStartSplitFrames:
        if self.timestamp_column not in interactions.columns:
            raise ValueError(
                f"ColdUserHoldoutSplitter requires a timestamp column "
                f"({self.timestamp_column!r}) for temporal online_session_history "
                f"ordering. Steam interactions have no timestamps; this splitter "
                f"is not supported for Steam in iteration 1."
            )

        # Step 1: partition users (random, seed-controlled)
        all_users = sorted(interactions[self.user_column].unique(), key=repr)
        n_users = len(all_users)
        n_cold = round((self.val_fraction + self.test_fraction) * n_users)
        cold_users = stable_sample_values(all_users, n=n_cold, seed=self.seed)

        n_val_cold = round(
            self.val_fraction / (self.val_fraction + self.test_fraction) * n_cold
        )
        val_cold_users = stable_sample_values(
            cold_users, n=n_val_cold, seed=self.seed + 1
        )
        val_cold_set = set(val_cold_users)
        cold_set = set(cold_users)
        test_cold_set = cold_set - val_cold_set
        warm_set = set(all_users) - cold_set

        train = (
            interactions[interactions[self.user_column].isin(warm_set)]
            .copy()
            .reset_index(drop=True)
        )

        # Step 2: temporal split within each cold user (mandatory, no random ordering)
        cold_interactions = (
            interactions[interactions[self.user_column].isin(cold_set)]
            .copy()
            .sort_values(
                [self.user_column, self.timestamp_column, self.item_column],
                ascending=True,
            )
        )
        rank = cold_interactions.groupby(self.user_column, sort=False).cumcount()

        history_mask = rank < self.k
        online_session_history = (
            cold_interactions[history_mask].copy().reset_index(drop=True)
        )

        post_k = cold_interactions[~history_mask].copy()
        val_mask = post_k[self.user_column].isin(val_cold_set)
        val_raw = post_k[val_mask].reset_index(drop=True)
        test_raw = post_k[~val_mask].reset_index(drop=True)

        # Drop cold users with zero post-k rows; record counts for the manifest
        val_users_with_postk = set(val_raw[self.user_column].unique())
        test_users_with_postk = set(test_raw[self.user_column].unique())
        dropped_val = len(val_cold_set - val_users_with_postk)
        dropped_test = len(test_cold_set - test_users_with_postk)

        active_cold = val_users_with_postk | test_users_with_postk
        online_session_history = (
            online_session_history[
                online_session_history[self.user_column].isin(active_cold)
            ]
            .copy()
            .reset_index(drop=True)
        )

        return ColdStartSplitFrames(
            train=train,
            val=val_raw,
            test=test_raw,
            online_session_history=online_session_history,
            warm_user_ids=frozenset(warm_set),
            val_cold_user_ids=frozenset(val_users_with_postk),
            test_cold_user_ids=frozenset(test_users_with_postk),
            dropped_val_cold_count=dropped_val,
            dropped_test_cold_count=dropped_test,
        )


@dataclass
class ColdStartTask(Task):
    """Task variant that carries cold users' online session history.

    `online_session_history` holds the first k temporal interactions per cold
    user, enriched with user/item features and history context columns — same
    column schema as `train`. Pass it to the LLM scorer's `fit()` instead of
    `train`; pass `train` to Popularity/ItemKNN scorers as usual.
    """

    online_session_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    k: int = 0


class ColdStartTaskBuilder:
    """Build a cold-start interaction candidate-set task.

    Cold users are entirely withheld from training. Their first k temporal
    interactions form `task.online_session_history` (for LLM/ItemKNN context).
    Post-k interactions become the evaluation target, sampled into candidate
    groups via the configured `CandidateSampler`.

    Item features are enriched from warm `task.train` interactions only, so
    item popularity statistics remain uncontaminated by cold user activity.
    """

    def __init__(
        self,
        name: str,
        dataset_filter: DatasetFilter,
        splitter: ColdUserHoldoutSplitter,
        sampler: CandidateSampler,
        *,
        target_source_column: str = "target_interact",
        target_column: str = "target",
        user_column: str = "user_id",
        item_column: str = "item_id",
        sampled_column: str = "sampled",
        candidate_group_column: str = "candidate_group",
        history_context_columns: tuple[str, ...] = (),
        item_feature_builder: ItemFeatureBuilder | None = None,
    ) -> None:
        self.name = name
        self.dataset_filter = dataset_filter
        self.splitter = splitter
        self.sampler = sampler
        self.target_source_column = target_source_column
        self.target_column = target_column
        self.user_column = user_column
        self.item_column = item_column
        self.sampled_column = sampled_column
        self.candidate_group_column = candidate_group_column
        self.history_context_columns = history_context_columns
        self.item_feature_builder = item_feature_builder

    def build(self, dataset: CanonicalDataset) -> ColdStartTask:
        users = dataset.load_users()
        items = dataset.load_items()
        interactions = dataset.load_interactions()

        users, items, interactions = self.dataset_filter.filter(
            users=users, items=items, interactions=interactions
        )
        split = self.splitter.split(interactions)

        # All cold interactions are the observed exclusion set for negative sampling
        # so profile items and post-k positives are never drawn as negatives.
        all_cold_interactions = pd.concat(
            [split.online_session_history, split.val, split.test],
            ignore_index=True,
        )

        val_rows = self.sampler.sample(
            split.val,
            interactions=all_cold_interactions,
            items=items,
        )
        val_negative_pairs = self._sampled_pairs(val_rows)
        test_rows = self.sampler.sample(
            split.test,
            interactions=all_cold_interactions,
            items=items,
            excluded_pairs=val_negative_pairs,
        )

        items, item_feature_manifest = self._enrich_items_from_train(
            items=items, train_interactions=split.train
        )
        user_features = TaskBuilder._prefixed_features(users, self.user_column, "user")
        item_features = TaskBuilder._prefixed_features(items, self.item_column, "item")
        feature_columns = TaskBuilder._feature_columns(
            user_features,
            item_features,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        task_columns = self._task_columns(feature_columns)

        train = TaskBuilder._with_features(
            rows=self._positive_rows(split.train, candidate_group=pd.NA),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        val = TaskBuilder._with_features(
            rows=self._without_history_context(val_rows),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        test = TaskBuilder._with_features(
            rows=self._without_history_context(test_rows),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        # online_session_history shares the same column schema as train so
        # scorers can call fit() on either frame interchangeably.
        online_session_history = TaskBuilder._with_features(
            rows=self._positive_rows(
                split.online_session_history, candidate_group=pd.NA
            ),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )

        schema = TaskSchema(
            target_column=self.target_column,
            feature_columns=feature_columns,
            id_columns=(self.user_column, self.item_column),
            candidate_group_column=self.candidate_group_column,
            sampled_column=self.sampled_column,
            history_context_columns=self.history_context_columns,
        )

        return ColdStartTask(
            name=self.name,
            train=train,
            val=val,
            test=test,
            online_session_history=online_session_history,
            k=self.splitter.k,
            schema=schema,
            manifest={
                "dataset": dataset.name,
                "dataset_version": dataset.version,
                "target_source_column": self.target_source_column,
                "target_column": self.target_column,
                "feature_columns": list(feature_columns),
                "history_context_columns": list(self.history_context_columns),
                "item_feature_builder": item_feature_manifest,
                "sampled_column": self.sampled_column,
                "candidate_group_column": self.candidate_group_column,
                "k": self.splitter.k,
                "filter": TaskBuilder._component_manifest(self.dataset_filter),
                "splitter": TaskBuilder._component_manifest(self.splitter),
                "sampler": TaskBuilder._component_manifest(self.sampler),
                "rows": {
                    "train": int(len(train)),
                    "val": int(len(val)),
                    "test": int(len(test)),
                    "online_session_history": int(len(online_session_history)),
                },
                "users": {
                    "warm": int(len(split.warm_user_ids)),
                    "val_cold": int(len(split.val_cold_user_ids)),
                    "test_cold": int(len(split.test_cold_user_ids)),
                    "dropped_val_cold": split.dropped_val_cold_count,
                    "dropped_test_cold": split.dropped_test_cold_count,
                },
                "items": int(items[self.item_column].nunique()),
            },
        )

    def _positive_rows(
        self,
        interactions: pd.DataFrame,
        *,
        candidate_group: Any,
    ) -> pd.DataFrame:
        rows = interactions[
            [self.user_column, self.item_column, *self.history_context_columns]
        ].copy()
        rows[self.target_column] = interactions[self.target_source_column]
        rows[self.sampled_column] = False
        rows[self.candidate_group_column] = candidate_group
        return rows

    def _without_history_context(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        for column in self.history_context_columns:
            rows[column] = pd.NA
        return rows

    def _sampled_pairs(self, rows: pd.DataFrame) -> set[tuple[Any, Any]]:
        if rows.empty:
            return set()
        sampled_rows = rows[rows[self.sampled_column].astype(bool)]
        return set(
            zip(
                sampled_rows[self.user_column],
                sampled_rows[self.item_column],
                strict=True,
            )
        )

    def _task_columns(self, feature_columns: tuple[str, ...]) -> list[str]:
        return [
            self.user_column,
            self.item_column,
            *feature_columns,
            *self.history_context_columns,
            self.target_column,
            self.sampled_column,
            self.candidate_group_column,
        ]

    def _enrich_items_from_train(
        self,
        *,
        items: pd.DataFrame,
        train_interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if self.item_feature_builder is None:
            return items, None
        return self.item_feature_builder.enrich_items(
            items=items,
            train_interactions=train_interactions,
            item_column=self.item_column,
        )
