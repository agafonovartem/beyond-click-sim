from __future__ import annotations

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks.base import (
    DatasetFilter,
    ItemFeatureBuilder,
    Splitter,
    Task,
    TaskBuilder,
    TaskSchema,
)
from beyond_click_sim.tasks.samplers import PostSplitUserSampler


class RegressionPredictionTaskBuilder(TaskBuilder):
    """Build observed-only intensity-regression data.

    Train, validation, and test rows are observed interaction rows with known
    numeric targets. An optional post-split eval sampler can cap validation/test
    users while preserving the full train split.
    """

    def __init__(
        self,
        name: str,
        dataset_filter: DatasetFilter,
        splitter: Splitter,
        *,
        target_source_column: str,
        target_column: str = "target",
        user_column: str = "user_id",
        item_column: str = "item_id",
        history_context_columns: tuple[str, ...] = (),
        eval_sampler: PostSplitUserSampler | None = None,
        item_feature_builder: ItemFeatureBuilder | None = None,
    ) -> None:
        super().__init__(
            name=name,
            target_source_column=target_source_column,
            dataset_filter=dataset_filter,
            splitter=splitter,
            sampler=None,
            target_column=target_column,
            user_column=user_column,
            item_column=item_column,
            sampled_column=None,
            candidate_group_column=None,
            history_context_columns=history_context_columns,
            item_feature_builder=item_feature_builder,
        )
        self.eval_sampler = eval_sampler
        self._validate_history_context_columns()
        if self.dataset_filter is None:
            raise ValueError("RegressionPredictionTaskBuilder requires dataset_filter.")
        if self.splitter is None:
            raise ValueError("RegressionPredictionTaskBuilder requires splitter.")

    def build(self, dataset: CanonicalDataset) -> Task:
        users = dataset.load_users()
        items = dataset.load_items()
        interactions = dataset.load_interactions()

        self._require_columns(interactions, [self.user_column, self.item_column])
        self._require_columns(
            interactions,
            [self.target_source_column, *self.history_context_columns],
        )
        self._require_columns(users, [self.user_column])
        self._require_columns(items, [self.item_column])

        users, items, interactions = self.dataset_filter.filter(
            users=users,
            items=items,
            interactions=interactions,
        )
        rows_before_target_filter = len(interactions)
        interactions = interactions[
            interactions[self.target_source_column].notna()
        ].copy()
        split = self.splitter.split(interactions)

        val_rows, val_eval_summary = self._sample_eval_rows(
            split.val,
            train=split.train,
        )
        test_rows, test_eval_summary = self._sample_eval_rows(
            split.test,
            train=split.train,
        )

        items, item_feature_manifest = self._enrich_items_from_train(
            items=items,
            train_interactions=split.train,
        )
        user_features = self._prefixed_features(users, self.user_column, "user")
        item_features = self._prefixed_features(items, self.item_column, "item")
        feature_columns = self._feature_columns(
            user_features,
            item_features,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        task_columns = self._task_columns(feature_columns)

        train = self._with_features(
            rows=self._target_rows(split.train),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        val = self._with_features(
            rows=self._without_history_context(self._target_rows(val_rows)),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        test = self._with_features(
            rows=self._without_history_context(self._target_rows(test_rows)),
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
            candidate_group_column=None,
            sampled_column=None,
            history_context_columns=self.history_context_columns,
        )

        return Task(
            name=self.name,
            train=train,
            val=val,
            test=test,
            schema=schema,
            manifest={
                "protocol": "regression",
                "dataset": dataset.name,
                "dataset_version": dataset.version,
                "target_source_column": self.target_source_column,
                "target_column": self.target_column,
                "feature_columns": list(feature_columns),
                "history_context_columns": list(self.history_context_columns),
                "item_feature_builder": item_feature_manifest,
                "sampled_column": None,
                "candidate_group_column": None,
                "filter": self._component_manifest(self.dataset_filter),
                "splitter": self._component_manifest(self.splitter),
                "sampler": None,
                "eval_sampler": (
                    None
                    if self.eval_sampler is None
                    else self._component_manifest(self.eval_sampler)
                ),
                "eval_user_selection": {
                    "kind": "post_split",
                    "val": val_eval_summary,
                    "test": test_eval_summary,
                },
                "rows": {
                    "train": int(len(train)),
                    "val": int(len(val)),
                    "test": int(len(test)),
                },
                "users": {
                    "filtered": int(interactions[self.user_column].nunique()),
                    "train": int(train[self.user_column].nunique()),
                    "val": int(val[self.user_column].nunique()),
                    "test": int(test[self.user_column].nunique()),
                },
                "items": int(items[self.item_column].nunique()),
                "target_rows_dropped": int(rows_before_target_filter - len(interactions)),
            },
        )

    def _target_rows(self, interactions: pd.DataFrame) -> pd.DataFrame:
        rows = interactions[
            [self.user_column, self.item_column, *self.history_context_columns]
        ].copy()
        rows[self.target_column] = interactions[self.target_source_column]
        return rows

    def _without_history_context(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        for column in self.history_context_columns:
            rows[column] = pd.NA
        return rows

    def _sample_eval_rows(
        self,
        rows: pd.DataFrame,
        *,
        train: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, int | None]]:
        if self.eval_sampler is not None:
            return self.eval_sampler.sample(rows, train=train)
        if rows.empty:
            return rows.copy(), {
                "eligible_users": 0,
                "selected_users": 0,
                "rows_before": 0,
                "rows_after": 0,
            }

        return rows.copy(), {
            "eligible_users": int(rows[self.user_column].nunique()),
            "selected_users": int(rows[self.user_column].nunique()),
            "rows_before": int(len(rows)),
            "rows_after": int(len(rows)),
        }

    def _task_columns(self, feature_columns: tuple[str, ...]) -> list[str]:
        return [
            self.user_column,
            self.item_column,
            *feature_columns,
            *self.history_context_columns,
            self.target_column,
        ]

    def _validate_history_context_columns(self) -> None:
        reserved = {
            self.user_column,
            self.item_column,
            self.target_column,
        }
        overlaps = [
            column
            for column in self.history_context_columns
            if column in reserved
        ]
        if overlaps:
            raise ValueError(
                "history_context_columns cannot include task control columns: "
                f"{overlaps}"
            )
