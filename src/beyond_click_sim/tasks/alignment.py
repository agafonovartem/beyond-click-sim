from __future__ import annotations

from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks.base import (
    CandidateSampler,
    DatasetFilter,
    ItemFeatureBuilder,
    Splitter,
    Task,
    TaskBuilder,
    TaskSchema,
)
from beyond_click_sim.tasks.item_enrichment import item_enrichment_manifest


class AlignmentInteractionTaskBuilder(TaskBuilder):
    """Build interaction candidate-set data for Agent4Rec/SimUSER-style alignment.

    Train contains only observed interactions as user history. Validation/test
    contain held-out observed interactions plus sampled non-interactions grouped
    into candidate sets. The selected target is `target_interact`.

    `history_context_columns` are copied from observed train interactions only.
    They are useful for LLM/user-history prompts, e.g. previous ratings or
    playtime, and are set to missing for validation/test candidates to avoid
    leaking held-out feedback into scorer inputs.
    """

    def __init__(
        self,
        name: str,
        dataset_filter: DatasetFilter,
        splitter: Splitter,
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
        super().__init__(
            name=name,
            target_source_column=target_source_column,
            dataset_filter=dataset_filter,
            splitter=splitter,
            sampler=sampler,
            target_column=target_column,
            user_column=user_column,
            item_column=item_column,
            sampled_column=sampled_column,
            candidate_group_column=candidate_group_column,
            history_context_columns=history_context_columns,
            item_feature_builder=item_feature_builder,
        )
        self._validate_history_context_columns()
        if self.dataset_filter is None:
            raise ValueError(
                "AlignmentInteractionTaskBuilder requires dataset_filter."
            )
        if self.splitter is None:
            raise ValueError("AlignmentInteractionTaskBuilder requires splitter.")
        if self.sampler is None:
            raise ValueError("AlignmentInteractionTaskBuilder requires sampler.")
        if self.sampled_column is None:
            raise ValueError("AlignmentInteractionTaskBuilder requires sampled_column.")
        if self.candidate_group_column is None:
            raise ValueError(
                "AlignmentInteractionTaskBuilder requires candidate_group_column."
            )

    def build(self, dataset: CanonicalDataset) -> Task:
        canonical_manifest = dataset.load_manifest()
        users = dataset.load_users()
        items = dataset.load_items()
        interactions = dataset.load_interactions()
        task_item_enrichment = item_enrichment_manifest(
            canonical_manifest,
            items,
        )

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
        split = self.splitter.split(interactions)

        val_rows = self.sampler.sample(
            split.val,
            interactions=interactions,
            items=items,
        )
        val_negative_pairs = self._sampled_pairs(val_rows)
        test_rows = self.sampler.sample(
            split.test,
            interactions=interactions,
            items=items,
            excluded_pairs=val_negative_pairs,
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
            rows=self._positive_rows(split.train, candidate_group=pd.NA),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        val = self._with_features(
            rows=self._without_history_context(val_rows),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        test = self._with_features(
            rows=self._without_history_context(test_rows),
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

        return Task(
            name=self.name,
            train=train,
            val=val,
            test=test,
            schema=schema,
            manifest={
                "dataset": dataset.name,
                "dataset_version": dataset.version,
                "target_source_column": self.target_source_column,
                "target_column": self.target_column,
                "feature_columns": list(feature_columns),
                "history_context_columns": list(self.history_context_columns),
                "item_enrichment": task_item_enrichment,
                "item_feature_builder": item_feature_manifest,
                "sampled_column": self.sampled_column,
                "candidate_group_column": self.candidate_group_column,
                "filter": self._component_manifest(self.dataset_filter),
                "splitter": self._component_manifest(self.splitter),
                "sampler": self._component_manifest(self.sampler),
                "rows": {
                    "train": int(len(train)),
                    "val": int(len(val)),
                    "test": int(len(test)),
                },
                "users": int(interactions[self.user_column].nunique()),
                "items": int(items[self.item_column].nunique()),
            },
        )

    def _positive_rows(
        self,
        interactions: pd.DataFrame,
        *,
        candidate_group: Any,
    ) -> pd.DataFrame:
        """Convert observed interaction rows into unsampled target rows."""

        rows = interactions[
            [self.user_column, self.item_column, *self.history_context_columns]
        ].copy()
        rows[self.target_column] = interactions[self.target_source_column]
        rows[self.sampled_column] = False
        rows[self.candidate_group_column] = candidate_group
        return rows

    def _without_history_context(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Add history-only columns as missing values for candidate rows."""

        rows = rows.copy()
        for column in self.history_context_columns:
            rows[column] = pd.NA
        return rows

    def _sampled_pairs(self, rows: pd.DataFrame) -> set[tuple[Any, Any]]:
        """Return sampled negative pairs to exclude from later splits."""

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

    def _task_columns(
        self,
        feature_columns: tuple[str, ...],
    ) -> list[str]:
        """Return the stable column order for this task."""

        return [
            self.user_column,
            self.item_column,
            *feature_columns,
            *self.history_context_columns,
            self.target_column,
            self.sampled_column,
            self.candidate_group_column,
        ]

    def _validate_history_context_columns(self) -> None:
        """Keep history-only columns separate from task control columns."""

        reserved = {
            self.user_column,
            self.item_column,
            self.target_column,
            self.sampled_column,
            self.candidate_group_column,
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
