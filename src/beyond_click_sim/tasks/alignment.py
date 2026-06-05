from __future__ import annotations

from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks.base import (
    CandidateSampler,
    DatasetFilter,
    Splitter,
    Task,
    TaskBuilder,
    TaskSchema,
)


class AlignmentInteractionTaskBuilder(TaskBuilder):
    """Build interaction candidate-set data for Agent4Rec/SimUSER-style alignment.

    Train contains only observed interactions as user history. Validation/test
    contain held-out observed interactions plus sampled non-interactions grouped
    into candidate sets. The selected target is `target_interact`.
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
        )
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
        users = dataset.load_users()
        items = dataset.load_items()
        interactions = dataset.load_interactions()

        self._require_columns(interactions, [self.user_column, self.item_column])
        self._require_columns(interactions, [self.target_source_column])
        self._require_columns(users, [self.user_column])
        self._require_columns(items, [self.item_column])

        users, items, interactions = self.dataset_filter.filter(
            users=users,
            items=items,
            interactions=interactions,
        )
        split = self.splitter.split(interactions)

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
            rows=self.sampler.sample(
                split.val,
                interactions=interactions,
                items=items,
            ),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        test = self._with_features(
            rows=self.sampler.sample(
                split.test,
                interactions=interactions,
                items=items,
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

        rows = interactions[[self.user_column, self.item_column]].copy()
        rows[self.target_column] = interactions[self.target_source_column]
        rows[self.sampled_column] = False
        rows[self.candidate_group_column] = candidate_group
        return rows

    def _task_columns(
        self,
        feature_columns: tuple[str, ...],
    ) -> list[str]:
        """Return the stable column order for this task."""

        # TODO: for LLM history we might need ratings as well. 
        return [
            self.user_column,
            self.item_column,
            *feature_columns,
            self.target_column,
            self.sampled_column,
            self.candidate_group_column,
        ]
