from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from beyond_click_sim.tasks.base import ItemFeatureBuilder


ITEM_RATING_MEAN_COLUMN = "rating_mean"
ITEM_RATING_COUNT_COLUMN = "rating_count"
PREFIXED_ITEM_RATING_MEAN_COLUMN = f"item_{ITEM_RATING_MEAN_COLUMN}"
PREFIXED_ITEM_RATING_COUNT_COLUMN = f"item_{ITEM_RATING_COUNT_COLUMN}"
PREFIXED_ITEM_RATING_STATS_COLUMNS = (
    PREFIXED_ITEM_RATING_MEAN_COLUMN,
    PREFIXED_ITEM_RATING_COUNT_COLUMN,
)


@dataclass(frozen=True)
class TrainItemRatingStatistics(ItemFeatureBuilder):
    """Add item rating mean/count computed only from train interactions."""

    value_column: str = "rating"
    mean_column: str = ITEM_RATING_MEAN_COLUMN
    count_column: str = ITEM_RATING_COUNT_COLUMN

    def enrich_items(
        self,
        *,
        items: pd.DataFrame,
        train_interactions: pd.DataFrame,
        item_column: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        missing_item_columns = [
            column for column in [item_column] if column not in items.columns
        ]
        missing_interaction_columns = [
            column
            for column in [item_column, self.value_column]
            if column not in train_interactions.columns
        ]
        if missing_item_columns:
            raise ValueError(f"Missing item columns: {missing_item_columns}")
        if missing_interaction_columns:
            raise ValueError(
                f"Missing train interaction columns: {missing_interaction_columns}"
            )

        stats = (
            train_interactions.groupby(item_column, dropna=False)[self.value_column]
            .agg(**{self.mean_column: "mean", self.count_column: "count"})
            .reset_index()
        )
        enriched = items.merge(stats, on=item_column, how="left")
        enriched[self.count_column] = enriched[self.count_column].fillna(0).astype(int)

        items_total = int(items[item_column].nunique())
        items_with_statistics = int(
            stats[stats[self.count_column] > 0][item_column].nunique()
        )
        manifest = {
            "class": self.__class__.__name__,
            "source": "train_split_only",
            "value_column": self.value_column,
            "mean_column": self.mean_column,
            "count_column": self.count_column,
            "missing_policy": {
                self.mean_column: "nan",
                self.count_column: 0,
            },
            "items_with_statistics": items_with_statistics,
            "items_total": items_total,
        }
        return enriched, manifest
