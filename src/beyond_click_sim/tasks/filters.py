from __future__ import annotations

import pandas as pd

from beyond_click_sim.tasks.base import DatasetFilter
from beyond_click_sim.tasks._sampling import stable_sample_values


class SequentialDatasetFilter(DatasetFilter):
    """Apply dataset filters in a fixed order."""

    def __init__(
        self,
        filters: list[DatasetFilter] | tuple[DatasetFilter, ...],
    ) -> None:
        self.filters = tuple(filters)
        if not self.filters:
            raise ValueError("filters must be non-empty.")

    def filter(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        for dataset_filter in self.filters:
            users, items, interactions = dataset_filter.filter(
                users=users,
                items=items,
                interactions=interactions,
            )
        return users, items, interactions


class MinUserInteractionsFilter(DatasetFilter):
    """Keep users with at least `min_interactions` observed rows."""

    def __init__(
        self,
        min_interactions: int = 10,
        user_column: str = "user_id",
    ) -> None:
        self.min_interactions = min_interactions
        self.user_column = user_column

    def filter(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.min_interactions < 1:
            raise ValueError("min_interactions must be positive.")

        counts = interactions.groupby(self.user_column, sort=False).size()
        keep_users = counts[counts >= self.min_interactions].index

        filtered_users = users[users[self.user_column].isin(keep_users)].copy()
        filtered_interactions = interactions[
            interactions[self.user_column].isin(keep_users)
        ].copy()

        return filtered_users, items.copy(), filtered_interactions


class SampleUsersFilter(DatasetFilter):
    """Keep a deterministic random sample of users before splitting."""

    def __init__(
        self,
        n_users: int,
        seed: int = 0,
        user_column: str = "user_id",
    ) -> None:
        self.n_users = n_users
        self.seed = seed
        self.user_column = user_column

    def filter(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.n_users < 1:
            raise ValueError("n_users must be positive.")

        sampled_users = set(
            stable_sample_values(
                users[self.user_column],
                n=self.n_users,
                seed=self.seed,
            )
        )
        filtered_users = users[users[self.user_column].isin(sampled_users)].copy()
        filtered_interactions = interactions[
            interactions[self.user_column].isin(sampled_users)
        ].copy()

        return filtered_users, items.copy(), filtered_interactions
