from __future__ import annotations

import pandas as pd

from beyond_click_sim.tasks.base import DatasetFilter


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
