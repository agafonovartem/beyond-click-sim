from __future__ import annotations

import pandas as pd

from beyond_click_sim.scorers.base import Scorer


class PopularityScorer(Scorer):
    """Score each row by the train target sum of its item."""

    name = "popularity"

    def __init__(self, item_column: str = "item_id") -> None:
        self.item_column = item_column
        self.item_scores_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PopularityScorer":
        if self.item_column not in X.columns:
            raise ValueError(f"Missing item column: {self.item_column!r}")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        frame = pd.DataFrame(
            {
                self.item_column: X[self.item_column].to_numpy(),
                "target": y.to_numpy(),
            }
        )
        self.item_scores_ = frame.groupby(self.item_column)["target"].sum().astype(float)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.item_scores_ is None:
            raise RuntimeError("PopularityScorer is not fitted")
        if self.item_column not in X.columns:
            raise ValueError(f"Missing item column: {self.item_column!r}")

        scores = X[self.item_column].map(self.item_scores_).fillna(0.0)
        return pd.Series(scores.to_numpy(dtype=float), index=X.index, name="score")
