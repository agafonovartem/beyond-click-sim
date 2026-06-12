from __future__ import annotations

import pandas as pd

from beyond_click_sim.scorers.base import Scorer


class MeanRegressor(Scorer):
    """Predict the train target mean for every row."""

    name = "mean_regressor"

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanRegressor":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(y) == 0:
            raise ValueError("Cannot fit MeanRegressor on empty targets")
        if y.isna().any():
            raise ValueError("y contains NA values")
        self.mean_ = float(y.astype(float).mean())
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.mean_ is None:
            raise RuntimeError("MeanRegressor is not fitted")
        return pd.Series(self.mean_, index=X.index, name="score", dtype=float)
