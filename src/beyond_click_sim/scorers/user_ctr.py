from __future__ import annotations

from typing import Any, Self

import pandas as pd

from beyond_click_sim.scorers.base import Scorer


class SmoothedUserCTRScorer(Scorer):
    """Predict from a short user's binary history with a Beta prior.

    This is the scalar-history control for click prediction.  Unlike the raw
    empirical CTR, it never emits exact zero or one from a five-row history.
    The default Beta(1, 1) prior is fixed rather than selected on target labels.
    """

    name = "smoothed_user_ctr"

    def __init__(
        self,
        *,
        history_value_column: str = "history_click",
        user_column: str = "user_id",
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.history_value_column = history_value_column
        self.user_column = user_column
        self.alpha = alpha
        self.beta = beta
        self.user_ctr_by_user_: dict[Any, float] | None = None
        self.user_count_by_user_: dict[Any, int] | None = None
        self.fit_diagnostics_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(
            X, [self.user_column, self.history_value_column]
        )
        values = X[self.history_value_column]
        if values.isna().any():
            raise ValueError("History click values must not contain NA")
        labels = values.astype(int)
        if not set(labels.unique()).issubset({0, 1}):
            raise ValueError("History click values must be binary")

        grouped = (
            pd.DataFrame(
                {
                    self.user_column: X[self.user_column].to_numpy(),
                    "_click": labels.to_numpy(),
                }
            )
            .groupby(self.user_column, sort=False)["_click"]
            .agg(["sum", "count"])
        )
        if grouped.empty:
            raise ValueError("Cannot fit SmoothedUserCTRScorer on empty history")
        denominator = grouped["count"] + self.alpha + self.beta
        estimates = (grouped["sum"] + self.alpha) / denominator
        self.user_ctr_by_user_ = estimates.astype(float).to_dict()
        self.user_count_by_user_ = grouped["count"].astype(int).to_dict()
        self.fit_diagnostics_ = {
            "n_rows": int(len(X)),
            "n_users": int(len(grouped)),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "prior_mean": float(self.alpha / (self.alpha + self.beta)),
            "history_rows_per_user_min": int(grouped["count"].min()),
            "history_rows_per_user_max": int(grouped["count"].max()),
        }
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.user_ctr_by_user_ is None:
            raise RuntimeError("SmoothedUserCTRScorer is not fitted")
        self._require_columns(X, [self.user_column])
        unknown = [
            user_id
            for user_id in X[self.user_column].drop_duplicates()
            if user_id not in self.user_ctr_by_user_
        ]
        if unknown:
            raise ValueError(
                f"Missing fitted history for {len(unknown)} users"
            )
        return pd.Series(
            [
                self.user_ctr_by_user_[user_id]
                for user_id in X[self.user_column]
            ],
            index=X.index,
            dtype=float,
            name="score",
        )

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame, columns: list[str]
    ) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
