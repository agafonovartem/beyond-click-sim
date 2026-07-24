from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder

from beyond_click_sim.scorers.base import Scorer


class ExposureLogisticBiasScorer(Scorer):
    """Exposure-aware logistic user/item bias baseline.

    Both clicked and non-clicked exposed rows are used.  One-hot user and item
    effects provide a simple probability baseline that stays well-defined for
    users with an all-zero visible history, unlike positive-only implicit
    recommenders.
    """

    name = "exposure_logistic_bias"

    def __init__(
        self,
        *,
        user_column: str = "user_id",
        item_column: str = "item_id",
        alpha: float = 1e-5,
        max_iter: int = 1_000,
        tolerance: float = 1e-4,
        seed: int = 0,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.user_column = user_column
        self.item_column = item_column
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.seed = seed
        self.encoder_: OneHotEncoder | None = None
        self.model_: SGDClassifier | None = None
        self.fit_diagnostics_: dict[str, Any] | None = None
        self.known_users_: set[object] | None = None
        self.known_items_: set[object] | None = None
        self.coverage_diagnostics_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self._require_columns(X)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if y.isna().any():
            raise ValueError("Exposure targets must not contain missing values")
        labels = y.astype(int)
        if not set(labels.unique()).issubset({0, 1}):
            raise ValueError("Exposure targets must be binary")
        if labels.nunique() != 2:
            raise ValueError("Exposure logistic baseline requires both classes")

        encoder = OneHotEncoder(
            handle_unknown="error",
            sparse_output=True,
            dtype=np.float64,
        )
        design = encoder.fit_transform(
            X[[self.user_column, self.item_column]].astype("string")
        )
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tolerance,
            random_state=self.seed,
            average=True,
            class_weight=None,
            fit_intercept=True,
        )
        model.fit(design, labels.to_numpy())

        self.encoder_ = encoder
        self.model_ = model
        self.known_users_ = set(X[self.user_column])
        self.known_items_ = set(X[self.item_column])
        self.fit_diagnostics_ = {
            "n_rows": int(len(X)),
            "n_users": int(X[self.user_column].nunique()),
            "n_items": int(X[self.item_column].nunique()),
            "positive_rate": float(labels.mean()),
            "uses_positive_and_negative_exposures": True,
            "class_weight": None,
        }
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if (
            self.encoder_ is None
            or self.model_ is None
            or self.known_users_ is None
            or self.known_items_ is None
        ):
            raise RuntimeError("ExposureLogisticBiasScorer is not fitted")
        self._require_columns(X)
        unknown_users = int(
            (~X[self.user_column].isin(self.known_users_)).sum()
        )
        unknown_items = int(
            (~X[self.item_column].isin(self.known_items_)).sum()
        )
        self.coverage_diagnostics_ = {
            "rows": int(len(X)),
            "unknown_user_rows": unknown_users,
            "unknown_item_rows": unknown_items,
            "fallback_rows": 0,
            "cold_score_policy": "error",
        }
        try:
            design = self.encoder_.transform(
                X[[self.user_column, self.item_column]].astype("string")
            )
        except ValueError as error:
            raise ValueError(
                "ExposureLogisticBiasScorer cannot score cold users/items: "
                f"unknown_users={unknown_users}, "
                f"unknown_items={unknown_items}"
            ) from error
        probabilities = self.model_.predict_proba(design)[:, 1]
        return pd.Series(
            probabilities,
            index=X.index,
            dtype=float,
            name="score",
        )

    def _require_columns(self, X: pd.DataFrame) -> None:
        missing = [
            column
            for column in (self.user_column, self.item_column)
            if column not in X.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
