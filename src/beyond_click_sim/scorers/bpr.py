from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.tasks.policies import BPRPolicy


class BPRScorer(Scorer):
    """Score fixed user-item candidates with an implicit-feedback BPR model.

    The existing :class:`BPRPolicy` generates top-k recommendations.  Evaluation
    tasks in this project instead provide an explicit candidate set, so this
    adapter reuses the policy's fitting implementation and returns the learned
    user/item factor dot product for every supplied row.

    Only rows with a positive binary target are used as implicit interactions.
    Candidate coverage is intentionally strict: unknown users or items raise
    instead of receiving a silent fallback score.  OOD task builders should
    report and enforce their warm-user/warm-item support assumptions.
    """

    name = "bpr"

    def __init__(
        self,
        *,
        user_column: str = "user_id",
        item_column: str = "item_id",
        n_factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        seed: int = 0,
        cold_score_policy: str = "error",
    ) -> None:
        if cold_score_policy not in {"error", "zero"}:
            raise ValueError("cold_score_policy must be 'error' or 'zero'")
        self.user_column = user_column
        self.item_column = item_column
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.cold_score_policy = cold_score_policy
        self.policy_: BPRPolicy | None = None
        self.user_index_: dict[Any, int] | None = None
        self.item_index_: dict[Any, int] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X)
        if y.isna().any():
            raise ValueError("BPRScorer targets must not contain missing values")

        positive_mask = y.astype(bool).to_numpy()
        positives = (
            X.loc[positive_mask, [self.user_column, self.item_column]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if positives.empty:
            raise ValueError("BPRScorer requires at least one positive interaction")

        policy = BPRPolicy(
            k=1,
            n_factors=self.n_factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            seed=self.seed,
        )
        items = pd.DataFrame(
            {self.item_column: positives[self.item_column].drop_duplicates()}
        )
        policy.fit(
            positives,
            items=items,
            user_column=self.user_column,
            item_column=self.item_column,
        )

        self.policy_ = policy
        self.user_index_ = dict(policy._user_idx_)
        self.item_index_ = {
            item_id: index
            for index, item_id in enumerate(policy._items_in_train_)
        }
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if (
            self.policy_ is None
            or self.user_index_ is None
            or self.item_index_ is None
        ):
            raise RuntimeError("BPRScorer is not fitted")
        self._require_columns(X)

        user_indices = X[self.user_column].map(self.user_index_)
        item_indices = X[self.item_column].map(self.item_index_)
        unknown_users = int(user_indices.isna().sum())
        unknown_items = int(item_indices.isna().sum())
        if (unknown_users or unknown_items) and self.cold_score_policy == "error":
            raise ValueError(
                "BPRScorer cannot score cold rows: "
                f"unknown_users={unknown_users}, unknown_items={unknown_items}"
            )

        model = self.policy_._model_
        known = user_indices.notna() & item_indices.notna()
        values = np.zeros(len(X), dtype=float)
        if known.any():
            known_user_indices = user_indices.loc[known].astype(int).to_numpy()
            known_item_indices = item_indices.loc[known].astype(int).to_numpy()
            user_factors = model.user_factors[known_user_indices]
            item_factors = model.item_factors[known_item_indices]
            values[known.to_numpy()] = np.einsum(
                "ij,ij->i",
                user_factors,
                item_factors,
            )
        self.coverage_diagnostics_ = {
            "rows": int(len(X)),
            "unknown_user_rows": unknown_users,
            "unknown_item_rows": unknown_items,
            "fallback_rows": int((~known).sum()),
            "cold_score_policy": self.cold_score_policy,
        }
        return pd.Series(values, index=X.index, dtype=float, name="score")

    def _require_columns(self, X: pd.DataFrame) -> None:
        missing = [
            column
            for column in (self.user_column, self.item_column)
            if column not in X.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
