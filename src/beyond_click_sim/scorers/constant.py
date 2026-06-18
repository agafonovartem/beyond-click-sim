from __future__ import annotations

from typing import Any

import pandas as pd

from beyond_click_sim.scorers.base import Scorer


def select_user_history_rows(
    X: pd.DataFrame,
    *,
    user_column: str = "user_id",
    max_history_items: int | None = 30,
) -> dict[Any, pd.DataFrame]:
    """Select the per-user history window used by prompt-window baselines.

    Selection is intentionally input-order based: for each user, keep the last
    `max_history_items` train rows exactly as the LLM prompt builder does.
    """

    history_by_user: dict[Any, pd.DataFrame] = {}
    for user_id, selected_positions in select_user_history_positions(
        X,
        user_column=user_column,
        max_history_items=max_history_items,
    ).items():
        history_by_user[user_id] = X.iloc[selected_positions].copy()
    return history_by_user


def select_user_history_positions(
    X: pd.DataFrame,
    *,
    user_column: str = "user_id",
    max_history_items: int | None = 30,
) -> dict[Any, list[int]]:
    if user_column not in X.columns:
        raise ValueError(f"Missing required columns: [{user_column!r}]")
    if max_history_items is not None and max_history_items < 0:
        raise ValueError("max_history_items must be non-negative")

    positions_by_user: dict[Any, list[int]] = {}
    for position, user_id in enumerate(X[user_column].tolist()):
        positions_by_user.setdefault(user_id, []).append(position)

    if max_history_items is None:
        return positions_by_user
    if max_history_items == 0:
        return {user_id: [] for user_id in positions_by_user}
    return {
        user_id: positions[-max_history_items:]
        for user_id, positions in positions_by_user.items()
    }


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


class ModeRegressor(Scorer):
    """Predict the most frequent train target for every row."""

    name = "mode_regressor"
    tie_break = "smallest"

    def __init__(self) -> None:
        self.mode_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModeRegressor":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(y) == 0:
            raise ValueError("Cannot fit ModeRegressor on empty targets")
        if y.isna().any():
            raise ValueError("y contains NA values")

        self.mode_ = smallest_tie_mode(y)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.mode_ is None:
            raise RuntimeError("ModeRegressor is not fitted")
        return pd.Series(self.mode_, index=X.index, name="score", dtype=float)


class ItemMeanRegressor(Scorer):
    """Predict each item's train target mean, with global mean for cold items."""

    name = "item_mean_regressor"
    stat_source = "train_targets_grouped_by_item"
    cold_item_policy = "global_fallback"

    def __init__(self, *, item_column: str = "item_id") -> None:
        if not item_column:
            raise ValueError("item_column must be non-empty")
        self.item_column = item_column
        self.fallback_scorer_: MeanRegressor | None = None
        self.fallback_: float | None = None
        self.item_mean_by_item_: dict[Any, float] | None = None
        self.item_count_by_item_: dict[Any, int] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ItemMeanRegressor":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.item_column])

        fallback_scorer = MeanRegressor().fit(X, y)
        targets = y.astype(float).reset_index(drop=True)
        item_ids = X[self.item_column].reset_index(drop=True)
        grouped = pd.DataFrame(
            {"item_id": item_ids, "target": targets}
        ).groupby("item_id", dropna=False)["target"]

        self.fallback_scorer_ = fallback_scorer
        self.fallback_ = fallback_scorer.mean_
        self.item_mean_by_item_ = {
            item_id: float(value) for item_id, value in grouped.mean().items()
        }
        self.item_count_by_item_ = {
            item_id: int(value) for item_id, value in grouped.count().items()
        }
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.item_mean_by_item_ is None or self.fallback_ is None:
            raise RuntimeError("ItemMeanRegressor is not fitted")
        self._require_columns(X, [self.item_column])
        return pd.Series(
            [
                self.item_mean_by_item_.get(item_id, self.fallback_)
                for item_id in X[self.item_column]
            ],
            index=X.index,
            name="score",
            dtype=float,
        )

    def cold_item_rows(self, X: pd.DataFrame) -> int:
        if self.item_count_by_item_ is None:
            raise RuntimeError("ItemMeanRegressor is not fitted")
        self._require_columns(X, [self.item_column])
        return int((~X[self.item_column].isin(self.item_count_by_item_)).sum())

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class ItemModeRegressor(Scorer):
    """Predict each item's train target mode, with global mode for cold items."""

    name = "item_mode_regressor"
    stat_source = "train_targets_grouped_by_item"
    cold_item_policy = "global_fallback"
    tie_break = "smallest"

    def __init__(self, *, item_column: str = "item_id") -> None:
        if not item_column:
            raise ValueError("item_column must be non-empty")
        self.item_column = item_column
        self.fallback_scorer_: ModeRegressor | None = None
        self.fallback_: float | None = None
        self.item_mode_by_item_: dict[Any, float] | None = None
        self.item_count_by_item_: dict[Any, int] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ItemModeRegressor":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.item_column])

        fallback_scorer = ModeRegressor().fit(X, y)
        targets = y.astype(float).reset_index(drop=True)
        item_ids = X[self.item_column].reset_index(drop=True)
        grouped = pd.DataFrame(
            {"item_id": item_ids, "target": targets}
        ).groupby("item_id", dropna=False)["target"]

        self.fallback_scorer_ = fallback_scorer
        self.fallback_ = fallback_scorer.mode_
        self.item_mode_by_item_ = {
            item_id: smallest_tie_mode(values)
            for item_id, values in grouped
        }
        self.item_count_by_item_ = {
            item_id: int(value) for item_id, value in grouped.count().items()
        }
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.item_mode_by_item_ is None or self.fallback_ is None:
            raise RuntimeError("ItemModeRegressor is not fitted")
        self._require_columns(X, [self.item_column])
        return pd.Series(
            [
                self.item_mode_by_item_.get(item_id, self.fallback_)
                for item_id in X[self.item_column]
            ],
            index=X.index,
            name="score",
            dtype=float,
        )

    def cold_item_rows(self, X: pd.DataFrame) -> int:
        if self.item_count_by_item_ is None:
            raise RuntimeError("ItemModeRegressor is not fitted")
        self._require_columns(X, [self.item_column])
        return int((~X[self.item_column].isin(self.item_count_by_item_)).sum())

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class UserMeanRegressor(Scorer):
    """Predict each user's mean over the train-history window shown to the LLM."""

    name = "user_mean_regressor"
    history_selection = "last_rows_in_input_order"

    def __init__(
        self,
        *,
        history_value_column: str,
        user_column: str = "user_id",
        max_history_items: int | None = 30,
    ) -> None:
        if not history_value_column:
            raise ValueError("history_value_column must be non-empty")
        if not user_column:
            raise ValueError("user_column must be non-empty")
        if max_history_items is not None and max_history_items < 0:
            raise ValueError("max_history_items must be non-negative")

        self.history_value_column = history_value_column
        self.user_column = user_column
        self.max_history_items = max_history_items
        self.user_mean_by_user_: dict[Any, float] | None = None
        self.user_count_by_user_: dict[Any, int] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UserMeanRegressor":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.user_column, self.history_value_column])

        history_positions_by_user = select_user_history_positions(
            X,
            user_column=self.user_column,
            max_history_items=self.max_history_items,
        )
        history_values = X[self.history_value_column].reset_index(drop=True)
        means: dict[Any, float] = {}
        counts: dict[Any, int] = {}
        for user_id, positions in history_positions_by_user.items():
            values = self._history_values(history_values.iloc[positions], user_id=user_id)
            means[user_id] = float(values.mean())
            counts[user_id] = int(len(values))

        if not means:
            raise ValueError("Cannot fit UserMeanRegressor on empty user history")

        self.user_mean_by_user_ = means
        self.user_count_by_user_ = counts
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.user_mean_by_user_ is None:
            raise RuntimeError("UserMeanRegressor is not fitted")
        self._require_columns(X, [self.user_column])
        self._require_known_users(X, self.user_mean_by_user_)
        return pd.Series(
            [self.user_mean_by_user_[user_id] for user_id in X[self.user_column]],
            index=X.index,
            name="score",
            dtype=float,
        )

    def _history_values(self, values: pd.Series, *, user_id: Any) -> pd.Series:
        if values.empty:
            raise ValueError(f"Empty history window for user {user_id!r}")
        if values.isna().any():
            raise ValueError(f"History values contain NA for user {user_id!r}")
        return values.astype(float)

    def _require_known_users(self, X: pd.DataFrame, values_by_user: dict[Any, float]) -> None:
        missing = [user_id for user_id in X[self.user_column].drop_duplicates() if user_id not in values_by_user]
        if missing:
            raise ValueError(f"Missing fitted history for users: {missing}")

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class UserModeRegressor(UserMeanRegressor):
    """Predict each user's mode over the train-history window shown to the LLM."""

    name = "user_mode_regressor"
    tie_break = "smallest"

    def __init__(
        self,
        *,
        history_value_column: str,
        user_column: str = "user_id",
        max_history_items: int | None = 30,
    ) -> None:
        super().__init__(
            history_value_column=history_value_column,
            user_column=user_column,
            max_history_items=max_history_items,
        )
        self.user_mode_by_user_: dict[Any, float] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UserModeRegressor":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X, [self.user_column, self.history_value_column])

        history_positions_by_user = select_user_history_positions(
            X,
            user_column=self.user_column,
            max_history_items=self.max_history_items,
        )
        history_values = X[self.history_value_column].reset_index(drop=True)
        modes: dict[Any, float] = {}
        counts: dict[Any, int] = {}
        for user_id, positions in history_positions_by_user.items():
            values = self._history_values(history_values.iloc[positions], user_id=user_id)
            modes[user_id] = smallest_tie_mode(values)
            counts[user_id] = int(len(values))

        if not modes:
            raise ValueError("Cannot fit UserModeRegressor on empty user history")

        self.user_mode_by_user_ = modes
        self.user_count_by_user_ = counts
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.user_mode_by_user_ is None:
            raise RuntimeError("UserModeRegressor is not fitted")
        self._require_columns(X, [self.user_column])
        self._require_known_users(X, self.user_mode_by_user_)
        return pd.Series(
            [self.user_mode_by_user_[user_id] for user_id in X[self.user_column]],
            index=X.index,
            name="score",
            dtype=float,
        )


def smallest_tie_mode(values: pd.Series) -> float:
    if len(values) == 0:
        raise ValueError("Cannot compute mode of empty values")
    if values.isna().any():
        raise ValueError("values contain NA")

    counts = values.astype(float).value_counts(sort=False)
    max_count = counts.max()
    modes = sorted(counts[counts == max_count].index)
    return float(modes[0])
