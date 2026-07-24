from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from beyond_click_sim.scorers.base import Scorer


class ItemKNNScorer(Scorer):
    """Score explicit candidates with positive-only item-based KNN.

    ``fit`` builds binary item-user cosine similarities from positive source
    interactions and stores each source user's positive item profile.  This is
    the explicit-candidate counterpart of :class:`ItemKNNPolicy`: candidate
    scores are the sum (or mean) of top-neighbor similarities contributed by
    the user's clicked source items.

    Coverage is strict.  A candidate user must have at least one positive
    source interaction and a candidate item must have positive source support.
    A score of zero is therefore meaningful: both nodes are warm, but none of
    the user's profile items contributes a retained neighbor edge.
    """

    name = "item_knn"

    def __init__(
        self,
        n_neighbors: int = 20,
        aggregation: str = "sum",
        item_column: str = "item_id",
        user_column: str = "user_id",
        cold_score_policy: str = "error",
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if aggregation not in ("mean", "sum"):
            raise ValueError(
                f"aggregation must be 'mean' or 'sum', got {aggregation!r}"
            )
        if cold_score_policy not in {"error", "zero"}:
            raise ValueError("cold_score_policy must be 'error' or 'zero'")
        self.n_neighbors = n_neighbors
        self.aggregation = aggregation
        self.item_column = item_column
        self.user_column = user_column
        self.cold_score_policy = cold_score_policy
        self._backend_: ColdItemKNNScorer | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self._require_columns(X)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if y.isna().any():
            raise ValueError("ItemKNNScorer targets must not contain missing values")
        unique_targets = set(y.unique().tolist())
        if not unique_targets.issubset({0, 1, False, True}):
            raise ValueError(
                "ItemKNNScorer targets must be binary values in {0, 1}"
            )

        positive_mask = y.astype(bool).to_numpy()
        positives = (
            X.loc[positive_mask, [self.user_column, self.item_column]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if positives.empty:
            raise ValueError(
                "ItemKNNScorer requires at least one positive interaction"
            )

        backend = ColdItemKNNScorer(
            n_neighbors=self.n_neighbors,
            aggregation=self.aggregation,
            item_column=self.item_column,
            user_column=self.user_column,
        )
        # The cold-start scorer already implements the same neighbor graph and
        # profile aggregation as ItemKNNPolicy.  Here both stages come from the
        # same positive-only source log.
        backend.fit_train(
            positives,
            pd.Series(1, index=positives.index, dtype=int),
        )
        backend.fit(
            positives,
            pd.Series(1, index=positives.index, dtype=int),
        )
        self._backend_ = backend
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self._backend_ is None:
            raise RuntimeError("ItemKNNScorer is not fitted")
        self._require_columns(X)

        known_users = set(self._backend_._user_profiles_)
        known_items = set(self._backend_._item_idx_)
        unknown_users = int((~X[self.user_column].isin(known_users)).sum())
        unknown_items = int((~X[self.item_column].isin(known_items)).sum())
        if (unknown_users or unknown_items) and self.cold_score_policy == "error":
            raise ValueError(
                "ItemKNNScorer cannot score cold rows: "
                f"unknown_users={unknown_users}, unknown_items={unknown_items}"
            )
        self.coverage_diagnostics_ = {
            "rows": int(len(X)),
            "unknown_user_rows": unknown_users,
            "unknown_item_rows": unknown_items,
            "fallback_rows": int(
                (
                    ~X[self.user_column].isin(known_users)
                    | ~X[self.item_column].isin(known_items)
                ).sum()
            ),
            "cold_score_policy": self.cold_score_policy,
        }
        return self._backend_.score(X)

    def _require_columns(self, X: pd.DataFrame) -> None:
        missing = [
            column
            for column in (self.user_column, self.item_column)
            if column not in X.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class ColdItemKNNScorer(Scorer):
    """Item-KNN scorer for the cold-start alignment task.

    Two-step fit protocol:
      1. fit_train(X_train, y_train) — builds item-item cosine similarity from warm interactions.
      2. fit(X_history, y_history)  — stores cold user k-item profiles from online_session_history.

    score(X) aggregates neighbor similarities over each cold user's profile items.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        aggregation: str = "mean",
        item_column: str = "item_id",
        user_column: str = "user_id",
    ) -> None:
        if aggregation not in ("mean", "sum"):
            raise ValueError(f"aggregation must be 'mean' or 'sum', got {aggregation!r}")
        self.n_neighbors = n_neighbors
        self.aggregation = aggregation
        self.item_column = item_column
        self.user_column = user_column
        self._neighbors_: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._item_idx_: dict | None = None
        self._n_items_: int | None = None

    def fit_train(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Build item-item cosine similarity matrix from warm training interactions.

        Mirrors ItemKNNPolicy.fit() (src/beyond_click_sim/tasks/policies.py).
        y is accepted for API consistency but not used — similarity is binary co-occurrence only.
        """
        items_seen = X[self.item_column].unique()
        users_seen = X[self.user_column].unique()

        item_idx: dict = {it: i for i, it in enumerate(items_seen)}
        user_idx: dict = {u: i for i, u in enumerate(users_seen)}

        row_ids = X[self.item_column].map(item_idx).to_numpy()
        col_ids = X[self.user_column].map(user_idx).to_numpy()

        n_items = len(items_seen)
        n_users = len(users_seen)
        item_user = csr_matrix(
            (np.ones(len(X), dtype=np.float32), (row_ids, col_ids)),
            shape=(n_items, n_users),
        )
        item_user_norm = normalize(item_user, norm="l2", axis=1)

        # +1 because kneighbors returns each item as its own nearest neighbor
        n_query = min(self.n_neighbors + 1, n_items)
        nn = NearestNeighbors(n_neighbors=n_query, metric="cosine", algorithm="brute")
        nn.fit(item_user_norm)
        distances, indices = nn.kneighbors(item_user_norm)
        similarities = 1.0 - distances  # cosine distance → cosine similarity

        # Drop self-match for each item
        neighbors: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_items):
            mask = indices[i] != i
            neighbors.append((indices[i][mask], similarities[i][mask]))

        self._neighbors_ = neighbors
        self._item_idx_ = item_idx
        self._n_items_ = n_items
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Store cold user profiles from online_session_history.

        Must be called after fit_train(). X is the online_session_history frame;
        the profile for each cold user is exactly the k items in that frame.
        y is accepted for API consistency but not used.
        """
        if self._neighbors_ is None:
            raise RuntimeError("ColdItemKNNScorer: call fit_train() before fit()")
        self._user_profiles_: dict[object, list] = (
            X.groupby(self.user_column)[self.item_column].apply(list).to_dict()
        )
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Score each (user_id, item_id) candidate row via aggregated neighbor similarity."""
        if self._neighbors_ is None or not hasattr(self, "_user_profiles_"):
            raise RuntimeError("ColdItemKNNScorer: call fit_train() and fit() before score()")

        result = pd.Series(0.0, index=X.index, dtype=float, name="score")

        for user_id, user_group in X.groupby(self.user_column, sort=False):
            profile = self._user_profiles_.get(user_id, [])
            if not profile:
                continue

            # Accumulate neighbor similarities across all profile items (ItemKNNPolicy pattern)
            user_scores = np.zeros(self._n_items_, dtype=np.float64)
            n_contributing = 0
            for profile_item_id in profile:
                profile_idx = self._item_idx_.get(profile_item_id)
                if profile_idx is None:
                    continue
                nbr_indices, nbr_sims = self._neighbors_[profile_idx]
                np.add.at(user_scores, nbr_indices, nbr_sims)
                n_contributing += 1

            if self.aggregation == "mean" and n_contributing > 0:
                user_scores /= n_contributing

            candidate_item_ids = user_group[self.item_column].to_numpy()
            candidate_indices = np.array(
                [self._item_idx_.get(iid, -1) for iid in candidate_item_ids]
            )
            valid = candidate_indices >= 0
            result.loc[user_group.index[valid]] = user_scores[candidate_indices[valid]]

        return result
