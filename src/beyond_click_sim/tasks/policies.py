from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from beyond_click_sim.tasks._sampling import stable_sample_values


class Policy(ABC):
    """A fixed recommender policy that produces top-k item lists per user.

    A policy decides which items to surface. It does NOT predict interaction
    probability — that is the scorer's (simulator's) job. Policies are fit on
    training interactions and must not use held-out or test data.

    The output of `recommend` is consumed by a Scorer that predicts how users
    would respond to each recommended item.
    """

    def __init__(self, k: int, seed: int = 0) -> None:
        if k < 1:
            raise ValueError("k must be positive.")
        self.k = k
        self.seed = seed

    @property
    def name(self) -> str:
        """Short identifier used in the policy column and manifests."""
        return self.__class__.__name__

    @abstractmethod
    def fit(
        self,
        train_interactions: pd.DataFrame,
        *,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> Self:
        """Fit policy state from training interactions."""
        ...

    @abstractmethod
    def recommend(
        self,
        users: pd.DataFrame,
        *,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> pd.DataFrame:
        """Return a DataFrame with columns [user_column, item_column, 'policy', 'rank'].

        Returns at most self.k rows per user. Recommended items must not appear
        in the user's training interactions. The 'policy' column holds self.name.
        The 'rank' column holds integer rank 1..k (1 = most recommended).
        """
        ...

    def _seen_items_by_user(
        self,
        train_interactions: pd.DataFrame,
        *,
        user_column: str,
        item_column: str,
    ) -> dict[Any, set[Any]]:
        """Build a {user_id: set(item_ids)} lookup from training interactions."""
        return (
            train_interactions.groupby(user_column, sort=False)[item_column]
            .agg(set)
            .to_dict()
        )

    def _build_recommendation_rows(
        self,
        user_id: Any,
        item_ids: list[Any],
        *,
        user_column: str,
        item_column: str,
    ) -> list[dict[str, Any]]:
        """Build output rows for one user's top-k recommendation list."""
        return [
            {
                user_column: user_id,
                item_column: item_id,
                "policy": self.name,
                "rank": rank,
            }
            for rank, item_id in enumerate(item_ids, start=1)
        ]


class RandomPolicy(Policy):
    """Recommend k random unseen items per user. Useful as a floor baseline.

    Items are sampled uniformly at random from the catalog, excluding items the
    user interacted with in training. The seed and user_id are combined to
    produce a stable per-user sample that does not depend on evaluation order.
    """

    def fit(
        self,
        train_interactions: pd.DataFrame,
        *,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> Self:
        self._all_items_: list[Any] = items[item_column].drop_duplicates().tolist()
        self._user_column_ = user_column
        self._item_column_ = item_column
        return self

    def recommend(
        self,
        users: pd.DataFrame,
        *,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> pd.DataFrame:
        all_items = self._all_items_
        seen_by_user = self._seen_items_by_user(
            train_interactions,
            user_column=user_column,
            item_column=item_column,
        )
        output_columns = [user_column, item_column, "policy", "rank"]
        rows: list[dict[str, Any]] = []

        for user_id in users[user_column]:
            seen = seen_by_user.get(user_id, set())
            unseen = [item for item in all_items if item not in seen]
            if not unseen:
                continue
            rng = random.Random(f"{self.seed}:{user_id}")
            selected = rng.sample(unseen, min(self.k, len(unseen)))
            rows.extend(
                self._build_recommendation_rows(
                    user_id,
                    selected,
                    user_column=user_column,
                    item_column=item_column,
                )
            )

        return pd.DataFrame(rows, columns=output_columns)


class PopularityPolicy(Policy):
    """Recommend the top-k most popular unseen items per user.

    Popularity is measured as total train interaction count per item (i.e. the
    number of users who interacted with each item in training). Items already
    seen by a user in training are excluded from their recommendation list.
    """

    def fit(
        self,
        train_interactions: pd.DataFrame,
        *,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> Self:
        # Item popularity: total interaction count in training.
        popularity = (
            train_interactions.groupby(item_column, sort=False)
            .size()
            .rename("popularity")
        )
        # Merge with full item catalog to include zero-interaction items.
        all_items = items[[item_column]].drop_duplicates().copy()
        all_items = all_items.merge(
            popularity.reset_index(),
            on=item_column,
            how="left",
        )
        all_items["popularity"] = all_items["popularity"].fillna(0)
        # Sort descending by popularity, then by stable item order for ties.
        stable_order = {
            item: i
            for i, item in enumerate(
                stable_sample_values(
                    all_items[item_column],
                    n=None,
                    seed=self.seed,
                )
            )
        }
        all_items["_stable_order_"] = all_items[item_column].map(stable_order)
        all_items = all_items.sort_values(
            ["popularity", "_stable_order_"],
            ascending=[False, True],
        )
        self._sorted_items_: list[Any] = all_items[item_column].tolist()
        self._user_column_ = user_column
        self._item_column_ = item_column
        return self

    def recommend(
        self,
        users: pd.DataFrame,
        *,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> pd.DataFrame:
        seen_by_user = self._seen_items_by_user(
            train_interactions,
            user_column=user_column,
            item_column=item_column,
        )
        output_columns = [user_column, item_column, "policy", "rank"]
        rows: list[dict[str, Any]] = []

        for user_id in users[user_column]:
            seen = seen_by_user.get(user_id, set())
            selected: list[Any] = []
            for item_id in self._sorted_items_:
                if item_id not in seen:
                    selected.append(item_id)
                if len(selected) == self.k:
                    break
            rows.extend(
                self._build_recommendation_rows(
                    user_id,
                    selected,
                    user_column=user_column,
                    item_column=item_column,
                )
            )

        return pd.DataFrame(rows, columns=output_columns)


class ALSPolicy(Policy):
    """Recommend top-k items via Alternating Least Squares on binary interactions.

    Fits an implicit-feedback ALS model (Hu, Koren & Volinsky 2008) on the
    binary user-item interaction matrix. Each observed interaction is treated
    as a positive signal; duplicate (user, item) pairs in training accumulate
    to higher confidence, consistent with standard implicit-feedback MF practice.

    Only items seen in training have latent factors and can be recommended.
    Cold-start users (absent from training) receive no recommendations.
    """

    def __init__(
        self,
        k: int,
        n_factors: int = 64,
        iterations: int = 20,
        seed: int = 0,
    ) -> None:
        super().__init__(k=k, seed=seed)
        if n_factors < 1:
            raise ValueError("n_factors must be positive.")
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        self.n_factors = n_factors
        self.iterations = iterations

    def fit(
        self,
        train_interactions: pd.DataFrame,
        *,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> Self:
        from implicit.als import AlternatingLeastSquares

        users_seen = train_interactions[user_column].unique()
        items_seen = train_interactions[item_column].unique()

        user_idx: dict[Any, int] = {u: i for i, u in enumerate(users_seen)}
        item_idx: dict[Any, int] = {it: i for i, it in enumerate(items_seen)}

        row_ids = train_interactions[user_column].map(user_idx).to_numpy()
        col_ids = train_interactions[item_column].map(item_idx).to_numpy()

        user_item = csr_matrix(
            (np.ones(len(train_interactions), dtype=np.float32), (row_ids, col_ids)),
            shape=(len(users_seen), len(items_seen)),
        )

        model = AlternatingLeastSquares(
            factors=self.n_factors,
            iterations=self.iterations,
            random_state=self.seed,
            use_gpu=False,
        )
        model.fit(user_item)

        self._model_ = model
        self._user_idx_: dict[Any, int] = user_idx
        self._items_in_train_: list[Any] = items_seen.tolist()
        self._user_item_ = user_item
        return self

    def recommend(
        self,
        users: pd.DataFrame,
        *,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> pd.DataFrame:
        output_columns = [user_column, item_column, "policy", "rank"]
        rows: list[dict[str, Any]] = []

        n_items = len(self._items_in_train_)
        for user_id in users[user_column]:
            user_row = self._user_idx_.get(user_id)
            if user_row is None:
                continue
            seen_indices = set(self._user_item_[user_row].indices.tolist())
            unseen_indices = [i for i in range(n_items) if i not in seen_indices]
            if not unseen_indices:
                continue
            item_indices, _ = self._model_.recommend(
                user_row,
                None,
                N=min(self.k, len(unseen_indices)),
                filter_already_liked_items=False,
                items=unseen_indices,
            )
            selected = [self._items_in_train_[i] for i in item_indices]
            rows.extend(
                self._build_recommendation_rows(
                    user_id,
                    selected,
                    user_column=user_column,
                    item_column=item_column,
                )
            )

        return pd.DataFrame(rows, columns=output_columns)
