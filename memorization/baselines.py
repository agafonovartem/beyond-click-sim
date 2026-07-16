"""Experiment-local baselines for the Di Palma et al. (2025) reproduction.

Lives in memorization/ on purpose: src/beyond_click_sim/tasks/policies.py is shared
with other experiments and must not be changed for this reproduction.

Why a local ItemKNN
-------------------
`ItemKNNPolicy` in src scores a candidate by walking the top-k neighbours of each
item in the *user's history*:

    for j in profile(u):            # history item
        scores[topk_neighbours(j)] += sim(j, topk_neighbours(j))

That is a non-standard variant. Classic item-based KNN (and Elliot's `ItemKNN`,
which produced the paper's numbers) restricts to the top-k neighbours of the
*candidate* item instead:

    score(u, i) = sum_{j in profile(u) ∩ topk_neighbours(i)} sim(i, j)

The two differ whenever the neighbour relation is asymmetric (it usually is), which
is why our src ItemKNN under-performs the paper's. This module implements the
classic formulation so the reproduction is comparable.

Both use cosine similarity over the binary item-user matrix, consistent with the
paper treating every rating as an interaction ("without any filtering").
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class StandardItemKNN:
    """Classic item-based KNN: top-k neighbours of the CANDIDATE item.

    Mirrors the `Policy` call signature used by memorization/run_classical.py so it
    can be dropped into the same registry, but is deliberately independent of src.
    """

    def __init__(self, k: int = 50, n_neighbors: int = 40, seed: int = 0) -> None:
        if k < 1:
            raise ValueError("k must be positive.")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        self.k = k
        self.n_neighbors = n_neighbors
        self.seed = seed

    @property
    def name(self) -> str:
        return "ItemKNNStd"

    def fit(
        self,
        train_interactions: pd.DataFrame,
        *,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> Self:
        pairs = train_interactions[[user_column, item_column]].drop_duplicates()
        users_seen = pairs[user_column].unique()
        items_seen = pairs[item_column].unique()

        self._user_idx_: dict[Any, int] = {u: i for i, u in enumerate(users_seen)}
        self._item_idx_: dict[Any, int] = {it: i for i, it in enumerate(items_seen)}
        self._idx_to_item_: list[Any] = items_seen.tolist()
        n_users, n_items = len(users_seen), len(items_seen)

        rows = pairs[item_column].map(self._item_idx_).to_numpy()
        cols = pairs[user_column].map(self._user_idx_).to_numpy()
        item_user = csr_matrix(
            (np.ones(len(pairs), dtype=np.float32), (rows, cols)),
            shape=(n_items, n_users),
        )
        # Cosine similarity = normalized item-user matrix times its transpose.
        item_user_norm = normalize(item_user, norm="l2", axis=1)
        sim = (item_user_norm @ item_user_norm.T).toarray()
        np.fill_diagonal(sim, 0.0)  # an item is not its own neighbour

        # Keep only the top-n_neighbors neighbours OF EACH CANDIDATE ITEM (row i).
        n_keep = min(self.n_neighbors, max(n_items - 1, 1))
        if n_keep < n_items:
            kth = n_items - n_keep
            thresh = np.partition(sim, kth - 1, axis=1)[:, kth - 1][:, None]
            sim[sim < thresh] = 0.0

        self._sim_ = csr_matrix(sim)  # rows = candidate items
        self._user_item_ = item_user.T.tocsr()  # rows = users
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

        for user_id in users[user_column]:
            u = self._user_idx_.get(user_id)
            if u is None:
                continue  # cold user: no profile, no recommendations
            profile = self._user_item_[u]
            if profile.nnz == 0:
                continue
            # score(u, i) = sum_j sim[i, j] * r[u, j] — restricted to top-k of i by construction.
            scores = np.asarray((self._sim_ @ profile.T).todense()).ravel()
            scores[profile.indices] = -np.inf  # exclude train-seen items

            valid = np.isfinite(scores) & (scores > 0)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            n_take = min(self.k, n_valid)
            cand = np.flatnonzero(valid)
            top = cand[np.argpartition(-scores[cand], n_take - 1)[:n_take]]
            top = top[np.argsort(-scores[top], kind="mergesort")]

            rows.extend(
                {
                    user_column: user_id,
                    item_column: self._idx_to_item_[i],
                    "policy": self.name,
                    "rank": rank,
                }
                for rank, i in enumerate(top, start=1)
            )

        return pd.DataFrame(rows, columns=output_columns)
