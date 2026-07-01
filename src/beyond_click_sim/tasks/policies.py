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


class ItemKNNPolicy(Policy):
    """Recommend top-k items via item-based K-nearest-neighbors collaborative filtering.

    Scores each candidate item as the sum of cosine similarities to the items
    the user has interacted with in training. Only the top `n_neighbors` most
    similar items of each candidate contribute, which prevents popular items
    from dominating and keeps scoring tractable.

    Similarity is computed on the binary item-user interaction matrix (L2-normalized
    per item). Items not seen in training have no latent representation and cannot
    be recommended. Cold-start users (absent from training) receive no recommendations.
    """

    def __init__(
        self,
        k: int,
        n_neighbors: int = 20,
        seed: int = 0,
    ) -> None:
        super().__init__(k=k, seed=seed)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        self.n_neighbors = n_neighbors

    def fit(
        self,
        train_interactions: pd.DataFrame,
        *,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> Self:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import normalize

        items_seen = train_interactions[item_column].unique()
        users_seen = train_interactions[user_column].unique()

        item_idx: dict[Any, int] = {it: i for i, it in enumerate(items_seen)}
        user_idx: dict[Any, int] = {u: i for i, u in enumerate(users_seen)}

        row_ids = train_interactions[item_column].map(item_idx).to_numpy()
        col_ids = train_interactions[user_column].map(user_idx).to_numpy()

        n_items = len(items_seen)
        n_users = len(users_seen)
        item_user = csr_matrix(
            (np.ones(len(train_interactions), dtype=np.float32), (row_ids, col_ids)),
            shape=(n_items, n_users),
        )
        item_user_norm = normalize(item_user, norm="l2", axis=1)

        # +1 because kneighbors returns each item as its own nearest neighbor
        n_query = min(self.n_neighbors + 1, n_items)
        nn = NearestNeighbors(n_neighbors=n_query, metric="cosine", algorithm="brute")
        nn.fit(item_user_norm)
        distances, indices = nn.kneighbors(item_user_norm)
        similarities = 1.0 - distances  # cosine distance → cosine similarity

        # Drop the self-match (the item is always returned as its own neighbor at distance ~0)
        neighbors: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_items):
            mask = indices[i] != i
            neighbors.append((indices[i][mask], similarities[i][mask]))

        self._neighbors_: list[tuple[np.ndarray, np.ndarray]] = neighbors
        self._item_idx_: dict[Any, int] = item_idx
        self._idx_to_item_: list[Any] = items_seen.tolist()
        self._n_items_: int = n_items
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
            seed_indices = [self._item_idx_[it] for it in seen if it in self._item_idx_]
            if not seed_indices:
                continue

            scores = np.zeros(self._n_items_, dtype=np.float64)
            for seed_idx in seed_indices:
                nbr_indices, nbr_sims = self._neighbors_[seed_idx]
                np.add.at(scores, nbr_indices, nbr_sims)

            seen_indices_set = set(seed_indices)
            for idx in seen_indices_set:
                scores[idx] = 0.0

            positive_indices = np.where(scores > 0)[0]
            if len(positive_indices) == 0:
                continue

            n_take = min(self.k, len(positive_indices))
            if len(positive_indices) <= n_take:
                top_indices = positive_indices[np.argsort(-scores[positive_indices])]
            else:
                local_top = np.argpartition(scores[positive_indices], -n_take)[-n_take:]
                top_indices = positive_indices[
                    local_top[np.argsort(-scores[positive_indices][local_top])]
                ]

            selected = [self._idx_to_item_[i] for i in top_indices]
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


class BPRPolicy(Policy):
    """Recommend top-k items via Bayesian Personalized Ranking Matrix Factorization.

    Fits a BPR-MF model (Rendle et al. 2009) on binary interactions. Unlike ALS,
    BPR optimizes a pairwise ranking objective via SGD: for each observed (user, item)
    pair it maximizes the predicted score over a randomly sampled unobserved item.
    This makes it better aligned with ranking tasks than pointwise MSE objectives.

    Only items seen in training have latent factors and can be recommended.
    Cold-start users (absent from training) receive no recommendations.
    """

    def __init__(
        self,
        k: int,
        n_factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        seed: int = 0,
    ) -> None:
        super().__init__(k=k, seed=seed)
        if n_factors < 1:
            raise ValueError("n_factors must be positive.")
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations

    def fit(
        self,
        train_interactions: pd.DataFrame,
        *,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> Self:
        from implicit.bpr import BayesianPersonalizedRanking

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

        model = BayesianPersonalizedRanking(
            factors=self.n_factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
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



def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LightGCNPolicy(Policy):
    """Recommend top-k items via LightGCN (He et al. 2020).

    Builds a bipartite user-item graph and propagates embeddings through L
    graph-convolutional layers, each of which is a fixed D^(-1/2) A D^(-1/2)
    sparse matrix multiplication. Because the GCN layers have no trainable
    weights, only the initial embeddings E0 are learned — via BPR loss and Adam
    — making the entire model implementable in pure NumPy/SciPy.

    Final embeddings are the mean across all L+1 layers. The backward pass
    through the symmetric normalized adjacency is identical to the forward pass
    (A_norm^T = A_norm), so gradients are computed by the same L sparse-dense
    multiplications applied to grad_E_final.

    L2 regularization is applied to E0 only, per the original paper (Section 3.3).

    Cold-start users (absent from training) receive no recommendations.
    """

    def __init__(
        self,
        k: int,
        n_factors: int = 64,
        n_layers: int = 3,
        learning_rate: float = 0.001,
        regularization: float = 1e-4,
        iterations: int = 200,
        batch_size: int = 2048,
        seed: int = 0,
    ) -> None:
        super().__init__(k=k, seed=seed)
        if n_factors < 1:
            raise ValueError("n_factors must be positive.")
        if n_layers < 1:
            raise ValueError("n_layers must be positive.")
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.batch_size = batch_size

    def fit(
        self,
        train_interactions,
        *,
        items,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ):
        from scipy.sparse import diags, hstack, vstack

        # Deduplicate: treat interactions as binary
        unique_ints = train_interactions[[user_column, item_column]].drop_duplicates()
        users_seen = unique_ints[user_column].unique()
        items_seen = unique_ints[item_column].unique()
        n_users = len(users_seen)
        n_items = len(items_seen)

        user_idx = {u: i for i, u in enumerate(users_seen)}
        item_idx = {it: i for i, it in enumerate(items_seen)}

        u_ids = unique_ints[user_column].map(user_idx).to_numpy()
        i_ids = unique_ints[item_column].map(item_idx).to_numpy()
        ones32 = np.ones(len(unique_ints), dtype=np.float32)

        # Binary user-item matrix R (n_users x n_items)
        R = csr_matrix((ones32, (u_ids, i_ids)), shape=(n_users, n_items))

        # Bipartite adjacency A = [[0, R], [R^T, 0]], shape (n_users+n_items, n_users+n_items)
        A = vstack([
            hstack([csr_matrix((n_users, n_users), dtype=np.float32), R]),
            hstack([R.T.tocsr(), csr_matrix((n_items, n_items), dtype=np.float32)]),
        ]).tocsr()

        # Symmetric normalization: A_norm = D^(-1/2) A D^(-1/2)
        deg = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0).astype(np.float32)
        D_inv_sqrt = diags(d_inv_sqrt)
        A_norm = (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()

        # Initialize E0 ~ N(0, 0.1^2), shape (n_users+n_items, n_factors)
        rng = np.random.default_rng(self.seed)
        E0 = (rng.standard_normal((n_users + n_items, self.n_factors)) * 0.1).astype(np.float64)

        # Adam state
        m_adam = np.zeros_like(E0)
        v_adam = np.zeros_like(E0)
        b1, b2, eps_adam = 0.9, 0.999, 1e-8

        n_pairs = len(u_ids)

        for epoch in range(self.iterations):
            # Forward pass: E_final = (E^0 + E^1 + ... + E^L) / (L+1)
            E = E0.copy()
            E_sum = E0.copy()
            for _ in range(self.n_layers):
                E = A_norm @ E
                E_sum += E
            E_final = E_sum / (self.n_layers + 1)

            E_users = E_final[:n_users]
            E_items = E_final[n_users:]

            # Uniform negative sampling (one negative per positive pair)
            j_neg = rng.integers(0, n_items, size=n_pairs)

            # Accumulate BPR gradient w.r.t. E_final in batches
            grad_E_final = np.zeros_like(E0)
            for start in range(0, n_pairs, self.batch_size):
                end = min(start + self.batch_size, n_pairs)
                ub = u_ids[start:end]
                ib = i_ids[start:end]
                jb = j_neg[start:end]

                x_ui = (E_users[ub] * E_items[ib]).sum(axis=1)
                x_uj = (E_users[ub] * E_items[jb]).sum(axis=1)
                # BPR gradient: d L/d e_u = sigmoid(x_uj - x_ui) * (e_j - e_i)
                coeff = _sigmoid(x_uj - x_ui)[:, None]  # shape (batch, 1)

                np.add.at(grad_E_final, ub,            coeff * (E_items[jb] - E_items[ib]))
                np.add.at(grad_E_final, n_users + ib, -coeff * E_users[ub])
                np.add.at(grad_E_final, n_users + jb,  coeff * E_users[ub])

            # Backward through GCN: grad_E0 = sum_{k=0}^{L} A_norm^k @ grad_E_final / (L+1)
            # A_norm is symmetric so A_norm^T = A_norm — backward = forward passes
            g = grad_E_final / (self.n_layers + 1)
            grad_E0 = g.copy()
            for _ in range(self.n_layers):
                g = A_norm @ g
                grad_E0 += g

            # L2 regularization on initial embeddings only (LightGCN paper Section 3.3)
            grad_E0 += self.regularization * E0

            # Adam update on E0
            t = epoch + 1
            m_adam = b1 * m_adam + (1 - b1) * grad_E0
            v_adam = b2 * v_adam + (1 - b2) * grad_E0 ** 2
            m_hat = m_adam / (1 - b1 ** t)
            v_hat = v_adam / (1 - b2 ** t)
            E0 -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps_adam)

        self._A_norm_ = A_norm
        self._E0_ = E0
        self._n_users_ = n_users
        self._n_items_ = n_items
        self._user_idx_ = user_idx
        self._item_idx_ = item_idx
        self._items_in_train_ = items_seen.tolist()
        return self

    def recommend(
        self,
        users,
        *,
        train_interactions,
        items,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ):
        # Forward pass: deterministic at inference (no randomness)
        E = self._E0_.copy()
        E_sum = self._E0_.copy()
        for _ in range(self.n_layers):
            E = self._A_norm_ @ E
            E_sum += E
        E_final = E_sum / (self.n_layers + 1)

        E_users = E_final[:self._n_users_]
        E_items = E_final[self._n_users_:]

        seen_by_user = self._seen_items_by_user(
            train_interactions,
            user_column=user_column,
            item_column=item_column,
        )
        output_columns = [user_column, item_column, "policy", "rank"]
        rows = []

        for user_id in users[user_column]:
            user_row = self._user_idx_.get(user_id)
            if user_row is None:
                continue

            seen = seen_by_user.get(user_id, set())
            seen_indices = {self._item_idx_[it] for it in seen if it in self._item_idx_}

            scores = E_items @ E_users[user_row]  # shape: (n_items,)
            for idx in seen_indices:
                scores[idx] = -np.inf

            n_candidates = self._n_items_ - len(seen_indices)
            if n_candidates == 0:
                continue
            n_take = min(self.k, n_candidates)
            top_indices = np.argpartition(scores, -n_take)[-n_take:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            top_indices = [i for i in top_indices if not np.isinf(scores[i])]

            selected = [self._items_in_train_[i] for i in top_indices]
            rows.extend(
                self._build_recommendation_rows(
                    user_id,
                    selected,
                    user_column=user_column,
                    item_column=item_column,
                )
            )

        return pd.DataFrame(rows, columns=output_columns)

