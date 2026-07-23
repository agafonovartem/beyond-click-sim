from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from beyond_click_sim.scorers.base import Scorer


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LightGCNScorer(Scorer):
    """LightGCN implicit-feedback scorer (He et al. 2020).

    A single-step scorer mirroring ``LightGCNPolicy``
    (src/beyond_click_sim/tasks/policies.py): it builds a bipartite user-item
    graph, learns initial embeddings ``E0`` via BPR loss and Adam over fixed
    symmetric-normalized propagation layers (pure NumPy/SciPy — the GCN layers
    have no trainable weights), and takes the layer-mean as the final
    embeddings. ``y`` is unused because the interaction-prediction train split is
    positives-only (``target_interact == 1``).

    ``score`` predicts each candidate row as the latent dot product
    ``E_users[u] · E_items[i]`` — the same quantity the policy ranks by. Users or
    items unseen in training receive 0.0, so ``nonzero_score_fraction`` is a
    meaningful coverage diagnostic.
    """

    name = "lightgcn"

    def __init__(
        self,
        n_factors: int = 64,
        n_layers: int = 3,
        learning_rate: float = 0.001,
        regularization: float = 1e-4,
        iterations: int = 200,
        batch_size: int = 2048,
        seed: int = 0,
        item_column: str = "item_id",
        user_column: str = "user_id",
    ) -> None:
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
        self.seed = seed
        self.item_column = item_column
        self.user_column = user_column
        self._user_embeddings_: np.ndarray | None = None
        self._item_embeddings_: np.ndarray | None = None
        self._user_idx_: dict | None = None
        self._item_idx_: dict | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        del y  # positives-only train split; BPR uses (user, item) pairs only.
        from scipy.sparse import diags, hstack, vstack

        for column in (self.user_column, self.item_column):
            if column not in X.columns:
                raise ValueError(f"Missing column: {column!r}")

        # Deduplicate: treat interactions as binary
        unique_ints = X[[self.user_column, self.item_column]].drop_duplicates()
        users_seen = unique_ints[self.user_column].unique()
        items_seen = unique_ints[self.item_column].unique()
        n_users = len(users_seen)
        n_items = len(items_seen)

        user_idx = {u: i for i, u in enumerate(users_seen)}
        item_idx = {it: i for i, it in enumerate(items_seen)}

        u_ids = unique_ints[self.user_column].map(user_idx).to_numpy()
        i_ids = unique_ints[self.item_column].map(item_idx).to_numpy()
        ones32 = np.ones(len(unique_ints), dtype=np.float32)

        # Binary user-item matrix R (n_users x n_items)
        R = csr_matrix((ones32, (u_ids, i_ids)), shape=(n_users, n_items))

        # Bipartite adjacency A = [[0, R], [R^T, 0]]
        A = vstack([
            hstack([csr_matrix((n_users, n_users), dtype=np.float32), R]),
            hstack([R.T.tocsr(), csr_matrix((n_items, n_items), dtype=np.float32)]),
        ]).tocsr()

        # Symmetric normalization: A_norm = D^(-1/2) A D^(-1/2)
        deg = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0).astype(np.float32)
        D_inv_sqrt = diags(d_inv_sqrt)
        A_norm = (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()

        # Initialize E0 ~ N(0, 0.1^2)
        rng = np.random.default_rng(self.seed)
        E0 = (rng.standard_normal((n_users + n_items, self.n_factors)) * 0.1).astype(
            np.float64
        )

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

            grad_E_final = np.zeros_like(E0)
            for start in range(0, n_pairs, self.batch_size):
                end = min(start + self.batch_size, n_pairs)
                ub = u_ids[start:end]
                ib = i_ids[start:end]
                jb = j_neg[start:end]

                x_ui = (E_users[ub] * E_items[ib]).sum(axis=1)
                x_uj = (E_users[ub] * E_items[jb]).sum(axis=1)
                coeff = _sigmoid(x_uj - x_ui)[:, None]

                np.add.at(grad_E_final, ub,            coeff * (E_items[jb] - E_items[ib]))
                np.add.at(grad_E_final, n_users + ib, -coeff * E_users[ub])
                np.add.at(grad_E_final, n_users + jb,  coeff * E_users[ub])

            # Backward through GCN (A_norm symmetric → backward = forward passes)
            g = grad_E_final / (self.n_layers + 1)
            grad_E0 = g.copy()
            for _ in range(self.n_layers):
                g = A_norm @ g
                grad_E0 += g

            # L2 regularization on initial embeddings only (paper Section 3.3)
            grad_E0 += self.regularization * E0

            # Adam update on E0
            t = epoch + 1
            m_adam = b1 * m_adam + (1 - b1) * grad_E0
            v_adam = b2 * v_adam + (1 - b2) * grad_E0 ** 2
            m_hat = m_adam / (1 - b1 ** t)
            v_hat = v_adam / (1 - b2 ** t)
            E0 -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Final propagation (deterministic at inference, no randomness)
        E = E0.copy()
        E_sum = E0.copy()
        for _ in range(self.n_layers):
            E = A_norm @ E
            E_sum += E
        E_final = E_sum / (self.n_layers + 1)

        self._user_embeddings_ = E_final[:n_users]
        self._item_embeddings_ = E_final[n_users:]
        self._user_idx_ = user_idx
        self._item_idx_ = item_idx
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self._user_embeddings_ is None or self._item_embeddings_ is None:
            raise RuntimeError("LightGCNScorer is not fitted")

        result = np.zeros(len(X), dtype=np.float64)
        u = X[self.user_column].map(self._user_idx_)
        i = X[self.item_column].map(self._item_idx_)
        mask = (u.notna() & i.notna()).to_numpy()
        if mask.any():
            uu = u.to_numpy()[mask].astype(int)
            ii = i.to_numpy()[mask].astype(int)
            result[mask] = (
                self._user_embeddings_[uu] * self._item_embeddings_[ii]
            ).sum(axis=1)
        return pd.Series(result, index=X.index, name="score")
