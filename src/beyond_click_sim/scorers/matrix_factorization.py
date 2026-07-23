from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from beyond_click_sim.scorers.base import Scorer


class _ImplicitMFScorer(Scorer):
    """Shared fit/score for implicit-feedback matrix-factorization scorers.

    Mirrors ``ALSPolicy``/``BPRPolicy`` (src/beyond_click_sim/tasks/policies.py)
    but exposes the ``fit(X, y) -> score(X)`` contract. ``fit`` builds a binary
    user-item matrix from the training interactions in ``X`` and fits an
    ``implicit`` model; ``y`` is unused because the interaction-prediction train
    split is positives-only (``target_interact == 1``).

    ``score`` predicts each candidate row as the raw latent dot product
    ``user_factors[u] · item_factors[i]`` — the same quantity the model ranks by,
    computed directly rather than via ``model.recommend`` because we score a
    fixed candidate set. Users or items unseen in training have no factors and
    receive 0.0, so ``nonzero_score_fraction`` is a meaningful coverage
    diagnostic for these baselines.
    """

    def __init__(
        self,
        *,
        n_factors: int,
        iterations: int,
        seed: int,
        item_column: str,
        user_column: str,
        num_threads: int = 1,
    ) -> None:
        if n_factors < 1:
            raise ValueError("n_factors must be positive.")
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        # num_threads=1 keeps implicit's SGD/ALS deterministic across runs for a
        # fixed seed: its parallel (HOGWILD-style) updates are otherwise
        # non-reproducible even with random_state set.
        self.n_factors = n_factors
        self.iterations = iterations
        self.seed = seed
        self.item_column = item_column
        self.user_column = user_column
        self.num_threads = num_threads
        self._user_factors_: np.ndarray | None = None
        self._item_factors_: np.ndarray | None = None
        self._user_idx_: dict | None = None
        self._item_idx_: dict | None = None

    def _build_model(self):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        del y  # positives-only train split; implicit feedback uses pairs only.
        for column in (self.user_column, self.item_column):
            if column not in X.columns:
                raise ValueError(f"Missing column: {column!r}")

        users_seen = X[self.user_column].unique()
        items_seen = X[self.item_column].unique()
        user_idx: dict = {u: i for i, u in enumerate(users_seen)}
        item_idx: dict = {it: i for i, it in enumerate(items_seen)}

        row_ids = X[self.user_column].map(user_idx).to_numpy()
        col_ids = X[self.item_column].map(item_idx).to_numpy()
        user_item = csr_matrix(
            (np.ones(len(X), dtype=np.float32), (row_ids, col_ids)),
            shape=(len(users_seen), len(items_seen)),
        )

        model = self._build_model()
        model.fit(user_item, show_progress=False)

        # implicit returns factors as numpy arrays with use_gpu=False. Both ALS
        # and BPR use factor widths that match between users and items (BPR pads
        # an extra bias column on both), so the plain dot product is valid.
        self._user_factors_ = np.asarray(model.user_factors)
        self._item_factors_ = np.asarray(model.item_factors)
        self._user_idx_ = user_idx
        self._item_idx_ = item_idx
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self._user_factors_ is None or self._item_factors_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")

        result = np.zeros(len(X), dtype=np.float64)
        u = X[self.user_column].map(self._user_idx_)
        i = X[self.item_column].map(self._item_idx_)
        mask = (u.notna() & i.notna()).to_numpy()
        if mask.any():
            uu = u.to_numpy()[mask].astype(int)
            ii = i.to_numpy()[mask].astype(int)
            result[mask] = (
                self._user_factors_[uu] * self._item_factors_[ii]
            ).sum(axis=1)
        return pd.Series(result, index=X.index, name="score")


class ALSScorer(_ImplicitMFScorer):
    """Alternating Least Squares implicit-feedback MF scorer (Hu et al. 2008)."""

    name = "als"

    def __init__(
        self,
        n_factors: int = 64,
        iterations: int = 20,
        regularization: float = 0.01,
        seed: int = 0,
        item_column: str = "item_id",
        user_column: str = "user_id",
        num_threads: int = 1,
    ) -> None:
        super().__init__(
            n_factors=n_factors,
            iterations=iterations,
            seed=seed,
            item_column=item_column,
            user_column=user_column,
            num_threads=num_threads,
        )
        self.regularization = regularization

    def _build_model(self):
        from implicit.als import AlternatingLeastSquares

        return AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
            num_threads=self.num_threads,
            use_gpu=False,
        )


class BPRScorer(_ImplicitMFScorer):
    """Bayesian Personalized Ranking MF scorer (Rendle et al. 2009)."""

    name = "bpr"

    def __init__(
        self,
        n_factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        seed: int = 0,
        item_column: str = "item_id",
        user_column: str = "user_id",
        num_threads: int = 1,
    ) -> None:
        super().__init__(
            n_factors=n_factors,
            iterations=iterations,
            seed=seed,
            item_column=item_column,
            user_column=user_column,
            num_threads=num_threads,
        )
        self.learning_rate = learning_rate
        self.regularization = regularization

    def _build_model(self):
        from implicit.bpr import BayesianPersonalizedRanking

        return BayesianPersonalizedRanking(
            factors=self.n_factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
            num_threads=self.num_threads,
            use_gpu=False,
        )
