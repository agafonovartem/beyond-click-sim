from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from beyond_click_sim.scorers.base import Scorer


class HistoryTextAffinityScorer(Scorer):
    """Rank candidates by affinity to a user's signed textual history profile.

    Text is represented with character TF-IDF, which avoids language-specific
    tokenization assumptions and is suitable for the native Chinese captions in
    KuaiRand.  A user's profile is the mean vector of clicked history items minus
    the mean vector of non-clicked history items.  The returned score is cosine
    similarity between that signed profile and each candidate.
    """

    name = "history_text_affinity"

    def __init__(
        self,
        *,
        text_columns: tuple[str, ...],
        user_column: str = "user_id",
        analyzer: str = "char",
        ngram_range: tuple[int, int] = (2, 5),
        min_df: int | float = 1,
        max_features: int | None = None,
        cold_score_policy: Literal["error", "zero"] = "error",
    ) -> None:
        if not text_columns:
            raise ValueError("text_columns must be non-empty")
        self.text_columns = text_columns
        self.user_column = user_column
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        if cold_score_policy not in {"error", "zero"}:
            raise ValueError(
                "cold_score_policy must be either 'error' or 'zero'"
            )
        self.cold_score_policy = cold_score_policy
        self.vectorizer_: TfidfVectorizer | None = None
        self.profile_by_user_: dict[object, csr_matrix] | None = None
        self.fit_diagnostics_: dict[str, int | float] | None = None
        self.coverage_diagnostics_: dict[str, int | float] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._require_columns(X)
        if y.isna().any():
            raise ValueError(
                "HistoryTextAffinityScorer targets must not contain missing values"
            )
        if X[self.user_column].isna().any():
            raise ValueError(
                "HistoryTextAffinityScorer user IDs must not contain missing values"
            )

        texts = self._combine_text(X)
        if not texts.str.strip().any():
            raise ValueError("HistoryTextAffinityScorer requires non-empty text")

        vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            lowercase=False,
            norm="l2",
        )
        item_vectors = vectorizer.fit_transform(texts)
        labels = y.astype(bool).to_numpy()

        # Build every signed user profile in one sparse matrix multiplication.
        # A Python loop that repeatedly evaluates ``user_values == user_id`` is
        # quadratic in users x rows on full recommendation logs.
        user_codes, source_users = pd.factorize(
            X[self.user_column],
            sort=False,
        )
        n_source_users = int(len(source_users))
        positive_counts = np.bincount(
            user_codes[labels],
            minlength=n_source_users,
        )
        negative_counts = np.bincount(
            user_codes[~labels],
            minlength=n_source_users,
        )
        profile_user_mask = (positive_counts > 0) & (negative_counts > 0)
        profile_user_codes = np.flatnonzero(profile_user_mask)
        profile_row_mask = profile_user_mask[user_codes]
        profile_row_positions = np.flatnonzero(profile_row_mask)
        profile_row_user_codes = user_codes[profile_row_mask]
        profile_row_labels = labels[profile_row_mask]
        profile_row_weights = np.where(
            profile_row_labels,
            1.0 / positive_counts[profile_row_user_codes],
            -1.0 / negative_counts[profile_row_user_codes],
        )
        user_row_aggregation = csr_matrix(
            (
                profile_row_weights,
                (profile_row_user_codes, profile_row_positions),
            ),
            shape=(n_source_users, len(X)),
        )
        profile_matrix = normalize(
            user_row_aggregation @ item_vectors,
            norm="l2",
        )
        profiles = {
            source_users[user_code]: csr_matrix(
                profile_matrix.getrow(user_code)
            )
            for user_code in profile_user_codes
        }

        n_profile_users = int(len(profile_user_codes))
        skipped_constant_label_users = n_source_users - n_profile_users
        self.fit_diagnostics_ = {
            "n_source_rows": int(len(X)),
            "n_source_users": n_source_users,
            "n_profile_users": n_profile_users,
            "n_skipped_constant_label_users": int(
                skipped_constant_label_users
            ),
            "profile_user_fraction": (
                float(n_profile_users / n_source_users)
                if n_source_users
                else 0.0
            ),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        }
        self.vectorizer_ = vectorizer
        self.profile_by_user_ = profiles
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.vectorizer_ is None or self.profile_by_user_ is None:
            raise RuntimeError("HistoryTextAffinityScorer is not fitted")
        self._require_columns(X)

        unknown_users = sorted(
            set(X[self.user_column]) - set(self.profile_by_user_),
            key=repr,
        )
        if unknown_users and self.cold_score_policy == "error":
            raise ValueError(
                "HistoryTextAffinityScorer cannot score users without profiles: "
                f"count={len(unknown_users)}, examples={unknown_users[:3]}"
            )

        candidate_vectors = self.vectorizer_.transform(self._combine_text(X))
        known_mask = X[self.user_column].isin(self.profile_by_user_).to_numpy()
        known_positions = np.flatnonzero(known_mask)
        values = np.zeros(len(X), dtype=float)
        if len(known_positions):
            profiles = vstack(
                [
                    self.profile_by_user_[X.iloc[position][self.user_column]]
                    for position in known_positions
                ]
            )
            values[known_positions] = np.asarray(
                candidate_vectors[known_positions]
                .multiply(profiles)
                .sum(axis=1)
            ).reshape(-1)
        self.coverage_diagnostics_ = {
            "n_target_rows": int(len(X)),
            "n_scored_profile_rows": int(known_mask.sum()),
            "n_zero_fallback_rows": int((~known_mask).sum()),
            "n_target_users": int(X[self.user_column].nunique()),
            "n_users_without_profiles": int(len(unknown_users)),
            "profile_row_fraction": (
                float(known_mask.mean()) if len(known_mask) else 0.0
            ),
        }
        return pd.Series(values, index=X.index, dtype=float, name="score")

    def _combine_text(self, X: pd.DataFrame) -> pd.Series:
        normalized = [
            X[column].fillna("").astype(str).str.strip()
            for column in self.text_columns
        ]
        combined = normalized[0]
        for values in normalized[1:]:
            combined = combined.str.cat(values, sep=" ")
        return combined.str.strip()

    def _require_columns(self, X: pd.DataFrame) -> None:
        required = [self.user_column, *self.text_columns]
        missing = [column for column in required if column not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
