from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.scorers import BPRScorer


def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "user_id": [
                "u1",
                "u1",
                "u2",
                "u2",
                "u3",
                "u3",
                "u4",
                "u4",
                "u5",
                "u5",
            ],
            "item_id": [
                "A",
                "B",
                "A",
                "C",
                "A",
                "C",
                "A",
                "C",
                "B",
                "D",
            ],
        }
    )
    # The explicit non-click on B must not become a positive implicit edge.
    y = pd.Series([1, 0, 1, 1, 1, 1, 1, 1, 1, 1], name="target")
    return X, y


def _scorer(seed: int = 0) -> BPRScorer:
    return BPRScorer(
        n_factors=4,
        learning_rate=0.05,
        iterations=100,
        seed=seed,
    )


def test_bpr_scorer_returns_aligned_finite_candidate_scores() -> None:
    X, y = _training_data()
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["C", "A"],
        },
        index=[11, 7],
    )

    scores = _scorer().fit(X, y).score(candidates)

    assert scores.index.tolist() == [11, 7]
    assert scores.name == "score"
    assert scores.notna().all()


def test_bpr_scorer_is_deterministic_for_a_fixed_seed() -> None:
    X, y = _training_data()
    candidates = X[["user_id", "item_id"]].copy()

    first = _scorer(seed=3).fit(X, y).score(candidates)
    second = _scorer(seed=3).fit(X, y).score(candidates)

    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize(
    "candidates, message",
    [
        (
            pd.DataFrame({"user_id": ["cold"], "item_id": ["A"]}),
            "unknown_users=1",
        ),
        (
            pd.DataFrame({"user_id": ["u1"], "item_id": ["cold"]}),
            "unknown_items=1",
        ),
    ],
)
def test_bpr_scorer_rejects_cold_candidate_rows(
    candidates: pd.DataFrame,
    message: str,
) -> None:
    X, y = _training_data()

    with pytest.raises(ValueError, match=message):
        _scorer().fit(X, y).score(candidates)


def test_bpr_scorer_requires_positive_interactions() -> None:
    X, _ = _training_data()
    y = pd.Series([0] * len(X))

    with pytest.raises(ValueError, match="at least one positive"):
        _scorer().fit(X, y)


def test_bpr_scorer_can_use_explicit_zero_fallback_for_cold_rows() -> None:
    X, y = _training_data()
    scorer = BPRScorer(
        n_factors=4,
        iterations=20,
        cold_score_policy="zero",
    ).fit(X, y)
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "cold", "u1"],
            "item_id": ["A", "A", "cold"],
        }
    )

    scores = scorer.score(candidates)

    assert scores.iloc[1:].tolist() == [0.0, 0.0]
    assert scorer.coverage_diagnostics_["fallback_rows"] == 2
