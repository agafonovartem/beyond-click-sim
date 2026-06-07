from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.scorers import PopularityScorer


def test_popularity_scorer_scores_items_by_train_target_sum() -> None:
    X_train = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i3"],
            "feature": [10, 20, 30, 40],
        }
    )
    y_train = pd.Series([1, 0.5, 2, 0], name="target")
    X_test = pd.DataFrame(
        {"item_id": ["i2", "i1", "missing", "i3"]},
        index=["a", "b", "c", "d"],
    )

    scores = PopularityScorer().fit(X_train, y_train).score(X_test)

    assert scores.tolist() == [2.0, 1.5, 0.0, 0.0]
    assert list(scores.index) == ["a", "b", "c", "d"]
    assert scores.name == "score"


def test_popularity_scorer_supports_custom_item_column() -> None:
    X_train = pd.DataFrame({"movie_id": [10, 20, 10]})
    y_train = pd.Series([1, 1, 1], name="target")
    X_test = pd.DataFrame({"movie_id": [20, 30, 10]})

    scores = PopularityScorer(item_column="movie_id").fit(X_train, y_train).score(X_test)

    assert scores.tolist() == [1.0, 0.0, 2.0]


def test_popularity_scorer_requires_fit_before_score() -> None:
    X_test = pd.DataFrame({"item_id": ["i1"]})

    with pytest.raises(RuntimeError, match="not fitted"):
        PopularityScorer().score(X_test)


def test_popularity_scorer_requires_item_column_in_fit() -> None:
    X_train = pd.DataFrame({"other_id": ["i1"]})
    y_train = pd.Series([1], name="target")

    with pytest.raises(ValueError, match="Missing item column"):
        PopularityScorer().fit(X_train, y_train)


def test_popularity_scorer_requires_same_fit_lengths() -> None:
    X_train = pd.DataFrame({"item_id": ["i1", "i2"]})
    y_train = pd.Series([1], name="target")

    with pytest.raises(ValueError, match="same length"):
        PopularityScorer().fit(X_train, y_train)


def test_popularity_scorer_requires_item_column_in_score() -> None:
    X_train = pd.DataFrame({"item_id": ["i1"]})
    y_train = pd.Series([1], name="target")
    X_test = pd.DataFrame({"other_id": ["i1"]})

    scorer = PopularityScorer().fit(X_train, y_train)

    with pytest.raises(ValueError, match="Missing item column"):
        scorer.score(X_test)
