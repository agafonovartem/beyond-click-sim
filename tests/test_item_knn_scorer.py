from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.scorers import ItemKNNScorer


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
            ],
            "item_id": [
                "A",
                "B",
                "A",
                "C",
                "B",
                "D",
            ],
        }
    )
    # u1 skipped B.  Its fitted positive profile must therefore contain only A.
    y = pd.Series([1, 0, 1, 1, 1, 1], name="target")
    return X, y


def test_item_knn_uses_only_positive_source_interactions() -> None:
    X, y = _training_data()
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["C", "D"],
        },
        index=[12, 4],
    )

    scores = ItemKNNScorer(n_neighbors=3).fit(X, y).score(candidates)

    assert scores.index.tolist() == [12, 4]
    assert scores.name == "score"
    assert scores.loc[12] > 0
    # D is connected only through skipped B, so it must not contribute.
    assert scores.loc[4] == pytest.approx(0.0)


def test_item_knn_deduplicates_positive_user_item_edges() -> None:
    X, y = _training_data()
    duplicate = X.iloc[[0]].copy()
    duplicated_X = pd.concat([X, duplicate], ignore_index=True)
    duplicated_y = pd.concat([y, pd.Series([1])], ignore_index=True)
    candidates = pd.DataFrame(
        {"user_id": ["u1", "u1"], "item_id": ["C", "D"]}
    )

    original = ItemKNNScorer(n_neighbors=3).fit(X, y).score(candidates)
    duplicated = (
        ItemKNNScorer(n_neighbors=3)
        .fit(duplicated_X, duplicated_y)
        .score(candidates)
    )

    pd.testing.assert_series_equal(original, duplicated)


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
def test_item_knn_rejects_cold_candidate_rows(
    candidates: pd.DataFrame,
    message: str,
) -> None:
    X, y = _training_data()

    with pytest.raises(ValueError, match=message):
        ItemKNNScorer().fit(X, y).score(candidates)


def test_item_knn_requires_binary_positive_interactions() -> None:
    X, _ = _training_data()

    with pytest.raises(ValueError, match="binary"):
        ItemKNNScorer().fit(X, pd.Series([1, 0, 1, 1, 1, 2]))

    with pytest.raises(ValueError, match="at least one positive"):
        ItemKNNScorer().fit(X, pd.Series([0] * len(X)))


def test_item_knn_can_use_explicit_zero_fallback() -> None:
    X, y = _training_data()
    scorer = ItemKNNScorer(
        n_neighbors=2,
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
