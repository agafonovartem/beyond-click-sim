from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.scorers import ExposureLogisticBiasScorer


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
            ],
            "item_id": ["A", "B", "A", "C", "B", "C", "A", "C"],
        }
    )
    y = pd.Series([1, 0, 0, 0, 1, 1, 1, 0], name="target")
    return X, y


def test_exposure_logistic_uses_non_clicks_and_scores_all_zero_user() -> None:
    X, y = _training_data()
    scorer = ExposureLogisticBiasScorer(seed=7).fit(X, y)
    candidates = pd.DataFrame(
        {
            "user_id": ["u2", "u2"],
            "item_id": ["A", "C"],
        }
    )

    scores = scorer.score(candidates)

    assert scores.between(0.0, 1.0).all()
    assert scorer.fit_diagnostics_["uses_positive_and_negative_exposures"] is True
    assert scorer.fit_diagnostics_["positive_rate"] == pytest.approx(0.5)
    assert scorer.coverage_diagnostics_ == {
        "rows": 2,
        "unknown_user_rows": 0,
        "unknown_item_rows": 0,
        "fallback_rows": 0,
        "cold_score_policy": "error",
    }


def test_exposure_logistic_is_deterministic() -> None:
    X, y = _training_data()

    first = ExposureLogisticBiasScorer(seed=3).fit(X, y).score(X)
    second = ExposureLogisticBiasScorer(seed=3).fit(X, y).score(X)

    pd.testing.assert_series_equal(first, second)


def test_exposure_logistic_rejects_cold_nodes() -> None:
    X, y = _training_data()
    scorer = ExposureLogisticBiasScorer().fit(X, y)

    with pytest.raises(ValueError, match="cold users/items"):
        scorer.score(pd.DataFrame({"user_id": ["cold"], "item_id": ["A"]}))
