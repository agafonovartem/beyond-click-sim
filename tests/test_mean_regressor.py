from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.scorers import MeanRegressor


def test_mean_regressor_predicts_train_target_mean() -> None:
    X_train = pd.DataFrame({"item_id": ["i1", "i2", "i3"]})
    y_train = pd.Series([1, 3, 5], name="target")
    X_test = pd.DataFrame({"item_id": ["i4", "i5"]}, index=["a", "b"])

    scorer = MeanRegressor().fit(X_train, y_train)
    scores = scorer.score(X_test)

    assert scorer.mean_ == pytest.approx(3.0)
    assert scores.tolist() == [3.0, 3.0]
    assert list(scores.index) == ["a", "b"]
    assert scores.name == "score"


def test_mean_regressor_requires_fit_before_score() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        MeanRegressor().score(pd.DataFrame({"item_id": ["i1"]}))


def test_mean_regressor_rejects_invalid_fit_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        MeanRegressor().fit(pd.DataFrame({"item_id": ["i1"]}), pd.Series([1, 2]))

    with pytest.raises(ValueError, match="empty targets"):
        MeanRegressor().fit(pd.DataFrame({"item_id": []}), pd.Series([], dtype=float))

    with pytest.raises(ValueError, match="NA values"):
        MeanRegressor().fit(pd.DataFrame({"item_id": ["i1"]}), pd.Series([pd.NA]))
