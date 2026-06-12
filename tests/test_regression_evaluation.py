from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.evaluation import (
    regression_metrics,
    user_grouped_regression_metrics,
)


def test_regression_metrics_compute_mae_and_rmse() -> None:
    y_true = pd.Series([1.0, 3.0, 5.0])
    scores = pd.Series([2.0, 3.0, 3.0])

    metrics = regression_metrics(y_true, scores)

    assert metrics == {
        "mae": pytest.approx(1.0),
        "rmse": pytest.approx((5 / 3) ** 0.5),
        "n": 3,
    }


def test_user_grouped_regression_metrics_average_users_equally() -> None:
    y_true = pd.Series([1.0, 3.0, 5.0])
    scores = pd.Series([2.0, 4.0, 3.0])
    users = pd.Series(["u1", "u1", "u2"])

    metrics = user_grouped_regression_metrics(y_true, scores, users)

    assert metrics["mae"] == pytest.approx((1.0 + 2.0) / 2)
    assert metrics["rmse"] == pytest.approx((1.0 + 2.0) / 2)
    assert metrics["n_users"] == 2
    assert metrics["n"] == 3


def test_regression_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        regression_metrics(pd.Series([1.0]), pd.Series([1.0, 2.0]))

    with pytest.raises(ValueError, match="empty inputs"):
        regression_metrics(pd.Series([], dtype=float), pd.Series([], dtype=float))

    with pytest.raises(ValueError, match="NA values"):
        regression_metrics(pd.Series([pd.NA]), pd.Series([1.0]))

    with pytest.raises(ValueError, match="NaN values"):
        regression_metrics(pd.Series([1.0]), pd.Series([float("nan")]))

    with pytest.raises(ValueError, match="users contains NA"):
        user_grouped_regression_metrics(
            pd.Series([1.0]),
            pd.Series([1.0]),
            pd.Series([pd.NA]),
        )
