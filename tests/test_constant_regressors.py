from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.scorers import (
    HistoryMeanRegressor,
    HistoryModeRegressor,
    MeanRegressor,
    ModeRegressor,
)
from beyond_click_sim.scorers.constant import (
    select_user_history_positions,
    select_user_history_rows,
)


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


def test_mode_regressor_predicts_train_target_mode() -> None:
    X_train = pd.DataFrame({"item_id": ["i1", "i2", "i3", "i4"]})
    y_train = pd.Series([1, 4, 4, 5], name="target")
    X_test = pd.DataFrame({"item_id": ["i5", "i6"]}, index=["a", "b"])

    scorer = ModeRegressor().fit(X_train, y_train)
    scores = scorer.score(X_test)

    assert scorer.mode_ == pytest.approx(4.0)
    assert scorer.tie_break == "smallest"
    assert scores.tolist() == [4.0, 4.0]
    assert list(scores.index) == ["a", "b"]
    assert scores.name == "score"


def test_mode_regressor_breaks_ties_with_smallest_value() -> None:
    scorer = ModeRegressor().fit(
        pd.DataFrame({"item_id": ["i1", "i2", "i3", "i4"]}),
        pd.Series([5, 1, 5, 1], name="target"),
    )

    assert scorer.mode_ == pytest.approx(1.0)


def test_mode_regressor_requires_fit_before_score() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        ModeRegressor().score(pd.DataFrame({"item_id": ["i1"]}))


def test_mode_regressor_rejects_invalid_fit_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        ModeRegressor().fit(pd.DataFrame({"item_id": ["i1"]}), pd.Series([1, 2]))

    with pytest.raises(ValueError, match="empty targets"):
        ModeRegressor().fit(pd.DataFrame({"item_id": []}), pd.Series([], dtype=float))

    with pytest.raises(ValueError, match="NA values"):
        ModeRegressor().fit(pd.DataFrame({"item_id": ["i1"]}), pd.Series([pd.NA]))


def test_select_user_history_rows_keeps_last_rows_in_input_order() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u1", "u1", "u2", "u1"],
            "rating": [5, 1, 2, 4, 3, 1],
            "row": ["a", "b", "c", "d", "e", "f"],
        }
    )

    selected = select_user_history_rows(
        X_train,
        user_column="user_id",
        max_history_items=3,
    )

    assert selected["u1"]["row"].tolist() == ["c", "d", "f"]
    assert selected["u1"]["rating"].tolist() == [2, 4, 1]
    assert selected["u2"]["row"].tolist() == ["b", "e"]
    assert select_user_history_positions(
        X_train,
        user_column="user_id",
        max_history_items=3,
    ) == {"u1": [2, 3, 5], "u2": [1, 4]}


def test_history_mean_regressor_predicts_user_prompt_window_mean() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2", "u2"],
            "rating": [5, 1, 4, 2, 1, 3],
        }
    )
    X_test = pd.DataFrame({"user_id": ["u1", "u2", "u1"]}, index=["a", "b", "c"])

    scorer = HistoryMeanRegressor(
        history_value_column="rating",
        max_history_items=3,
    ).fit(X_train, pd.Series([5, 1, 4, 2, 1, 3]))
    scores = scorer.score(X_test)

    assert scorer.history_mean_by_user_ == {"u1": pytest.approx(7 / 3), "u2": 2.0}
    assert scorer.history_count_by_user_ == {"u1": 3, "u2": 2}
    assert scores.tolist() == pytest.approx([7 / 3, 2.0, 7 / 3])
    assert list(scores.index) == ["a", "b", "c"]
    assert scores.name == "score"


def test_history_mode_regressor_predicts_user_prompt_window_mode() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2", "u2"],
            "rating": [1, 5, 5, 1, 4, 2],
        }
    )
    X_test = pd.DataFrame({"user_id": ["u1", "u2"]}, index=["a", "b"])

    scorer = HistoryModeRegressor(
        history_value_column="rating",
        max_history_items=4,
    ).fit(X_train, pd.Series([1, 5, 5, 1, 4, 2]))
    scores = scorer.score(X_test)

    assert scorer.history_mode_by_user_ == {"u1": 1.0, "u2": 2.0}
    assert scorer.history_count_by_user_ == {"u1": 4, "u2": 2}
    assert scorer.tie_break == "smallest"
    assert scores.tolist() == [1.0, 2.0]
    assert list(scores.index) == ["a", "b"]


def test_history_regressors_require_fit_before_score() -> None:
    X_test = pd.DataFrame({"user_id": ["u1"]})

    with pytest.raises(RuntimeError, match="not fitted"):
        HistoryMeanRegressor(history_value_column="rating").score(X_test)

    with pytest.raises(RuntimeError, match="not fitted"):
        HistoryModeRegressor(history_value_column="rating").score(X_test)


def test_history_regressors_reject_unknown_score_users() -> None:
    X_train = pd.DataFrame({"user_id": ["u1"], "rating": [5]})
    y_train = pd.Series([5])
    X_test = pd.DataFrame({"user_id": ["u2"]})

    with pytest.raises(ValueError, match="Missing fitted history"):
        HistoryMeanRegressor(history_value_column="rating").fit(X_train, y_train).score(X_test)

    with pytest.raises(ValueError, match="Missing fitted history"):
        HistoryModeRegressor(history_value_column="rating").fit(X_train, y_train).score(X_test)


def test_history_regressors_reject_invalid_fit_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        HistoryMeanRegressor(history_value_column="rating").fit(
            pd.DataFrame({"user_id": ["u1"], "rating": [5]}),
            pd.Series([5, 4]),
        )

    with pytest.raises(ValueError, match="Missing required columns"):
        HistoryModeRegressor(history_value_column="rating").fit(
            pd.DataFrame({"user_id": ["u1"]}),
            pd.Series([5]),
        )

    with pytest.raises(ValueError, match="History values contain NA"):
        HistoryMeanRegressor(history_value_column="rating").fit(
            pd.DataFrame({"user_id": ["u1"], "rating": [pd.NA]}),
            pd.Series([5]),
        )
