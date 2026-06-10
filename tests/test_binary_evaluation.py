from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from beyond_click_sim.evaluation import (
    apply_threshold,
    binary_classification_metrics,
    find_best_group_threshold,
    find_best_threshold,
    find_best_threshold_by_metric,
    find_best_user_group_threshold,
    grouped_binary_classification_metrics,
    user_grouped_binary_classification_metrics,
)


def test_apply_threshold_returns_boolean_predictions_with_input_index() -> None:
    scores = pd.Series([0.1, 0.5, 0.9], index=["a", "b", "c"], name="score")

    predictions = apply_threshold(scores, threshold=0.5)

    assert predictions.tolist() == [False, True, True]
    assert list(predictions.index) == ["a", "b", "c"]
    assert predictions.name == "prediction"


def test_binary_classification_metrics_use_sklearn_semantics() -> None:
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([1, 0, 1, 0])

    metrics = binary_classification_metrics(y_true, y_pred)

    assert metrics == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "n": 4,
        "n_positive": 2,
        "n_predicted_positive": 2,
    }


def test_binary_classification_metrics_use_zero_division_zero() -> None:
    y_true = pd.Series([1, 0, 0])
    y_pred = pd.Series([0, 0, 0])

    metrics = binary_classification_metrics(y_true, y_pred)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["n_predicted_positive"] == 0


def test_grouped_binary_classification_metrics_average_groups_equally() -> None:
    y_true = pd.Series([1, 0, 0, 0, 1, 0])
    y_pred = pd.Series([1, 1, 1, 1, 1, 0])
    groups = pd.Series(["large", "large", "large", "large", "small", "small"])

    metrics = grouped_binary_classification_metrics(y_true, y_pred, groups)

    assert metrics["accuracy"] == pytest.approx((0.25 + 1.0) / 2)
    assert metrics["precision"] == pytest.approx((0.25 + 1.0) / 2)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx((0.4 + 1.0) / 2)
    assert metrics["n_groups"] == 2
    assert metrics["n"] == 6
    assert metrics["n_positive"] == 2
    assert metrics["n_predicted_positive"] == 5


def test_user_grouped_binary_classification_metrics_average_groups_then_users() -> None:
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    y_pred = pd.Series([1, 1, 0, 0, 1, 0])
    groups = pd.Series(["u1-g1", "u1-g1", "u1-g2", "u1-g2", "u2-g1", "u2-g1"])
    users = pd.Series(["u1", "u1", "u1", "u1", "u2", "u2"])

    group_metrics = grouped_binary_classification_metrics(y_true, y_pred, groups)
    user_group_metrics = user_grouped_binary_classification_metrics(
        y_true,
        y_pred,
        groups,
        users,
    )

    assert group_metrics["f1"] == pytest.approx(((2 / 3) + 0.0 + 1.0) / 3)
    assert user_group_metrics["f1"] == pytest.approx((((2 / 3) + 0.0) / 2 + 1.0) / 2)
    assert user_group_metrics["f1"] != pytest.approx(group_metrics["f1"])
    assert user_group_metrics["accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert user_group_metrics["precision"] == pytest.approx((0.25 + 1.0) / 2)
    assert user_group_metrics["recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert user_group_metrics["n_users"] == 2
    assert user_group_metrics["n_groups"] == 3
    assert user_group_metrics["n"] == 6
    assert user_group_metrics["n_positive"] == 3
    assert user_group_metrics["n_predicted_positive"] == 3


def test_user_grouped_binary_classification_matches_group_macro_for_one_group_per_user() -> None:
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 1, 1, 0])
    groups = pd.Series(["u1-g1", "u1-g1", "u2-g1", "u2-g1"])
    users = pd.Series(["u1", "u1", "u2", "u2"])

    group_metrics = grouped_binary_classification_metrics(y_true, y_pred, groups)
    user_group_metrics = user_grouped_binary_classification_metrics(
        y_true,
        y_pred,
        groups,
        users,
    )

    for metric in ["accuracy", "precision", "recall", "f1"]:
        assert user_group_metrics[metric] == pytest.approx(group_metrics[metric])


def test_find_best_threshold_selects_threshold_by_metric() -> None:
    y_true = pd.Series([1, 1, 0, 0])
    scores = pd.Series([0.9, 0.8, 0.7, 0.1])

    selection = find_best_threshold(y_true, scores, metric="f1")

    assert selection == {
        "threshold": 0.8,
        "metric": "f1",
        "metric_value": 1.0,
    }


def test_find_best_threshold_allows_all_negative_predictions() -> None:
    y_true = pd.Series([0, 0])
    scores = pd.Series([0.1, 0.2])
    expected_threshold = float(np.nextafter(0.2, np.inf))

    selection = find_best_threshold(y_true, scores, metric="accuracy")

    assert selection == {
        "threshold": expected_threshold,
        "metric": "accuracy",
        "metric_value": 1.0,
    }


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1"])
def test_find_best_threshold_matches_brute_force(metric: str) -> None:
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    scores = pd.Series([0.3, 0.3, 0.8, 0.1, 0.5, 0.8])
    unique_scores = sorted(float(score) for score in scores.dropna().unique())
    thresholds = [float(np.nextafter(unique_scores[-1], np.inf)), *unique_scores]

    brute_threshold = thresholds[0]
    brute_value = -1.0
    for threshold in thresholds:
        value = binary_classification_metrics(
            y_true,
            apply_threshold(scores, threshold),
        )[metric]
        if value > brute_value:
            brute_threshold = threshold
            brute_value = float(value)

    selection = find_best_threshold(y_true, scores, metric=metric)  # type: ignore[arg-type]

    assert selection["threshold"] == brute_threshold
    assert selection["metric"] == metric
    assert selection["metric_value"] == pytest.approx(brute_value)


def test_find_best_threshold_by_metric_uses_custom_metric_function() -> None:
    y_true = pd.Series([1, 0, 0, 0, 1, 0])
    scores = pd.Series([0.9, 0.8, 0.7, 0.6, 0.1, 0.0])
    groups = pd.Series(["large", "large", "large", "large", "small", "small"])

    selection = find_best_threshold_by_metric(
        y_true,
        scores,
        metric_fn=lambda y, pred: grouped_binary_classification_metrics(
            y,
            pred,
            groups,
        )["f1"],
        metric_name="macro_group_f1",
    )

    assert selection == {
        "threshold": 0.1,
        "metric": "macro_group_f1",
        "metric_value": pytest.approx(0.7),
    }


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1"])
def test_find_best_group_threshold_matches_brute_force(metric: str) -> None:
    y_true = pd.Series([1, 0, 0, 0, 1, 0])
    scores = pd.Series([0.9, 0.8, 0.7, 0.6, 0.1, 0.0])
    groups = pd.Series(["large", "large", "large", "large", "small", "small"])
    unique_scores = sorted(float(score) for score in scores.dropna().unique())
    thresholds = [float(np.nextafter(unique_scores[-1], np.inf)), *unique_scores]

    brute_threshold = thresholds[0]
    brute_value = -1.0
    for threshold in thresholds:
        value = grouped_binary_classification_metrics(
            y_true,
            apply_threshold(scores, threshold),
            groups,
        )[metric]
        if value > brute_value:
            brute_threshold = threshold
            brute_value = float(value)

    selection = find_best_group_threshold(
        y_true,
        scores,
        groups,
        metric=metric,  # type: ignore[arg-type]
    )

    assert selection["threshold"] == brute_threshold
    assert selection["metric"] == f"macro_group_{metric}"
    assert selection["metric_value"] == pytest.approx(brute_value)


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1"])
def test_find_best_user_group_threshold_matches_brute_force(metric: str) -> None:
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    scores = pd.Series([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    groups = pd.Series(["u1-g1", "u1-g1", "u1-g2", "u1-g2", "u2-g1", "u2-g1"])
    users = pd.Series(["u1", "u1", "u1", "u1", "u2", "u2"])
    unique_scores = sorted(float(score) for score in scores.dropna().unique())
    thresholds = [float(np.nextafter(unique_scores[-1], np.inf)), *unique_scores]

    brute_threshold = thresholds[0]
    brute_value = -1.0
    for threshold in thresholds:
        value = user_grouped_binary_classification_metrics(
            y_true,
            apply_threshold(scores, threshold),
            groups,
            users,
        )[metric]
        if value > brute_value:
            brute_threshold = threshold
            brute_value = float(value)

    selection = find_best_user_group_threshold(
        y_true,
        scores,
        groups,
        users,
        metric=metric,  # type: ignore[arg-type]
    )

    assert selection["threshold"] == brute_threshold
    assert selection["metric"] == f"macro_by_user_group_mean_{metric}"
    assert selection["metric_value"] == pytest.approx(brute_value)


def test_find_best_threshold_rejects_unsupported_metric() -> None:
    y_true = pd.Series([1])
    scores = pd.Series([0.5])

    with pytest.raises(ValueError, match="Unsupported metric"):
        find_best_threshold(y_true, scores, metric="auc")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported metric"):
        find_best_group_threshold(
            y_true,
            scores,
            pd.Series(["a"]),
            metric="auc",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Unsupported metric"):
        find_best_user_group_threshold(
            y_true,
            scores,
            pd.Series(["a"]),
            pd.Series(["u"]),
            metric="auc",  # type: ignore[arg-type]
        )


def test_find_best_threshold_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="empty scores"):
        find_best_threshold(pd.Series([], dtype=int), pd.Series([], dtype=float))


def test_find_best_threshold_rejects_nan_scores() -> None:
    y_true = pd.Series([1, 0])
    scores = pd.Series([0.5, float("nan")])

    with pytest.raises(ValueError, match="NaN values"):
        find_best_threshold(y_true, scores)


def test_binary_helpers_require_same_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        binary_classification_metrics(pd.Series([1]), pd.Series([1, 0]))

    with pytest.raises(ValueError, match="same length"):
        find_best_threshold(pd.Series([1]), pd.Series([0.1, 0.2]))

    with pytest.raises(ValueError, match="same length"):
        find_best_group_threshold(
            pd.Series([1]),
            pd.Series([0.1]),
            pd.Series(["a", "b"]),
        )

    with pytest.raises(ValueError, match="same length"):
        grouped_binary_classification_metrics(
            pd.Series([1]),
            pd.Series([1]),
            pd.Series(["a", "b"]),
        )

    with pytest.raises(ValueError, match="same length"):
        user_grouped_binary_classification_metrics(
            pd.Series([1]),
            pd.Series([1]),
            pd.Series(["a"]),
            pd.Series(["u", "v"]),
        )

    with pytest.raises(ValueError, match="same length"):
        find_best_user_group_threshold(
            pd.Series([1]),
            pd.Series([0.1]),
            pd.Series(["a"]),
            pd.Series(["u", "v"]),
        )

    with pytest.raises(ValueError, match="same length"):
        find_best_threshold_by_metric(
            pd.Series([1]),
            pd.Series([0.1, 0.2]),
            metric_fn=lambda y, pred: 0.0,
            metric_name="custom",
        )
