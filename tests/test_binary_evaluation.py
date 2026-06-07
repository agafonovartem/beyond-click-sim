from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from beyond_click_sim.evaluation import (
    apply_threshold,
    binary_classification_metrics,
    find_best_threshold,
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


def test_find_best_threshold_rejects_unsupported_metric() -> None:
    y_true = pd.Series([1])
    scores = pd.Series([0.5])

    with pytest.raises(ValueError, match="Unsupported metric"):
        find_best_threshold(y_true, scores, metric="auc")  # type: ignore[arg-type]


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
