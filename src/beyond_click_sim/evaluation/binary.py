from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


SelectionMetric = Literal["accuracy", "precision", "recall", "f1"]


def apply_threshold(scores: pd.Series, threshold: float) -> pd.Series:
    """Convert scores into binary predictions using scores >= threshold."""

    return pd.Series(
        scores >= threshold,
        index=scores.index,
        name="prediction",
    )


def binary_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float | int]:
    """Compute binary classification metrics and simple diagnostic counts."""

    _require_same_length(y_true, y_pred, left_name="y_true", right_name="y_pred")

    true = y_true.astype(bool)
    pred = y_pred.astype(bool)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
        "n": int(len(true)),
        "n_positive": int(true.sum()),
        "n_predicted_positive": int(pred.sum()),
    }


def find_best_threshold(
    y_true: pd.Series,
    scores: pd.Series,
    *,
    metric: SelectionMetric = "f1",
) -> dict[str, float | str]:
    """Select the score threshold with the best validation metric value.

    Ties are resolved by the first threshold in threshold order.
    """

    _require_same_length(y_true, scores, left_name="y_true", right_name="scores")
    if metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError(f"Unsupported metric: {metric!r}")
    if len(scores) == 0:
        raise ValueError("Cannot select threshold from empty scores")
    if scores.isna().any():
        raise ValueError("scores contains NaN values")

    thresholds, values = _threshold_metric_values(y_true, scores, metric=metric)
    best_position = int(np.argmax(values))

    return {
        "threshold": float(thresholds[best_position]),
        "metric": metric,
        "metric_value": float(values[best_position]),
    }


def _threshold_metric_values(
    y_true: pd.Series,
    scores: pd.Series,
    *,
    metric: SelectionMetric,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute validation metric values for all score thresholds in one pass."""

    score_values = scores.astype(float).to_numpy()
    unique_scores = np.sort(np.unique(score_values))

    max_score = unique_scores[-1]
    if np.isfinite(max_score):
        no_positive_threshold = np.nextafter(max_score, np.inf)
    else:
        no_positive_threshold = np.inf
    thresholds = np.concatenate([[no_positive_threshold], unique_scores])
    true = y_true.astype(bool).to_numpy()

    order = np.argsort(score_values, kind="mergesort")
    sorted_scores = score_values[order]
    sorted_true = true[order].astype(int)
    positives_from_position = np.concatenate(
        [np.cumsum(sorted_true[::-1])[::-1], np.array([0])]
    )

    starts = np.searchsorted(sorted_scores, thresholds, side="left")
    predicted_positive = len(sorted_scores) - starts
    true_positive = positives_from_position[starts]

    total_positive = int(true.sum())
    false_positive = predicted_positive - true_positive
    true_negative = len(true) - total_positive - false_positive

    if metric == "accuracy":
        values = (true_positive + true_negative) / len(true)
    elif metric == "precision":
        values = np.divide(
            true_positive,
            predicted_positive,
            out=np.zeros_like(true_positive, dtype=float),
            where=predicted_positive != 0,
        )
    elif metric == "recall":
        values = np.divide(
            true_positive,
            total_positive,
            out=np.zeros_like(true_positive, dtype=float),
            where=total_positive != 0,
        )
    else:
        precision = np.divide(
            true_positive,
            predicted_positive,
            out=np.zeros_like(true_positive, dtype=float),
            where=predicted_positive != 0,
        )
        recall = np.divide(
            true_positive,
            total_positive,
            out=np.zeros_like(true_positive, dtype=float),
            where=total_positive != 0,
        )
        values = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision, dtype=float),
            where=(precision + recall) != 0,
        )

    return thresholds, values


def _require_same_length(
    left: pd.Series,
    right: pd.Series,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if len(left) != len(right):
        raise ValueError(f"{left_name} and {right_name} must have the same length")
