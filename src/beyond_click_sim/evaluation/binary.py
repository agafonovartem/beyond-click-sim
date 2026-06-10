from __future__ import annotations

from collections.abc import Callable
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


def grouped_binary_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    groups: pd.Series,
) -> dict[str, float | int]:
    """Compute binary metrics per group, then average groups equally."""

    _require_same_length(y_true, y_pred, left_name="y_true", right_name="y_pred")
    _require_same_length(y_true, groups, left_name="y_true", right_name="groups")
    if len(y_true) == 0:
        raise ValueError("Cannot compute grouped metrics on empty inputs")

    true = y_true.astype(bool).to_numpy()
    pred = y_pred.astype(bool).to_numpy()
    group_codes, _ = pd.factorize(groups, sort=False)
    n_groups = int(group_codes.max()) + 1

    group_sizes = np.bincount(group_codes, minlength=n_groups)
    total_positive = np.bincount(
        group_codes,
        weights=true.astype(int),
        minlength=n_groups,
    )
    predicted_positive = np.bincount(
        group_codes,
        weights=pred.astype(int),
        minlength=n_groups,
    )
    true_positive = np.bincount(
        group_codes,
        weights=(true & pred).astype(int),
        minlength=n_groups,
    )

    return {
        "accuracy": float(
            _per_group_metric_values(
                group_sizes=group_sizes,
                total_positive=total_positive,
                predicted_positive=predicted_positive,
                true_positive=true_positive,
                metric="accuracy",
            ).mean()
        ),
        "precision": float(
            _per_group_metric_values(
                group_sizes=group_sizes,
                total_positive=total_positive,
                predicted_positive=predicted_positive,
                true_positive=true_positive,
                metric="precision",
            ).mean()
        ),
        "recall": float(
            _per_group_metric_values(
                group_sizes=group_sizes,
                total_positive=total_positive,
                predicted_positive=predicted_positive,
                true_positive=true_positive,
                metric="recall",
            ).mean()
        ),
        "f1": float(
            _per_group_metric_values(
                group_sizes=group_sizes,
                total_positive=total_positive,
                predicted_positive=predicted_positive,
                true_positive=true_positive,
                metric="f1",
            ).mean()
        ),
        "n_groups": n_groups,
        "n": int(len(true)),
        "n_positive": int(true.sum()),
        "n_predicted_positive": int(pred.sum()),
    }


def user_grouped_binary_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    groups: pd.Series,
    users: pd.Series,
) -> dict[str, float | int]:
    """Compute per-group metrics, average groups per user, then average users."""

    _require_same_length(y_true, y_pred, left_name="y_true", right_name="y_pred")
    _require_same_length(y_true, groups, left_name="y_true", right_name="groups")
    _require_same_length(y_true, users, left_name="y_true", right_name="users")
    if len(y_true) == 0:
        raise ValueError("Cannot compute user-grouped metrics on empty inputs")

    true = y_true.astype(bool).to_numpy()
    pred = y_pred.astype(bool).to_numpy()
    group_codes, n_groups, _, n_users, group_user_codes = _user_group_codes(
        groups=groups,
        users=users,
    )

    group_sizes = np.bincount(group_codes, minlength=n_groups)
    total_positive = np.bincount(
        group_codes,
        weights=true.astype(int),
        minlength=n_groups,
    )
    predicted_positive = np.bincount(
        group_codes,
        weights=pred.astype(int),
        minlength=n_groups,
    )
    true_positive = np.bincount(
        group_codes,
        weights=(true & pred).astype(int),
        minlength=n_groups,
    )
    groups_per_user = np.bincount(group_user_codes, minlength=n_users)

    return {
        "accuracy": _macro_user_group_metric_value(
            group_sizes=group_sizes,
            total_positive=total_positive,
            predicted_positive=predicted_positive,
            true_positive=true_positive,
            group_user_codes=group_user_codes,
            groups_per_user=groups_per_user,
            metric="accuracy",
        ),
        "precision": _macro_user_group_metric_value(
            group_sizes=group_sizes,
            total_positive=total_positive,
            predicted_positive=predicted_positive,
            true_positive=true_positive,
            group_user_codes=group_user_codes,
            groups_per_user=groups_per_user,
            metric="precision",
        ),
        "recall": _macro_user_group_metric_value(
            group_sizes=group_sizes,
            total_positive=total_positive,
            predicted_positive=predicted_positive,
            true_positive=true_positive,
            group_user_codes=group_user_codes,
            groups_per_user=groups_per_user,
            metric="recall",
        ),
        "f1": _macro_user_group_metric_value(
            group_sizes=group_sizes,
            total_positive=total_positive,
            predicted_positive=predicted_positive,
            true_positive=true_positive,
            group_user_codes=group_user_codes,
            groups_per_user=groups_per_user,
            metric="f1",
        ),
        "n_users": int(n_users),
        "n_groups": int(n_groups),
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


def find_best_threshold_by_metric(
    y_true: pd.Series,
    scores: pd.Series,
    *,
    metric_fn: Callable[[pd.Series, pd.Series], float],
    metric_name: str,
) -> dict[str, float | str]:
    """Select threshold using a caller-provided validation metric."""

    _require_same_length(y_true, scores, left_name="y_true", right_name="scores")
    if len(scores) == 0:
        raise ValueError("Cannot select threshold from empty scores")
    if scores.isna().any():
        raise ValueError("scores contains NaN values")

    thresholds = _threshold_candidates(scores)
    best_threshold = float(thresholds[0])
    best_value = -np.inf
    for threshold in thresholds:
        predictions = apply_threshold(scores, float(threshold))
        value = float(metric_fn(y_true, predictions))
        if value > best_value:
            best_threshold = float(threshold)
            best_value = value

    return {
        "threshold": best_threshold,
        "metric": metric_name,
        "metric_value": float(best_value),
    }


def find_best_group_threshold(
    y_true: pd.Series,
    scores: pd.Series,
    groups: pd.Series,
    *,
    metric: SelectionMetric = "f1",
) -> dict[str, float | str]:
    """Select threshold by macro-averaged per-group binary metric."""

    _require_same_length(y_true, scores, left_name="y_true", right_name="scores")
    _require_same_length(y_true, groups, left_name="y_true", right_name="groups")
    if metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError(f"Unsupported metric: {metric!r}")
    if len(scores) == 0:
        raise ValueError("Cannot select threshold from empty scores")
    if scores.isna().any():
        raise ValueError("scores contains NaN values")

    thresholds, values = _group_threshold_metric_values(
        y_true,
        scores,
        groups,
        metric=metric,
    )
    best_position = int(np.argmax(values))
    return {
        "threshold": float(thresholds[best_position]),
        "metric": f"macro_group_{metric}",
        "metric_value": float(values[best_position]),
    }


def find_best_user_group_threshold(
    y_true: pd.Series,
    scores: pd.Series,
    groups: pd.Series,
    users: pd.Series,
    *,
    metric: SelectionMetric = "f1",
) -> dict[str, float | str]:
    """Select threshold by per-group metric, averaged within user then across users."""

    _require_same_length(y_true, scores, left_name="y_true", right_name="scores")
    _require_same_length(y_true, groups, left_name="y_true", right_name="groups")
    _require_same_length(y_true, users, left_name="y_true", right_name="users")
    if metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError(f"Unsupported metric: {metric!r}")
    if len(scores) == 0:
        raise ValueError("Cannot select threshold from empty scores")
    if scores.isna().any():
        raise ValueError("scores contains NaN values")

    thresholds, values = _user_group_threshold_metric_values(
        y_true,
        scores,
        groups,
        users,
        metric=metric,
    )
    best_position = int(np.argmax(values))
    return {
        "threshold": float(thresholds[best_position]),
        "metric": f"macro_by_user_group_mean_{metric}",
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
    thresholds = _threshold_candidates(scores)
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


def _group_threshold_metric_values(
    y_true: pd.Series,
    scores: pd.Series,
    groups: pd.Series,
    *,
    metric: SelectionMetric,
) -> tuple[np.ndarray, np.ndarray]:
    true = y_true.astype(bool).to_numpy()
    score_values = scores.astype(float).to_numpy()
    group_codes, _ = pd.factorize(groups, sort=False)
    n_groups = int(group_codes.max()) + 1

    group_sizes = np.bincount(group_codes, minlength=n_groups)
    total_positive = np.bincount(
        group_codes,
        weights=true.astype(int),
        minlength=n_groups,
    )
    predicted_positive = np.zeros(n_groups, dtype=float)
    true_positive = np.zeros(n_groups, dtype=float)
    per_group_values = _per_group_metric_values(
        group_sizes=group_sizes,
        total_positive=total_positive,
        predicted_positive=predicted_positive,
        true_positive=true_positive,
        metric=metric,
    )
    metric_sum = float(per_group_values.sum())

    order = np.argsort(score_values, kind="mergesort")[::-1]
    sorted_scores = score_values[order]
    bucket_starts = np.flatnonzero(
        np.concatenate([[True], sorted_scores[1:] != sorted_scores[:-1]])
    )
    bucket_ends = np.concatenate([bucket_starts[1:], [len(sorted_scores)]])
    unique_scores_desc = sorted_scores[bucket_starts]
    max_score = unique_scores_desc[0]
    if np.isfinite(max_score):
        no_positive_threshold = np.nextafter(max_score, np.inf)
    else:
        no_positive_threshold = np.inf

    thresholds_desc = np.concatenate([[no_positive_threshold], unique_scores_desc])
    values_desc = np.empty(len(thresholds_desc), dtype=float)
    values_desc[0] = metric_sum / n_groups

    for position, (start, end) in enumerate(
        zip(bucket_starts, bucket_ends, strict=True),
        start=1,
    ):
        bucket_order = order[start:end]
        bucket_groups = group_codes[bucket_order]
        changed_groups, inverse = np.unique(bucket_groups, return_inverse=True)
        predicted_increment = np.bincount(inverse)
        true_positive_increment = np.bincount(
            inverse,
            weights=true[bucket_order].astype(int),
        )

        old_values = per_group_values[changed_groups].copy()
        predicted_positive[changed_groups] += predicted_increment
        true_positive[changed_groups] += true_positive_increment
        new_values = _per_group_metric_values(
            group_sizes=group_sizes[changed_groups],
            total_positive=total_positive[changed_groups],
            predicted_positive=predicted_positive[changed_groups],
            true_positive=true_positive[changed_groups],
            metric=metric,
        )
        per_group_values[changed_groups] = new_values
        metric_sum += float(new_values.sum() - old_values.sum())
        values_desc[position] = metric_sum / n_groups

    # Preserve the same tie order as find_best_threshold:
    # no-positive threshold first, then score thresholds ascending.
    return (
        np.concatenate([[thresholds_desc[0]], thresholds_desc[:0:-1]]),
        np.concatenate([[values_desc[0]], values_desc[:0:-1]]),
    )


def _user_group_threshold_metric_values(
    y_true: pd.Series,
    scores: pd.Series,
    groups: pd.Series,
    users: pd.Series,
    *,
    metric: SelectionMetric,
) -> tuple[np.ndarray, np.ndarray]:
    true = y_true.astype(bool).to_numpy()
    score_values = scores.astype(float).to_numpy()
    group_codes, n_groups, _, n_users, group_user_codes = _user_group_codes(
        groups=groups,
        users=users,
    )

    group_sizes = np.bincount(group_codes, minlength=n_groups)
    total_positive = np.bincount(
        group_codes,
        weights=true.astype(int),
        minlength=n_groups,
    )
    predicted_positive = np.zeros(n_groups, dtype=float)
    true_positive = np.zeros(n_groups, dtype=float)
    per_group_values = _per_group_metric_values(
        group_sizes=group_sizes,
        total_positive=total_positive,
        predicted_positive=predicted_positive,
        true_positive=true_positive,
        metric=metric,
    )
    groups_per_user = np.bincount(group_user_codes, minlength=n_users)
    user_metric_sums = np.bincount(
        group_user_codes,
        weights=per_group_values,
        minlength=n_users,
    )

    order = np.argsort(score_values, kind="mergesort")[::-1]
    sorted_scores = score_values[order]
    bucket_starts = np.flatnonzero(
        np.concatenate([[True], sorted_scores[1:] != sorted_scores[:-1]])
    )
    bucket_ends = np.concatenate([bucket_starts[1:], [len(sorted_scores)]])
    unique_scores_desc = sorted_scores[bucket_starts]
    max_score = unique_scores_desc[0]
    if np.isfinite(max_score):
        no_positive_threshold = np.nextafter(max_score, np.inf)
    else:
        no_positive_threshold = np.inf

    thresholds_desc = np.concatenate([[no_positive_threshold], unique_scores_desc])
    values_desc = np.empty(len(thresholds_desc), dtype=float)
    values_desc[0] = _mean_user_metric(
        user_metric_sums=user_metric_sums,
        groups_per_user=groups_per_user,
    )

    for position, (start, end) in enumerate(
        zip(bucket_starts, bucket_ends, strict=True),
        start=1,
    ):
        bucket_order = order[start:end]
        bucket_groups = group_codes[bucket_order]
        changed_groups, inverse = np.unique(bucket_groups, return_inverse=True)
        predicted_increment = np.bincount(inverse)
        true_positive_increment = np.bincount(
            inverse,
            weights=true[bucket_order].astype(int),
        )

        old_values = per_group_values[changed_groups].copy()
        predicted_positive[changed_groups] += predicted_increment
        true_positive[changed_groups] += true_positive_increment
        new_values = _per_group_metric_values(
            group_sizes=group_sizes[changed_groups],
            total_positive=total_positive[changed_groups],
            predicted_positive=predicted_positive[changed_groups],
            true_positive=true_positive[changed_groups],
            metric=metric,
        )
        per_group_values[changed_groups] = new_values

        changed_users = group_user_codes[changed_groups]
        user_metric_sums += np.bincount(
            changed_users,
            weights=new_values - old_values,
            minlength=n_users,
        )
        values_desc[position] = _mean_user_metric(
            user_metric_sums=user_metric_sums,
            groups_per_user=groups_per_user,
        )

    # Preserve the same tie order as find_best_threshold:
    # no-positive threshold first, then score thresholds ascending.
    return (
        np.concatenate([[thresholds_desc[0]], thresholds_desc[:0:-1]]),
        np.concatenate([[values_desc[0]], values_desc[:0:-1]]),
    )


def _macro_group_metric_value(
    *,
    group_sizes: np.ndarray,
    total_positive: np.ndarray,
    predicted_positive: np.ndarray,
    true_positive: np.ndarray,
    metric: SelectionMetric,
) -> float:
    return float(
        _per_group_metric_values(
            group_sizes=group_sizes,
            total_positive=total_positive,
            predicted_positive=predicted_positive,
            true_positive=true_positive,
            metric=metric,
        ).mean()
    )


def _macro_user_group_metric_value(
    *,
    group_sizes: np.ndarray,
    total_positive: np.ndarray,
    predicted_positive: np.ndarray,
    true_positive: np.ndarray,
    group_user_codes: np.ndarray,
    groups_per_user: np.ndarray,
    metric: SelectionMetric,
) -> float:
    per_group_values = _per_group_metric_values(
        group_sizes=group_sizes,
        total_positive=total_positive,
        predicted_positive=predicted_positive,
        true_positive=true_positive,
        metric=metric,
    )
    user_metric_sums = np.bincount(
        group_user_codes,
        weights=per_group_values,
        minlength=len(groups_per_user),
    )
    return _mean_user_metric(
        user_metric_sums=user_metric_sums,
        groups_per_user=groups_per_user,
    )


def _mean_user_metric(
    *,
    user_metric_sums: np.ndarray,
    groups_per_user: np.ndarray,
) -> float:
    user_metric_values = user_metric_sums / groups_per_user
    return float(user_metric_values.mean())


def _per_group_metric_values(
    *,
    group_sizes: np.ndarray,
    total_positive: np.ndarray,
    predicted_positive: np.ndarray,
    true_positive: np.ndarray,
    metric: SelectionMetric,
) -> np.ndarray:
    false_positive = predicted_positive - true_positive
    true_negative = group_sizes - total_positive - false_positive

    if metric == "accuracy":
        values = (true_positive + true_negative) / group_sizes
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

    return values


def _user_group_codes(
    *,
    groups: pd.Series,
    users: pd.Series,
) -> tuple[np.ndarray, int, np.ndarray, int, np.ndarray]:
    group_keys = pd.MultiIndex.from_arrays([users, groups])
    group_codes, _ = pd.factorize(group_keys, sort=False)
    user_codes, _ = pd.factorize(users, sort=False)
    if (group_codes < 0).any() or (user_codes < 0).any():
        raise ValueError("groups and users must not contain NA values")

    n_groups = int(group_codes.max()) + 1
    n_users = int(user_codes.max()) + 1
    group_user_codes = np.empty(n_groups, dtype=int)
    group_user_codes[group_codes] = user_codes
    return group_codes, n_groups, user_codes, n_users, group_user_codes


def _threshold_candidates(scores: pd.Series) -> np.ndarray:
    score_values = scores.astype(float).to_numpy()
    unique_scores = np.sort(np.unique(score_values))
    max_score = unique_scores[-1]
    if np.isfinite(max_score):
        no_positive_threshold = np.nextafter(max_score, np.inf)
    else:
        no_positive_threshold = np.inf
    return np.concatenate([[no_positive_threshold], unique_scores])


def _require_same_length(
    left: pd.Series,
    right: pd.Series,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if len(left) != len(right):
        raise ValueError(f"{left_name} and {right_name} must have the same length")
