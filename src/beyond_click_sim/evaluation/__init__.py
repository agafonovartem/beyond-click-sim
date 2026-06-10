"""Metric functions and decision-rule utilities."""

from beyond_click_sim.evaluation.binary import (
    apply_threshold,
    binary_classification_metrics,
    find_best_group_threshold,
    find_best_threshold,
    find_best_threshold_by_metric,
    find_best_user_group_threshold,
    grouped_binary_classification_metrics,
    user_grouped_binary_classification_metrics,
)

__all__ = [
    "apply_threshold",
    "binary_classification_metrics",
    "find_best_group_threshold",
    "find_best_threshold",
    "find_best_threshold_by_metric",
    "find_best_user_group_threshold",
    "grouped_binary_classification_metrics",
    "user_grouped_binary_classification_metrics",
]
