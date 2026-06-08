"""Metric functions and decision-rule utilities."""

from beyond_click_sim.evaluation.binary import (
    apply_threshold,
    binary_classification_metrics,
    find_best_group_threshold,
    find_best_threshold,
    find_best_threshold_by_metric,
    grouped_binary_classification_metrics,
)

__all__ = [
    "apply_threshold",
    "binary_classification_metrics",
    "find_best_group_threshold",
    "find_best_threshold",
    "find_best_threshold_by_metric",
    "grouped_binary_classification_metrics",
]
