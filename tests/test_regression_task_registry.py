from __future__ import annotations

from runners.in_distribution.regression_prediction.task_builders import (
    DEFAULT_TASK_NAMES,
    EVAL1000_ITEM_STATS_TASK_BUILDERS,
    EVAL1000_ROWS_PER_USER_ITEM_STATS_TASK_BUILDERS,
    EVAL1000_ROWS_PER_USER_TASK_BUILDERS,
    EVAL1000_TASK_BUILDERS,
    EVAL100_ITEM_STATS_TASK_BUILDERS,
    EVAL100_TASK_BUILDERS,
    TASK_BUILDERS,
)


def test_eval100_regression_tasks_are_debug_variants_not_defaults() -> None:
    assert len(EVAL100_TASK_BUILDERS) == 5
    assert len(EVAL100_ITEM_STATS_TASK_BUILDERS) == 5
    assert "ml-1m_rating_eval_users100_seed0" in TASK_BUILDERS
    assert "ml-1m_rating_item_stats_eval_users100_seed0" in TASK_BUILDERS
    assert set(EVAL100_TASK_BUILDERS).isdisjoint(DEFAULT_TASK_NAMES)
    assert set(EVAL100_ITEM_STATS_TASK_BUILDERS).isdisjoint(DEFAULT_TASK_NAMES)


def test_regression_defaults_use_eval1000_rows_per_user5_without_item_stats() -> None:
    assert list(DEFAULT_TASK_NAMES) == list(EVAL1000_ROWS_PER_USER_TASK_BUILDERS)
    assert "ml-1m_rating_eval_users1000_rows_per_user5_seed0" in DEFAULT_TASK_NAMES
    assert "ml-1m_rating_eval_users1000_seed0" not in DEFAULT_TASK_NAMES
    assert "ml-1m_rating_item_stats_eval_users1000_seed0" not in DEFAULT_TASK_NAMES
    assert "ml-1m_rating_eval_users1000_seed0" in TASK_BUILDERS
    assert (
        "ml-1m_rating_item_stats_eval_users1000_rows_per_user5_seed0"
        in TASK_BUILDERS
    )
    assert set(EVAL1000_TASK_BUILDERS).issubset(TASK_BUILDERS)
    assert set(EVAL1000_ITEM_STATS_TASK_BUILDERS).issubset(TASK_BUILDERS)
    assert set(EVAL1000_ROWS_PER_USER_ITEM_STATS_TASK_BUILDERS).issubset(
        TASK_BUILDERS
    )
