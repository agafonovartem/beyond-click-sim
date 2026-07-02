from __future__ import annotations

from runners.in_distribution.interaction_prediction.task_builders import (
    EVAL100_CG5_ITEM_STATS_TASK_BUILDERS,
    EVAL100_CG5_TASK_BUILDERS,
    DEFAULT_TASK_NAMES,
    EVAL1000_CG5_ITEM_STATS_TASK_BUILDERS,
    EVAL1000_CG5_TASK_BUILDERS,
    EVAL1000_TASK_BUILDERS,
    TASK_BUILDERS,
)


def test_cg5_tasks_are_default_reduced_protocol() -> None:
    assert len(DEFAULT_TASK_NAMES) == 30
    assert set(DEFAULT_TASK_NAMES) == set(EVAL1000_CG5_TASK_BUILDERS)
    assert all("_cg5_" in task_name for task_name in DEFAULT_TASK_NAMES)
    assert not any(
        "_seed3" in task_name or "_seed4" in task_name
        for task_name in DEFAULT_TASK_NAMES
    )


def test_cg5_registry_keeps_old_eval1000_tasks_available() -> None:
    assert "ml-1m_cap20_eval_users1000_cg5_m19_seed0" in TASK_BUILDERS
    assert "steam_cap20_eval_users1000_cg5_m19_seed2" in TASK_BUILDERS
    assert "ml-1m_item_stats_cap20_eval_users1000_cg5_m19_seed0" in TASK_BUILDERS
    assert "ml-1m_cap20_eval_users1000_m19_seed4" in EVAL1000_TASK_BUILDERS
    assert "ml-1m_cap20_eval_users1000_m19_seed4" in TASK_BUILDERS


def test_cg5_item_stats_are_ml1m_only_and_not_defaults() -> None:
    assert len(EVAL1000_CG5_ITEM_STATS_TASK_BUILDERS) == 15
    assert all(
        task_name.startswith("ml-1m_item_stats_")
        for task_name in EVAL1000_CG5_ITEM_STATS_TASK_BUILDERS
    )
    assert set(EVAL1000_CG5_ITEM_STATS_TASK_BUILDERS).isdisjoint(DEFAULT_TASK_NAMES)


def test_eval100_cg5_tasks_are_debug_variants_not_defaults() -> None:
    assert len(EVAL100_CG5_TASK_BUILDERS) == 30
    assert len(EVAL100_CG5_ITEM_STATS_TASK_BUILDERS) == 15
    assert "ml-1m_cap20_eval_users100_cg5_m1_seed0" in TASK_BUILDERS
    assert "steam_cap20_eval_users100_cg5_m19_seed2" in TASK_BUILDERS
    assert "ml-1m_item_stats_cap20_eval_users100_cg5_m1_seed0" in TASK_BUILDERS
    assert set(EVAL100_CG5_TASK_BUILDERS).isdisjoint(DEFAULT_TASK_NAMES)
    assert set(EVAL100_CG5_ITEM_STATS_TASK_BUILDERS).isdisjoint(DEFAULT_TASK_NAMES)
