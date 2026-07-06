from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.preference_prediction.task_builders import (
    DEFAULT_TASK_NAMES,
    EVAL100_CG5_TASK_BUILDERS,
    EVAL1000_CG5_TASK_BUILDERS,
    NEGATIVE_RATIOS,
    TASK_BUILDERS,
)


def test_preference_cg5_tasks_are_default_reduced_protocol() -> None:
    assert NEGATIVE_RATIOS == (1, 2, 3, 9)
    assert len(DEFAULT_TASK_NAMES) == 24
    assert set(DEFAULT_TASK_NAMES) == set(EVAL1000_CG5_TASK_BUILDERS)
    assert all("_preference_cap10_" in task_name for task_name in DEFAULT_TASK_NAMES)
    assert all("_cg5_" in task_name for task_name in DEFAULT_TASK_NAMES)
    assert not any("_m19_" in task_name for task_name in DEFAULT_TASK_NAMES)
    assert not any(
        "_seed3" in task_name or "_seed4" in task_name
        for task_name in DEFAULT_TASK_NAMES
    )


def test_preference_registry_keeps_debug_tasks_available() -> None:
    assert len(EVAL100_CG5_TASK_BUILDERS) == 24
    assert "ml-1m_preference_cap10_eval_users1000_cg5_m1_seed0" in TASK_BUILDERS
    assert "steam_preference_cap10_eval_users1000_cg5_m9_seed2" in TASK_BUILDERS
    assert "ml-1m_preference_cap10_eval_users100_cg5_m3_seed0" in TASK_BUILDERS
    assert "steam_preference_cap10_eval_users100_cg5_m9_seed2" in TASK_BUILDERS
    assert "ml-1m_preference_cap10_eval_users1000_cg5_m19_seed0" not in TASK_BUILDERS
    assert set(EVAL100_CG5_TASK_BUILDERS).isdisjoint(DEFAULT_TASK_NAMES)
