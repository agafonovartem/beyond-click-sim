from __future__ import annotations

import re

from runners.in_distribution.cold_start.task_builders import (
    C_TARGETS,
    DEFAULT_TASK_NAMES,
    EVAL_USERS,
    K_VALUES,
    NEGATIVE_RATIOS,
    NEGATIVE_STRATEGIES,
    POOL_K,
    SAMPLING_SEEDS,
    SPLIT_SEED,
    TASK_BUILDERS,
    cold_start_task_name,
)


NAME_PATTERN = re.compile(
    r"^(?P<dataset>ml-1m(?:_item_stats)?)_cold_like_k(?P<k>\d+)"
    r"_c(?P<c>\d+)_eval(?P<eval>\d+)_m(?P<m>\d+)"
    r"_(?P<strategy>negu|negpop)_split(?P<split>\d+)_sample(?P<sample>\d+)$"
)


def test_registry_covers_full_grid() -> None:
    expected = (
        2  # plain + item_stats
        * len(K_VALUES)
        * len(NEGATIVE_RATIOS)
        * len(NEGATIVE_STRATEGIES)
        * len(SAMPLING_SEEDS)
    )
    assert len(TASK_BUILDERS) == expected
    assert len(TASK_BUILDERS) == 180


def test_task_names_parse_and_match_grid_constants() -> None:
    for name in TASK_BUILDERS:
        match = NAME_PATTERN.match(name)
        assert match is not None, f"unparseable task name: {name}"
        assert int(match["k"]) in K_VALUES
        assert int(match["c"]) == C_TARGETS
        assert int(match["eval"]) == EVAL_USERS
        assert int(match["m"]) in NEGATIVE_RATIOS
        assert int(match["split"]) == SPLIT_SEED
        assert int(match["sample"]) in SAMPLING_SEEDS


def test_task_name_roundtrip() -> None:
    name = cold_start_task_name("ml-1m", 3, 9, "popularity_matched", 2)
    assert name == "ml-1m_cold_like_k3_c5_eval1000_m9_negpop_split0_sample2"
    stats_name = cold_start_task_name(
        "ml-1m", 1, 1, "uniform", 0, use_item_rating_stats=True
    )
    assert stats_name == (
        "ml-1m_item_stats_cold_like_k1_c5_eval1000_m1_negu_split0_sample0"
    )


def test_default_tasks_are_small_smoke_subset() -> None:
    assert len(DEFAULT_TASK_NAMES) == 4
    assert set(DEFAULT_TASK_NAMES) <= set(TASK_BUILDERS)
    assert all("_k3_" in name for name in DEFAULT_TASK_NAMES)
    assert all(name.endswith("_sample0") for name in DEFAULT_TASK_NAMES)
    assert not any("item_stats" in name for name in DEFAULT_TASK_NAMES)


def test_pool_k_is_largest_k() -> None:
    # Eval-user eligibility is defined at the largest k so the same users are
    # eligible at every k in the grid.
    assert POOL_K == max(K_VALUES)
