from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import (
    CappedUserInteractionCandidateSampler,
    ColdStartTaskBuilder,
    ColdUserHoldoutSplitter,
    MinUserInteractionsFilter,
    TemporalColdStartCandidateSampler,
    TrainItemRatingStatistics,
)
from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.interaction_prediction.task_builders import (
    load_canonical_dataset,
    repo_root,
)


K_VALUES = (1, 3, 5)
DATASETS = ("ml-1m",)  # Steam has no timestamps; ColdUserHoldoutSplitter raises
SEEDS = (0, 1, 2)
REDUCED_SEEDS = (0, 1, 2)
MIN_INTERACTIONS = 10
NEGATIVE_RATIOS = (1, 2, 3, 9, 19)
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
TOTAL_CANDIDATE_ITEMS = 20
EVAL_USERS = 1000
MAX_GROUPS_PER_USER = 5

DATASET_HISTORY_CONTEXT_COLUMNS = {
    "ml-1m": ("rating",),
}
DATASET_ITEM_RATING_VALUE_COLUMNS = {
    "ml-1m": "rating",
}


def build_cold_start_task(
    dataset_name: str,
    k: int,
    negative_ratio: int,
    seed: int,
    *,
    max_eval_users: int | None = None,
    max_groups_per_user: int | None = None,
    group_offset: int = 0,
    use_item_rating_stats: bool = False,
    task_name: str | None = None,
) -> ColdStartTask:
    dataset = load_canonical_dataset(dataset_name)
    item_stats_infix = "_item_stats" if use_item_rating_stats else ""
    dataset_prefix = f"{dataset_name}{item_stats_infix}"
    if task_name is None:
        if max_groups_per_user is not None:
            task_name = (
                f"{dataset_prefix}_cold_start_k{k}_cap20"
                f"_eval_users{max_eval_users}_cg{max_groups_per_user}"
                f"_m{negative_ratio}_seed{seed}"
            )
        else:
            task_name = f"{dataset_prefix}_cold_start_k{k}_cap20_m{negative_ratio}_seed{seed}"
    if use_item_rating_stats:
        if dataset_name not in DATASET_ITEM_RATING_VALUE_COLUMNS:
            raise ValueError(f"Item rating statistics not supported for dataset: {dataset_name!r}")
        item_feature_builder: TrainItemRatingStatistics | None = TrainItemRatingStatistics(
            value_column=DATASET_ITEM_RATING_VALUE_COLUMNS[dataset_name]
        )
    else:
        item_feature_builder = None
    if max_groups_per_user is not None:
        sampler = TemporalColdStartCandidateSampler(
            negative_ratio=negative_ratio,
            total_items=TOTAL_CANDIDATE_ITEMS,
            max_eval_users=max_eval_users,
            max_candidate_groups_per_user=max_groups_per_user,
            group_offset=group_offset,
            seed=seed,
        )
    else:
        sampler = CappedUserInteractionCandidateSampler(
            negative_ratio=negative_ratio,
            total_items=TOTAL_CANDIDATE_ITEMS,
            seed=seed,
        )
    return ColdStartTaskBuilder(
        name=task_name,
        dataset_filter=MinUserInteractionsFilter(min_interactions=MIN_INTERACTIONS),
        splitter=ColdUserHoldoutSplitter(
            k=k,
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=seed,
        ),
        sampler=sampler,
        history_context_columns=DATASET_HISTORY_CONTEXT_COLUMNS[dataset_name],
        item_feature_builder=item_feature_builder,
    ).build(dataset)


def _make_builder(
    dataset_name: str,
    k: int,
    negative_ratio: int,
    seed: int,
    *,
    max_eval_users: int | None = None,
    max_groups_per_user: int | None = None,
    group_offset: int = 0,
    use_item_rating_stats: bool = False,
) -> Callable[[], ColdStartTask]:
    def build() -> ColdStartTask:
        return build_cold_start_task(
            dataset_name,
            k,
            negative_ratio,
            seed,
            max_eval_users=max_eval_users,
            max_groups_per_user=max_groups_per_user,
            group_offset=group_offset,
            use_item_rating_stats=use_item_rating_stats,
        )

    return build


EVAL1000_CG5_TASK_BUILDERS: dict[str, Callable[[], ColdStartTask]] = {
    f"{dataset}_cold_start_k{k}_cap20_eval_users{EVAL_USERS}_cg{MAX_GROUPS_PER_USER}_m{ratio}_seed{seed}": _make_builder(
        dataset,
        k,
        ratio,
        seed,
        max_eval_users=EVAL_USERS,
        max_groups_per_user=MAX_GROUPS_PER_USER,
        group_offset=seed * MAX_GROUPS_PER_USER,
    )
    for dataset in DATASETS
    for k in K_VALUES
    for ratio in NEGATIVE_RATIOS
    for seed in SEEDS
}

EVAL1000_CG5_ITEM_STATS_TASK_BUILDERS: dict[str, Callable[[], ColdStartTask]] = {
    f"ml-1m_item_stats_cold_start_k{k}_cap20_eval_users{EVAL_USERS}_cg{MAX_GROUPS_PER_USER}_m{ratio}_seed{seed}": _make_builder(
        "ml-1m",
        k,
        ratio,
        seed,
        max_eval_users=EVAL_USERS,
        max_groups_per_user=MAX_GROUPS_PER_USER,
        group_offset=seed * MAX_GROUPS_PER_USER,
        use_item_rating_stats=True,
    )
    for k in K_VALUES
    for ratio in NEGATIVE_RATIOS
    for seed in SEEDS
}

TASK_BUILDERS: dict[str, Callable[[], ColdStartTask]] = {
    f"{dataset}_cold_start_k{k}_cap20_m{ratio}_seed{seed}": _make_builder(
        dataset, k, ratio, seed
    )
    for dataset in DATASETS
    for k in K_VALUES
    for ratio in NEGATIVE_RATIOS
    for seed in SEEDS
}
TASK_BUILDERS.update(EVAL1000_CG5_TASK_BUILDERS)
TASK_BUILDERS.update(EVAL1000_CG5_ITEM_STATS_TASK_BUILDERS)

DEFAULT_TASK_NAMES: list[str] = [
    f"ml-1m_cold_start_k{k}_cap20_eval_users{EVAL_USERS}_cg{MAX_GROUPS_PER_USER}_m{ratio}_seed{seed}"
    for k in K_VALUES
    for seed in REDUCED_SEEDS
    for ratio in NEGATIVE_RATIOS
]
