from __future__ import annotations

from collections.abc import Callable

from beyond_click_sim.tasks import (
    ColdStartTaskBuilder,
    ColdUserHoldoutSplitter,
    MinUserInteractionsFilter,
    PerTargetCandidateSampler,
    TrainItemRatingStatistics,
)
from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.interaction_prediction.task_builders import (
    load_canonical_dataset,
)


K_VALUES = (1, 3, 5)
DATASETS = ("ml-1m",)  # Steam has no timestamps; ColdUserHoldoutSplitter raises
SAMPLING_SEEDS = (0, 1, 2)
MIN_INTERACTIONS = 10
NEGATIVE_RATIOS = (1, 2, 3, 9, 19)
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
C_TARGETS = 5
EVAL_USERS = 1000
POOL_K = 5
SPLIT_SEED = 0
TARGET_SOURCE_COLUMN = "target_like_ge4"

NEGATIVE_STRATEGY_UNIFORM = "uniform"
NEGATIVE_STRATEGY_POPULARITY_MATCHED = "popularity_matched"
NEGATIVE_STRATEGIES = (
    NEGATIVE_STRATEGY_UNIFORM,
    NEGATIVE_STRATEGY_POPULARITY_MATCHED,
)
NEGATIVE_STRATEGY_NAME_SEGMENTS = {
    NEGATIVE_STRATEGY_UNIFORM: "negu",
    NEGATIVE_STRATEGY_POPULARITY_MATCHED: "negpop",
}

DATASET_HISTORY_CONTEXT_COLUMNS = {
    "ml-1m": ("rating",),
}
DATASET_ITEM_RATING_VALUE_COLUMNS = {
    "ml-1m": "rating",
}


def cold_start_task_name(
    dataset_name: str,
    k: int,
    negative_ratio: int,
    negative_strategy: str,
    sampling_seed: int,
    *,
    use_item_rating_stats: bool = False,
) -> str:
    dataset_prefix = (
        f"{dataset_name}_item_stats" if use_item_rating_stats else dataset_name
    )
    return (
        f"{dataset_prefix}_cold_like_k{k}_c{C_TARGETS}_eval{EVAL_USERS}"
        f"_m{negative_ratio}"
        f"_{NEGATIVE_STRATEGY_NAME_SEGMENTS[negative_strategy]}"
        f"_split{SPLIT_SEED}_sample{sampling_seed}"
    )


def build_cold_start_task(
    dataset_name: str,
    k: int,
    negative_ratio: int,
    negative_strategy: str,
    sampling_seed: int,
    *,
    use_item_rating_stats: bool = False,
    task_name: str | None = None,
) -> ColdStartTask:
    dataset = load_canonical_dataset(dataset_name)
    if task_name is None:
        task_name = cold_start_task_name(
            dataset_name,
            k,
            negative_ratio,
            negative_strategy,
            sampling_seed,
            use_item_rating_stats=use_item_rating_stats,
        )
    if use_item_rating_stats:
        if dataset_name not in DATASET_ITEM_RATING_VALUE_COLUMNS:
            raise ValueError(f"Item rating statistics not supported for dataset: {dataset_name!r}")
        item_feature_builder: TrainItemRatingStatistics | None = TrainItemRatingStatistics(
            value_column=DATASET_ITEM_RATING_VALUE_COLUMNS[dataset_name]
        )
    else:
        item_feature_builder = None
    return ColdStartTaskBuilder(
        name=task_name,
        dataset_filter=MinUserInteractionsFilter(min_interactions=MIN_INTERACTIONS),
        splitter=ColdUserHoldoutSplitter(
            k=k,
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=SPLIT_SEED,
        ),
        sampler=PerTargetCandidateSampler(
            negative_ratio=negative_ratio,
            strategy=negative_strategy,
            seed=sampling_seed,
        ),
        targets_per_user=C_TARGETS,
        target_source_column=TARGET_SOURCE_COLUMN,
        max_eval_users=EVAL_USERS,
        pool_k=POOL_K,
        split_seed=SPLIT_SEED,
        history_context_columns=DATASET_HISTORY_CONTEXT_COLUMNS[dataset_name],
        item_feature_builder=item_feature_builder,
    ).build(dataset)


def _make_builder(
    dataset_name: str,
    k: int,
    negative_ratio: int,
    negative_strategy: str,
    sampling_seed: int,
    *,
    use_item_rating_stats: bool = False,
) -> Callable[[], ColdStartTask]:
    def build() -> ColdStartTask:
        return build_cold_start_task(
            dataset_name,
            k,
            negative_ratio,
            negative_strategy,
            sampling_seed,
            use_item_rating_stats=use_item_rating_stats,
        )

    return build


TASK_BUILDERS: dict[str, Callable[[], ColdStartTask]] = {
    cold_start_task_name(
        dataset,
        k,
        ratio,
        strategy,
        sampling_seed,
        use_item_rating_stats=use_item_stats,
    ): _make_builder(
        dataset,
        k,
        ratio,
        strategy,
        sampling_seed,
        use_item_rating_stats=use_item_stats,
    )
    for dataset in DATASETS
    for use_item_stats in (False, True)
    if not use_item_stats or dataset in DATASET_ITEM_RATING_VALUE_COLUMNS
    for k in K_VALUES
    for ratio in NEGATIVE_RATIOS
    for strategy in NEGATIVE_STRATEGIES
    for sampling_seed in SAMPLING_SEEDS
}

DEFAULT_TASK_NAMES: list[str] = [
    cold_start_task_name("ml-1m", 3, ratio, strategy, 0)
    for ratio in (1, 19)
    for strategy in NEGATIVE_STRATEGIES
]
