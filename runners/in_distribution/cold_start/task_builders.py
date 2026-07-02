from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import (
    CappedUserInteractionCandidateSampler,
    ColdStartTaskBuilder,
    ColdUserHoldoutSplitter,
    MinUserInteractionsFilter,
)
from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.interaction_prediction.task_builders import (
    load_canonical_dataset,
    repo_root,
)


K_VALUES = (1, 3, 5)
DATASETS = ("ml-1m",)  # Steam has no timestamps; ColdUserHoldoutSplitter raises
SEEDS = (0, 1, 2, 3, 4)
REDUCED_SEEDS = (0, 1, 2)
MIN_INTERACTIONS = 10
NEGATIVE_RATIOS = (1, 2, 3, 9, 19)
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
TOTAL_CANDIDATE_ITEMS = 20

DATASET_HISTORY_CONTEXT_COLUMNS = {
    "ml-1m": ("rating",),
}


def build_cold_start_task(
    dataset_name: str,
    k: int,
    negative_ratio: int,
    seed: int,
    *,
    task_name: str | None = None,
) -> ColdStartTask:
    dataset = load_canonical_dataset(dataset_name)
    if task_name is None:
        task_name = f"{dataset_name}_cold_start_k{k}_cap20_m{negative_ratio}_seed{seed}"
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
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=negative_ratio,
            total_items=TOTAL_CANDIDATE_ITEMS,
            seed=seed,
        ),
        history_context_columns=DATASET_HISTORY_CONTEXT_COLUMNS[dataset_name],
    ).build(dataset)


def _make_builder(
    dataset_name: str,
    k: int,
    negative_ratio: int,
    seed: int,
) -> Callable[[], ColdStartTask]:
    def build() -> ColdStartTask:
        return build_cold_start_task(dataset_name, k, negative_ratio, seed)

    return build


TASK_BUILDERS: dict[str, Callable[[], ColdStartTask]] = {
    f"{dataset}_cold_start_k{k}_cap20_m{ratio}_seed{seed}": _make_builder(
        dataset, k, ratio, seed
    )
    for dataset in DATASETS
    for k in K_VALUES
    for ratio in NEGATIVE_RATIOS
    for seed in SEEDS
}

DEFAULT_TASK_NAMES: list[str] = [
    f"ml-1m_cold_start_k{k}_cap20_m1_seed{seed}"
    for k in K_VALUES
    for seed in REDUCED_SEEDS
]
