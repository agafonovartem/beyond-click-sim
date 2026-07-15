from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    MinUserInteractionsFilter,
    PostSplitUserSampler,
    RandomFractionSplitter,
    RegressionPredictionTaskBuilder,
    Task,
    TrainItemRatingStatistics,
)


VERSION = "v1"
SEEDS = (0, 1, 2, 3, 4)
MIN_INTERACTIONS = 10
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
EVAL_USERS = 1000
DEBUG_EVAL_USERS = 100
EVAL_ROWS_PER_USER = 5

DATASET_TARGETS = {
    "ml-1m": {
        "rating": {
            "target_source_column": "target_rating",
            "history_context_columns": ("rating",),
        },
    },
}
DATASET_ITEM_RATING_VALUE_COLUMNS = {
    "ml-1m": "rating",
}


def build_regression_task(
    dataset_name: str,
    target_name: str,
    seed: int,
    *,
    max_eval_users: int | None = None,
    max_rows_per_user: int | None = None,
    use_item_rating_stats: bool = False,
) -> Task:
    """Build one observed-only regression task."""

    dataset = load_canonical_dataset(dataset_name)
    target_config = DATASET_TARGETS[dataset_name][target_name]
    eval_suffix = "" if max_eval_users is None else f"_eval_users{max_eval_users}"
    row_suffix = (
        "" if max_rows_per_user is None else f"_rows_per_user{max_rows_per_user}"
    )
    item_stats_suffix = "_item_stats" if use_item_rating_stats else ""
    task_name = (
        f"{dataset_name}_{target_name}{item_stats_suffix}"
        f"{eval_suffix}{row_suffix}_seed{seed}"
    )
    builder = RegressionPredictionTaskBuilder(
        name=task_name,
        dataset_filter=MinUserInteractionsFilter(min_interactions=MIN_INTERACTIONS),
        splitter=RandomFractionSplitter(
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=seed,
        ),
        target_source_column=str(target_config["target_source_column"]),
        history_context_columns=target_config["history_context_columns"],
        eval_sampler=PostSplitUserSampler(
            n_users=max_eval_users,
            seed=seed,
            max_rows_per_user=max_rows_per_user,
        ),
        item_feature_builder=_item_feature_builder(
            dataset_name,
            use_item_rating_stats=use_item_rating_stats,
        ),
    )
    return builder.build(dataset)


def load_canonical_dataset(dataset_name: str) -> CanonicalDataset:
    root = data_root() / dataset_name / VERSION
    return CanonicalDataset(
        name=dataset_name,
        version=VERSION,
        root=root,
        users_path=root / "users.parquet",
        items_path=root / "items.parquet",
        interactions_path=root / "interactions.parquet",
        manifest_path=root / "manifest.json",
    )


def repo_root() -> Path:
    for path in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (path / "pyproject.toml").exists() and (
            path / "src" / "beyond_click_sim"
        ).is_dir():
            return path
    raise RuntimeError("Could not find beyond-click-sim repo root")


def data_root() -> Path:
    for path in [repo_root(), *repo_root().parents]:
        candidate = path / "data" / "canonical"
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not find data/canonical")


def _make_builder(
    dataset_name: str,
    target_name: str,
    seed: int,
    *,
    max_eval_users: int | None = None,
    max_rows_per_user: int | None = None,
    use_item_rating_stats: bool = False,
) -> Callable[[], Task]:
    def build() -> Task:
        return build_regression_task(
            dataset_name,
            target_name,
            seed,
            max_eval_users=max_eval_users,
            max_rows_per_user=max_rows_per_user,
            use_item_rating_stats=use_item_rating_stats,
        )

    return build


def _item_feature_builder(
    dataset_name: str,
    *,
    use_item_rating_stats: bool,
) -> TrainItemRatingStatistics | None:
    if not use_item_rating_stats:
        return None
    if dataset_name not in DATASET_ITEM_RATING_VALUE_COLUMNS:
        raise ValueError(
            f"No item rating statistics config for dataset: {dataset_name}"
        )
    return TrainItemRatingStatistics(
        value_column=DATASET_ITEM_RATING_VALUE_COLUMNS[dataset_name],
    )


EVAL1000_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    f"{dataset_name}_{target_name}_eval_users{EVAL_USERS}_seed{seed}": _make_builder(
        dataset_name,
        target_name,
        seed,
        max_eval_users=EVAL_USERS,
    )
    for dataset_name, targets in DATASET_TARGETS.items()
    for target_name in targets
    for seed in SEEDS
}

EVAL1000_ITEM_STATS_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    f"{dataset_name}_{target_name}_item_stats_eval_users{EVAL_USERS}_seed{seed}": _make_builder(
        dataset_name,
        target_name,
        seed,
        max_eval_users=EVAL_USERS,
        use_item_rating_stats=True,
    )
    for dataset_name, targets in DATASET_TARGETS.items()
    if dataset_name in DATASET_ITEM_RATING_VALUE_COLUMNS
    for target_name in targets
    for seed in SEEDS
}

EVAL1000_ROWS_PER_USER_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    (
        f"{dataset_name}_{target_name}_eval_users{EVAL_USERS}"
        f"_rows_per_user{EVAL_ROWS_PER_USER}_seed{seed}"
    ): _make_builder(
        dataset_name,
        target_name,
        seed,
        max_eval_users=EVAL_USERS,
        max_rows_per_user=EVAL_ROWS_PER_USER,
    )
    for dataset_name, targets in DATASET_TARGETS.items()
    for target_name in targets
    for seed in SEEDS
}

EVAL1000_ROWS_PER_USER_ITEM_STATS_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    (
        f"{dataset_name}_{target_name}_item_stats_eval_users{EVAL_USERS}"
        f"_rows_per_user{EVAL_ROWS_PER_USER}_seed{seed}"
    ): _make_builder(
        dataset_name,
        target_name,
        seed,
        max_eval_users=EVAL_USERS,
        max_rows_per_user=EVAL_ROWS_PER_USER,
        use_item_rating_stats=True,
    )
    for dataset_name, targets in DATASET_TARGETS.items()
    if dataset_name in DATASET_ITEM_RATING_VALUE_COLUMNS
    for target_name in targets
    for seed in SEEDS
}

EVAL100_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    f"{dataset_name}_{target_name}_eval_users{DEBUG_EVAL_USERS}_seed{seed}": _make_builder(
        dataset_name,
        target_name,
        seed,
        max_eval_users=DEBUG_EVAL_USERS,
    )
    for dataset_name, targets in DATASET_TARGETS.items()
    for target_name in targets
    for seed in SEEDS
}

EVAL100_ITEM_STATS_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    f"{dataset_name}_{target_name}_item_stats_eval_users{DEBUG_EVAL_USERS}_seed{seed}": _make_builder(
        dataset_name,
        target_name,
        seed,
        max_eval_users=DEBUG_EVAL_USERS,
        use_item_rating_stats=True,
    )
    for dataset_name, targets in DATASET_TARGETS.items()
    if dataset_name in DATASET_ITEM_RATING_VALUE_COLUMNS
    for target_name in targets
    for seed in SEEDS
}

DEFAULT_TASK_NAMES = list(EVAL1000_ROWS_PER_USER_TASK_BUILDERS)
TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    **EVAL1000_ROWS_PER_USER_TASK_BUILDERS,
    **EVAL1000_ROWS_PER_USER_ITEM_STATS_TASK_BUILDERS,
    **EVAL1000_TASK_BUILDERS,
    **EVAL1000_ITEM_STATS_TASK_BUILDERS,
    **EVAL100_TASK_BUILDERS,
    **EVAL100_ITEM_STATS_TASK_BUILDERS,
}
