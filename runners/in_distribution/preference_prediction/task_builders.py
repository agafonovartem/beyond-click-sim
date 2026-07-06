from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    CappedObservedPreferenceCandidateSampler,
    MinUserInteractionsFilter,
    PreferencePredictionTaskBuilder,
    RandomFractionSplitter,
    Task,
)


VERSION = "v1"
SEEDS = (0, 1, 2, 3, 4)
REDUCED_SEEDS = (0, 1, 2)
MIN_INTERACTIONS = 10
NEGATIVE_RATIOS = (1, 2, 3, 9)
DATASETS = ("ml-1m", "steam")
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
TOTAL_CANDIDATE_ITEMS = 10
EVAL_USERS = 1000
DEBUG_EVAL_USERS = 100
MAX_CANDIDATE_GROUPS_PER_USER = 5

DATASET_TARGET_SOURCE_COLUMNS = {
    "ml-1m": "target_like_ge4",
    "steam": "target_played_120",
}
DATASET_HISTORY_CONTEXT_COLUMNS = {
    "ml-1m": ("rating",),
    "steam": ("playtime_forever",),
}


def build_preference_task(
    dataset_name: str,
    negative_ratio: int,
    seed: int,
    *,
    max_eval_users: int | None = None,
    max_candidate_groups_per_user: int | None = None,
    task_name: str | None = None,
) -> Task:
    """Build one observed preference-prediction task."""

    dataset = load_canonical_dataset(dataset_name)
    target_source_column = DATASET_TARGET_SOURCE_COLUMNS[dataset_name]
    eval_suffix = "" if max_eval_users is None else f"_eval_users{max_eval_users}"
    candidate_group_suffix = (
        ""
        if max_candidate_groups_per_user is None
        else f"_cg{max_candidate_groups_per_user}"
    )
    if task_name is None:
        task_name = (
            f"{dataset_name}_preference_cap10"
            f"{eval_suffix}{candidate_group_suffix}_m{negative_ratio}_seed{seed}"
        )
    builder = PreferencePredictionTaskBuilder(
        name=task_name,
        target_source_column=target_source_column,
        dataset_filter=MinUserInteractionsFilter(min_interactions=MIN_INTERACTIONS),
        splitter=RandomFractionSplitter(
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=seed,
        ),
        sampler=CappedObservedPreferenceCandidateSampler(
            negative_ratio=negative_ratio,
            total_items=TOTAL_CANDIDATE_ITEMS,
            max_eval_users=max_eval_users,
            max_candidate_groups_per_user=max_candidate_groups_per_user,
            seed=seed,
            target_source_column=target_source_column,
        ),
        history_context_columns=DATASET_HISTORY_CONTEXT_COLUMNS[dataset_name],
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
        if path.name == "beyond-click-sim" and (path / "pyproject.toml").exists():
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
    negative_ratio: int,
    seed: int,
    *,
    max_eval_users: int | None = None,
    max_candidate_groups_per_user: int | None = None,
    task_name: str | None = None,
) -> Callable[[], Task]:
    def build() -> Task:
        return build_preference_task(
            dataset_name,
            negative_ratio,
            seed,
            max_eval_users=max_eval_users,
            max_candidate_groups_per_user=max_candidate_groups_per_user,
            task_name=task_name,
        )

    return build


EVAL1000_CG5_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    (
        f"{dataset_name}_preference_cap10_eval_users{EVAL_USERS}"
        f"_cg{MAX_CANDIDATE_GROUPS_PER_USER}_m{negative_ratio}_seed{seed}"
    ): _make_builder(
        dataset_name,
        negative_ratio,
        seed,
        max_eval_users=EVAL_USERS,
        max_candidate_groups_per_user=MAX_CANDIDATE_GROUPS_PER_USER,
        task_name=(
            f"{dataset_name}_preference_cap10_eval_users{EVAL_USERS}"
            f"_cg{MAX_CANDIDATE_GROUPS_PER_USER}_m{negative_ratio}_seed{seed}"
        ),
    )
    for dataset_name in DATASETS
    for negative_ratio in NEGATIVE_RATIOS
    for seed in REDUCED_SEEDS
}

EVAL100_CG5_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    (
        f"{dataset_name}_preference_cap10_eval_users{DEBUG_EVAL_USERS}"
        f"_cg{MAX_CANDIDATE_GROUPS_PER_USER}_m{negative_ratio}_seed{seed}"
    ): _make_builder(
        dataset_name,
        negative_ratio,
        seed,
        max_eval_users=DEBUG_EVAL_USERS,
        max_candidate_groups_per_user=MAX_CANDIDATE_GROUPS_PER_USER,
        task_name=(
            f"{dataset_name}_preference_cap10_eval_users{DEBUG_EVAL_USERS}"
            f"_cg{MAX_CANDIDATE_GROUPS_PER_USER}_m{negative_ratio}_seed{seed}"
        ),
    )
    for dataset_name in DATASETS
    for negative_ratio in NEGATIVE_RATIOS
    for seed in REDUCED_SEEDS
}

DEFAULT_TASK_NAMES = list(EVAL1000_CG5_TASK_BUILDERS)
TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    **EVAL1000_CG5_TASK_BUILDERS,
    **EVAL100_CG5_TASK_BUILDERS,
}
