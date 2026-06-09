from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    AlignmentInteractionTaskBuilder,
    CappedUserInteractionCandidateSampler,
    MinUserInteractionsFilter,
    RandomFractionSplitter,
    Task,
)


VERSION = "v1"
SEEDS = (0, 1, 2, 3, 4)
MIN_INTERACTIONS = 10
NEGATIVE_RATIOS = (1, 2, 3, 9, 19)
DATASETS = ("ml-1m", "steam")
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
TOTAL_CANDIDATE_ITEMS = 20 # Cap, not guaranteed size: for m=1 and 5 held-out positives, group size is 10.
EVAL_USERS = 1000

DATASET_HISTORY_CONTEXT_COLUMNS = {
    "ml-1m": ("rating",),
    "steam": ("playtime_forever",),
}


def build_alignment_task(
    dataset_name: str,
    negative_ratio: int,
    seed: int,
    *,
    max_eval_users: int | None = None,
) -> Task:
    """Build one interaction-prediction task for a dataset, ratio, and seed."""

    dataset = load_canonical_dataset(dataset_name)
    eval_suffix = "" if max_eval_users is None else f"_eval_users{max_eval_users}"
    builder = AlignmentInteractionTaskBuilder(
        name=f"{dataset_name}_interaction_cap20{eval_suffix}_m{negative_ratio}_seed{seed}",
        dataset_filter=MinUserInteractionsFilter(min_interactions=MIN_INTERACTIONS),
        splitter=RandomFractionSplitter(
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=seed,
        ),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=negative_ratio,
            total_items=TOTAL_CANDIDATE_ITEMS,
            max_eval_users=max_eval_users,
            seed=seed,
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
) -> Callable[[], Task]:
    def build() -> Task:
        return build_alignment_task(
            dataset_name,
            negative_ratio,
            seed,
            max_eval_users=max_eval_users,
        )

    return build


# TODO: make it parameter dependent. I want to pass seeds, ratios, dataset and get the collections of tasks.
FULL_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    f"{dataset_name}_cap20_m{negative_ratio}_seed{seed}": _make_builder(
        dataset_name,
        negative_ratio,
        seed,
    )
    for dataset_name in DATASETS
    for negative_ratio in NEGATIVE_RATIOS
    for seed in SEEDS
}

EVAL1000_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    f"{dataset_name}_cap20_eval_users{EVAL_USERS}_m{negative_ratio}_seed{seed}": _make_builder(
        dataset_name,
        negative_ratio,
        seed,
        max_eval_users=EVAL_USERS,
    )
    for dataset_name in DATASETS
    for negative_ratio in NEGATIVE_RATIOS
    for seed in SEEDS
}

DEFAULT_TASK_NAMES = list(EVAL1000_TASK_BUILDERS)
TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    **EVAL1000_TASK_BUILDERS,
    **FULL_TASK_BUILDERS,
}
