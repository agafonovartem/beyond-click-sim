from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    MinUserInteractionsFilter,
    RandomFractionSplitter,
    Task,
)
from beyond_click_sim.tasks.policies import Policy
from beyond_click_sim.tasks.policy_config import load_policy_roster
from beyond_click_sim.tasks.policy_ranking import PolicyRankingTaskBuilder


VERSION = "v1"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
MIN_INTERACTIONS = 10
DATASETS = ("ml-1m", "steam")
TRAIN_FRACTION = 0.8
TEST_FRACTION = 0.2
EVAL_USERS = 1000
POLICY_K = 10

DATASET_HISTORY_CONTEXT_COLUMNS: dict[str, tuple[str, ...]] = {
    "ml-1m": ("rating",),
    "steam": ("playtime_forever",),
}


def _make_policies(dataset_name: str, seed: int) -> list[Policy]:
    """The policy roster from the committed, manually reviewed config.

    Raises if `configs/policies/interaction.json` has no section for this
    (dataset, seed) — see `docs/policy_configs.md`.
    """

    return load_policy_roster(
        setup="interaction",
        dataset_name=dataset_name,
        split_seed=seed,
        policy_k=POLICY_K,
        positive_feedback_column=None,
        repo_root=repo_root(),
    )


def build_policy_ranking_task(
    dataset_name: str,
    seed: int,
    *,
    max_eval_users: int | None = None,
) -> Task:
    dataset = load_canonical_dataset(dataset_name)
    eval_suffix = "" if max_eval_users is None else f"_eval_users{max_eval_users}"
    builder = PolicyRankingTaskBuilder(
        name=f"{dataset_name}_policy_ranking{eval_suffix}_seed{seed}",
        dataset_filter=MinUserInteractionsFilter(min_interactions=MIN_INTERACTIONS),
        splitter=RandomFractionSplitter(
            train_fraction=TRAIN_FRACTION,
            val_fraction=0.0,
            test_fraction=TEST_FRACTION,
            seed=seed,
        ),
        policies=_make_policies(dataset_name, seed),
        history_context_columns=DATASET_HISTORY_CONTEXT_COLUMNS[dataset_name],
        eval_sampler=None if max_eval_users is None else _user_sampler(max_eval_users, seed),
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
        pyproject = path / "pyproject.toml"
        if pyproject.exists() and _is_beyond_click_sim_pyproject(pyproject):
            return path
    raise RuntimeError("Could not find beyond-click-sim repo root")


def data_root() -> Path:
    for path in [repo_root(), *repo_root().parents]:
        candidate = path / "data" / "canonical"
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not find data/canonical")


def _is_beyond_click_sim_pyproject(path: Path) -> bool:
    return 'name = "beyond-click-sim"' in path.read_text(encoding="utf-8")


def _user_sampler(max_eval_users: int, seed: int):
    from beyond_click_sim.tasks import PostSplitUserSampler

    return PostSplitUserSampler(n_users=max_eval_users, seed=seed)


def _make_builder(
    dataset_name: str,
    seed: int,
    *,
    max_eval_users: int | None = None,
) -> Callable[[], Task]:
    def build() -> Task:
        return build_policy_ranking_task(
            dataset_name,
            seed,
            max_eval_users=max_eval_users,
        )

    return build


EVAL1000_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    f"{dataset_name}_policy_ranking_eval_users{EVAL_USERS}_seed{seed}": _make_builder(
        dataset_name,
        seed,
        max_eval_users=EVAL_USERS,
    )
    for dataset_name in DATASETS
    for seed in SEEDS
}

FULL_TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    f"{dataset_name}_policy_ranking_seed{seed}": _make_builder(
        dataset_name,
        seed,
    )
    for dataset_name in DATASETS
    for seed in SEEDS
}

DEFAULT_TASK_NAMES = list(EVAL1000_TASK_BUILDERS)
TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    **EVAL1000_TASK_BUILDERS,
    **FULL_TASK_BUILDERS,
}
