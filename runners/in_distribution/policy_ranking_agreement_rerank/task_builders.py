"""Task registry for preference-target policy ranking agreement.

Same filtering, splitting, policy roster and recommendation construction as
`policy_ranking_agreement`, with two deliberate differences:

1. `target_source_column`: the real held-out utility of a policy is the rate at
   which its recommendations were *liked* (`target_like_ge4` /
   `target_played_120`) rather than merely interacted with.
2. Policy training signal: every trainable policy is fit only on
   positive-preference train rows (`positive_feedback_column` =
   the dataset's preference target), so policies actually optimize the target
   they are evaluated on. Seen-item exclusion still covers all consumed items,
   liked or not. `RandomPolicy` has no training signal and is unchanged.

Policy hyperparameters come from the committed, manually reviewed policy
config (`configs/policies/preference.json`, see `docs/policy_configs.md`),
drafted from offline Optuna tuning artifacts
(`src/beyond_click_sim/tasks/policy_tuning.py`).

Users with zero positive-preference train rows are cold-start for the CF
policies (no recommendations); utility aggregation is restricted to users
covered by every policy (see `methods/common.py`) so the comparison stays over
one shared user population.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import MinUserInteractionsFilter, Task
from beyond_click_sim.tasks.policies import Policy
from beyond_click_sim.tasks.policy_config import load_policy_roster
from beyond_click_sim.tasks.policy_reranking import PolicyRerankingTaskBuilder
from beyond_click_sim.tasks.splitters import SourcePartitionSplitter


VERSION = "v1"
# One seed only, and that is not an oversight. On an unbiased re-ranking run the
# reference is the entire MCAR half and every user is evaluated, so there is no
# split and no user sampling for a seed to vary; `policy_seed` stays 0 as it is
# on ml-1m. Ten seeds would produce ten byte-identical runs.
SEEDS = (0,)
MIN_INTERACTIONS = 10
DATASETS = ("coat",)
# k = 5 of a 16-item pool. k = 10 would select 62% of the pool and leave almost
# no room for policies to differ.
POLICY_K = 5

DATASET_HISTORY_CONTEXT_COLUMNS: dict[str, tuple[str, ...]] = {
    # Carrying `rating` matters more here than on ml-1m: 48% of Coat's
    # self-selected ratings are <= 2, so a history without them would read as an
    # endorsement of coats the participant actively disliked.
    "coat": ("rating",),
}

# Positive-preference targets, identical to the preference-prediction protocol:
# the held-out outcome is "did the user like it", not "did an interaction occur".
DATASET_TARGET_SOURCE_COLUMNS: dict[str, str] = {
    "coat": "target_like_ge4",
}


def _make_policies(dataset_name: str, seed: int) -> list[Policy]:
    """The policy roster from the committed, manually reviewed config.

    Policies train on the dataset's positive-preference target
    (`positive_feedback_column`). Raises if
    `configs/policies/preference.json` has no section for this (dataset, seed)
    — see `docs/policy_configs.md`.
    """

    return load_policy_roster(
        setup="preference",
        dataset_name=dataset_name,
        split_seed=seed,
        policy_k=POLICY_K,
        positive_feedback_column=DATASET_TARGET_SOURCE_COLUMNS[dataset_name],
        repo_root=repo_root(),
    )


def build_policy_ranking_task(
    dataset_name: str,
    seed: int,
) -> Task:
    """Build one preference-target policy re-ranking task."""

    dataset = load_canonical_dataset(dataset_name)
    builder = PolicyRerankingTaskBuilder(
        name=f"{dataset_name}_policy_ranking_preference_rerank_seed{seed}",
        dataset_filter=MinUserInteractionsFilter(min_interactions=MIN_INTERACTIONS),
        splitter=SourcePartitionSplitter(),
        policies=_make_policies(dataset_name, seed),
        target_source_column=DATASET_TARGET_SOURCE_COLUMNS[dataset_name],
        history_context_columns=DATASET_HISTORY_CONTEXT_COLUMNS[dataset_name],
        eval_sampler=None,
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


def _make_builder(dataset_name: str, seed: int) -> Callable[[], Task]:
    def build() -> Task:
        return build_policy_ranking_task(dataset_name, seed)

    return build


# Registry keys are kept byte-identical to the `task.name` set by the builder so
# that an output directory and the `task` field inside its metrics.json agree.
#
# One registry, not two: the ml-1m families carry an `eval_users{N}` variant
# because they subsample evaluation users. Here every user is evaluated, so the
# suffix would be noise and there is nothing to sample.
TASK_BUILDERS: dict[str, Callable[[], Task]] = {
    f"{dataset_name}_policy_ranking_preference_rerank_seed{seed}": _make_builder(
        dataset_name,
        seed,
    )
    for dataset_name in DATASETS
    for seed in SEEDS
}

DEFAULT_TASK_NAMES = list(TASK_BUILDERS)
