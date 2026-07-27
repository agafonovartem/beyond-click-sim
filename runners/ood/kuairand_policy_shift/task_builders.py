from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    KuaiRandStandardToRandomTask,
    KuaiRandStandardToRandomTaskBuilder,
    TargetPool,
)


DATASET_NAME = "kuairand-pure"
DATASET_VERSION = "v1"
DEFAULT_CANDIDATE_SEEDS = (0, 1, 2)
DEFAULT_COHORT_SEED = 0
DEFAULT_HISTORY_SIZE = 20
DEFAULT_CANDIDATE_SIZE = 10
DEFAULT_VALIDATION_USERS = 200
DEFAULT_TEST_USERS = 500
DEFAULT_MIN_SOURCE_EXPOSURES = 20
DEFAULT_MIN_SOURCE_USERS = 5


def repo_root() -> Path:
    """Locate the beyond-click-sim repository without assuming the CWD."""

    for path in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        pyproject = path / "pyproject.toml"
        if pyproject.exists() and 'name = "beyond-click-sim"' in pyproject.read_text(
            encoding="utf-8"
        ):
            return path
    raise RuntimeError("Could not find beyond-click-sim repo root")


def workspace_data_root() -> Path:
    """Locate the shared research-workspace data directory."""

    for path in [repo_root(), *repo_root().parents]:
        candidate = path / "data"
        if (candidate / "canonical").exists():
            return candidate
    return repo_root().parents[2] / "data"


def load_canonical_dataset(
    root: Path | None = None,
) -> CanonicalDataset:
    canonical_root = (
        workspace_data_root()
        / "canonical"
        / DATASET_NAME
        / DATASET_VERSION
        if root is None
        else root.expanduser().resolve()
    )
    return CanonicalDataset(
        name=DATASET_NAME,
        version=DATASET_VERSION,
        root=canonical_root,
        users_path=canonical_root / "users.parquet",
        items_path=canonical_root / "items.parquet",
        interactions_path=canonical_root / "interactions.parquet",
        manifest_path=canonical_root / "manifest.json",
    )


def default_translation_dir() -> Path:
    return (
        workspace_data_root()
        / "processed"
        / "kuairand"
        / "semantic_translation_gpt55"
    )


def build_tasks(
    *,
    candidate_seeds: Iterable[int] = DEFAULT_CANDIDATE_SEEDS,
    target_pool: TargetPool = "common_warm",
    dataset: CanonicalDataset | None = None,
    translation_dir: Path | None = None,
) -> dict[int, KuaiRandStandardToRandomTask]:
    """Build every candidate seed and validate their shared source/cohorts."""

    seeds = list(dict.fromkeys(int(seed) for seed in candidate_seeds))
    if not seeds:
        raise ValueError("candidate_seeds must not be empty")
    canonical = dataset or load_canonical_dataset()
    semantics_root = (
        default_translation_dir()
        if translation_dir is None
        else translation_dir.expanduser().resolve()
    )
    tasks = {
        seed: KuaiRandStandardToRandomTaskBuilder(
            translation_dir=semantics_root,
            target_pool=target_pool,
            candidate_seed=seed,
            cohort_seed=DEFAULT_COHORT_SEED,
            history_size=DEFAULT_HISTORY_SIZE,
            candidate_size=DEFAULT_CANDIDATE_SIZE,
            validation_users=DEFAULT_VALIDATION_USERS,
            test_users=DEFAULT_TEST_USERS,
            min_source_exposures=DEFAULT_MIN_SOURCE_EXPOSURES,
            min_source_users=DEFAULT_MIN_SOURCE_USERS,
        ).build(canonical)
        for seed in seeds
    }
    _validate_tasks(tasks)
    return tasks


def shared_test_users(
    tasks_by_seed: dict[int, KuaiRandStandardToRandomTask],
) -> list[object]:
    """Return users present under both target regimes for every seed."""

    if not tasks_by_seed:
        raise ValueError("tasks_by_seed must not be empty")
    shared: set[object] | None = None
    for task in tasks_by_seed.values():
        task_users = set(task.test_standard_targets["user_id"])
        task_users &= set(task.test_random_targets["user_id"])
        shared = task_users if shared is None else shared & task_users
    return sorted(shared or set(), key=repr)


def select_test_users(
    users: Iterable[object],
    *,
    max_users: int | None,
    seed: int = 0,
) -> list[object]:
    """Select a deterministic, outcome-blind subset for smoke runs."""

    unique = set(users)
    ordered = sorted(
        unique,
        key=lambda user_id: (
            _stable_int(seed, "runner-test-subset", user_id),
            repr(user_id),
        ),
    )
    if max_users is None or max_users == 0:
        return ordered
    if max_users < 0:
        raise ValueError("max_users must be non-negative")
    return ordered[:max_users]


def _validate_tasks(
    tasks_by_seed: dict[int, KuaiRandStandardToRandomTask],
) -> None:
    source_hashes = {
        task.manifest["hashes"]["frames"]["training_source"]
        for task in tasks_by_seed.values()
    }
    visible_hashes = {
        task.manifest["hashes"]["frames"]["visible_history"]
        for task in tasks_by_seed.values()
    }
    validation_cohorts = {
        task.manifest["cohorts"]["validation"]["user_ids_sha256"]
        for task in tasks_by_seed.values()
    }
    test_cohorts = {
        task.manifest["cohorts"]["test"]["user_ids_sha256"]
        for task in tasks_by_seed.values()
    }
    if len(source_hashes) != 1:
        raise ValueError("Candidate seeds produced different source frames")
    if len(visible_hashes) != 1:
        raise ValueError("Candidate seeds produced different visible histories")
    if len(validation_cohorts) != 1 or len(test_cohorts) != 1:
        raise ValueError("Candidate seeds produced different user cohorts")
    if len(shared_test_users(tasks_by_seed)) != DEFAULT_TEST_USERS:
        raise ValueError("Candidate seeds do not share the fixed 500-user test cohort")


def _stable_int(seed: int, namespace: str, value: object) -> int:
    payload = f"{seed}\0{namespace}\0{value!r}".encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=16).digest(),
        "big",
    )
