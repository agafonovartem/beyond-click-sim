from __future__ import annotations

from collections.abc import Iterable
import hashlib
import inspect
from pathlib import Path

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    KuaiRandNaturalExposureTaskBuilder,
    KuaiRandPolicyShiftTask,
    KuaiRandPolicyShiftTaskBuilder,
)


VERSION = "v1"
DATASET_NAME = "kuairand-pure"
DEFAULT_SOURCE_SEED = 42
DEFAULT_CANDIDATE_SEEDS = (42, 43, 44)
DEFAULT_COHORT_SEED = 20260724
DEFAULT_NATURAL_CANDIDATE_SEEDS = (0, 1, 2)


def repo_root() -> Path:
    """Locate the beyond-click-sim repository without assuming the CWD."""

    for path in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        pyproject = path / "pyproject.toml"
        if pyproject.exists() and 'name = "beyond-click-sim"' in pyproject.read_text(
            encoding="utf-8"
        ):
            return path
    raise RuntimeError("Could not find beyond-click-sim repo root")


def data_root() -> Path:
    """Locate the shared canonical-data directory."""

    for path in [repo_root(), *repo_root().parents]:
        candidate = path / "data" / "canonical"
        if candidate.exists():
            return candidate
    # Returning the conventional path gives a useful missing-file error even
    # before the first canonicalization run.
    return repo_root().parents[1] / "data" / "canonical"


def load_canonical_dataset(root: Path | None = None) -> CanonicalDataset:
    canonical_root = (
        data_root() / DATASET_NAME / VERSION
        if root is None
        else root.expanduser().resolve()
    )
    return CanonicalDataset(
        name=DATASET_NAME,
        version=VERSION,
        root=canonical_root,
        users_path=canonical_root / "users.parquet",
        items_path=canonical_root / "items.parquet",
        interactions_path=canonical_root / "interactions.parquet",
        manifest_path=canonical_root / "manifest.json",
    )


def build_policy_shift_task(
    *,
    candidate_seed: int,
    source_seed: int = DEFAULT_SOURCE_SEED,
    min_positive_source_users_per_policy: int = 1,
    dataset: CanonicalDataset | None = None,
) -> KuaiRandPolicyShiftTask:
    """Build one task seed while remaining compatible with the seed API.

    Early versions of the specialized builder exposed one ``seed``. The final
    protocol separates source-history sampling from candidate sampling. This
    small compatibility shim lets the runner work during that migration and
    records the effective seed interface in the task manifest.
    """

    builder_parameters = inspect.signature(
        KuaiRandPolicyShiftTaskBuilder
    ).parameters
    name = (
        "kuairand_tab1_policy_shift_c3"
        f"_posdeg{min_positive_source_users_per_policy}"
        f"_source{source_seed}_candidate{candidate_seed}"
    )
    if {"source_seed", "candidate_seed"}.issubset(builder_parameters):
        builder = KuaiRandPolicyShiftTaskBuilder(
            name=name,
            source_seed=source_seed,
            candidate_seed=candidate_seed,
            min_positive_source_users_per_policy=(
                min_positive_source_users_per_policy
            ),
        )
    else:
        if source_seed != candidate_seed and source_seed != DEFAULT_SOURCE_SEED:
            raise ValueError(
                "This task-builder version has a single seed; source_seed must "
                f"remain {DEFAULT_SOURCE_SEED}"
            )
        builder = KuaiRandPolicyShiftTaskBuilder(
            name=name,
            seed=candidate_seed,
        )
        if min_positive_source_users_per_policy != 1:
            raise ValueError(
                "This task-builder version cannot set "
                "min_positive_source_users_per_policy"
            )
    task = builder.build(dataset or load_canonical_dataset())
    task.manifest.setdefault("runner_seed_interface", {})
    task.manifest["runner_seed_interface"].update(
        {
            "requested_source_seed": source_seed,
            "requested_candidate_seed": candidate_seed,
            "requested_min_positive_source_users_per_policy": (
                min_positive_source_users_per_policy
            ),
            "builder_has_separate_source_candidate_seeds": {
                "source_seed",
                "candidate_seed",
            }.issubset(builder_parameters),
        }
    )
    return task


def build_natural_policy_shift_task(
    *,
    candidate_seed: int,
    dataset: CanonicalDataset | None = None,
) -> KuaiRandPolicyShiftTask:
    """Build the locked H=5, m=5 label-blind natural-exposure task."""

    return KuaiRandNaturalExposureTaskBuilder(
        candidate_seed=candidate_seed,
    ).build(dataset or load_canonical_dataset())


def shared_test_users(
    tasks_by_seed: dict[int, KuaiRandPolicyShiftTask],
) -> list[object]:
    """Return users evaluable under both target policies for every seed."""

    if not tasks_by_seed:
        raise ValueError("tasks_by_seed must not be empty")
    shared: set[object] | None = None
    for task in tasks_by_seed.values():
        task_users = set(task.test_standard_targets["user_id"])
        task_users &= set(task.test_random_targets["user_id"])
        shared = task_users if shared is None else shared & task_users
    return sorted(shared or set(), key=repr)


def select_user_cohort(
    users: Iterable[object],
    *,
    max_users: int | None,
    cohort_seed: int = DEFAULT_COHORT_SEED,
) -> list[object]:
    """Select an outcome-blind stable prefix shared by every task seed."""

    unique = set(users)
    ordered = sorted(
        unique,
        key=lambda user_id: (
            _stable_int(cohort_seed, "kuairand-policy-shift-cohort", user_id),
            repr(user_id),
        ),
    )
    if max_users is None or max_users == 0:
        return ordered
    if max_users < 0:
        raise ValueError("max_users must be non-negative")
    return ordered[:max_users]


def _stable_int(seed: int, namespace: str, value: object) -> int:
    payload = f"{seed}\0{namespace}\0{value!r}".encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=16).digest(),
        byteorder="big",
    )
