"""Offline hyperparameter tuning for policy-ranking recommender policies.

For each (setup, dataset, split seed) this runs an Optuna search on an inner
validation split carved from the task's train portion (test data is never
touched) and writes one JSON artifact per job to `default_artifacts_dir()`
(`<repo root>/artifacts/`). The policy-ranking task builders consume these
artifacts via `build_tuned_policy_roster`.

Usage:
    uv run python -m runners.in_distribution.policy_tuning.run
    uv run python -m runners.in_distribution.policy_tuning.run \
        --datasets ml-1m --seeds 0 --policies LightGCNTorchPolicy
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import optuna

from beyond_click_sim.tasks import MinUserInteractionsFilter, RandomFractionSplitter
from beyond_click_sim.tasks.base import TaskBuilder
from beyond_click_sim.tasks.policy_config import (
    build_config_skeleton,
    default_policies_config_dir,
)
from beyond_click_sim.tasks.policy_tuning import (
    POLICY_CLASSES,
    artifact_filename,
    default_artifacts_dir,
    tune_policies_for_split,
)
from runners.in_distribution.policy_ranking_agreement.methods.common import (
    current_git_commit,
    json_safe,
)
from runners.in_distribution.policy_ranking_agreement.task_builders import (
    DATASETS,
    MIN_INTERACTIONS,
    POLICY_K,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VERSION,
    load_canonical_dataset,
    repo_root,
)
from runners.in_distribution.policy_ranking_agreement.task_builders import (
    SEEDS as INTERACTION_SEEDS,
)
from runners.in_distribution.policy_ranking_agreement_preference.task_builders import (
    DATASET_TARGET_SOURCE_COLUMNS,
)
from runners.in_distribution.policy_ranking_agreement_preference.task_builders import (
    SEEDS as PREFERENCE_SEEDS,
)

SETUP_NAMES = ("interaction", "preference")
SETUP_SEEDS = {
    "interaction": INTERACTION_SEEDS,
    "preference": PREFERENCE_SEEDS,
}
DEFAULT_ARTIFACTS_DIR = default_artifacts_dir()


def resolve_seeds(setup: str, requested: list[int] | None) -> tuple[int, ...]:
    """Seeds to tune for one setup: explicit --seeds wins, else the setup's registry."""

    if requested is not None:
        return tuple(requested)
    return tuple(SETUP_SEEDS[setup])


def setup_target(setup: str, dataset_name: str) -> tuple[str, str | None]:
    """(target_source_column, positive_feedback_column) for one setup."""

    if setup == "interaction":
        return "target_interact", None
    if setup == "preference":
        target = DATASET_TARGET_SOURCE_COLUMNS[dataset_name]
        return target, target
    raise ValueError(f"Unknown setup: {setup!r}. Available: {sorted(SETUP_SEEDS)}")


def tune_one(
    setup: str,
    dataset_name: str,
    seed: int,
    *,
    filtered_frames: tuple,
    dataset_filter: MinUserInteractionsFilter,
    splitter: RandomFractionSplitter,
    output_dir: Path,
    policy_class_names: list[str] | None,
) -> Path:
    users, items, train_interactions = filtered_frames
    target_source_column, positive_feedback_column = setup_target(setup, dataset_name)
    policy_classes = (
        None
        if policy_class_names is None
        else tuple(POLICY_CLASSES[name] for name in policy_class_names)
    )
    artifact = tune_policies_for_split(
        train_interactions=train_interactions,
        users=users,
        items=items,
        dataset_name=dataset_name,
        dataset_version=VERSION,
        target_source_column=target_source_column,
        positive_feedback_column=positive_feedback_column,
        split_seed=seed,
        policy_k=POLICY_K,
        filter_config=TaskBuilder._component_manifest(dataset_filter),
        splitter_config=TaskBuilder._component_manifest(splitter),
        policy_classes=policy_classes,
    )
    artifact["setup"] = setup
    artifact["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact["git_commit"] = current_git_commit(repo_root())
    artifact["optuna_version"] = optuna.__version__

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / artifact_filename(dataset_name, target_source_column, seed)
    if policy_class_names is not None and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for name, entry in previous["policies"].items():
            if name not in artifact["policies"]:
                artifact["policies"][name] = entry
    elif policy_class_names is not None:
        print(
            f"warning: writing partial artifact {path.name} "
            f"(policies={sorted(artifact['policies'])})",
            flush=True,
        )
    path.write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def emit_configs(
    *,
    setups: list[str],
    datasets: list[str],
    requested_seeds: list[int] | None,
    artifacts_dir: Path,
    overwrite: bool,
) -> None:
    """Draft `configs/policies/{setup}.json` skeletons from tuning artifacts.

    The skeleton is a starting point for manual review: default rosters come
    from each dataset's lowest-seed artifact, and `_review_notes` summarize
    cross-seed hyperparameter divergence. Existing config files are never
    overwritten unless `overwrite=True`.
    """

    config_dir = default_policies_config_dir()
    for setup in setups:
        dataset_sections: dict[str, dict[int, dict]] = {}
        for dataset_name in datasets:
            target, _ = setup_target(setup, dataset_name)
            seed_to_artifact: dict[int, dict] = {}
            for seed in resolve_seeds(setup, requested_seeds):
                path = artifacts_dir / artifact_filename(dataset_name, target, seed)
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    missing = sorted(set(POLICY_CLASSES) - set(payload.get("policies", {})))
                    if missing:
                        raise RuntimeError(
                            f"Partial artifact {path.name} is missing policies "
                            f"{missing}; finish tuning before --emit-config."
                        )
                    seed_to_artifact[seed] = payload
                else:
                    print(
                        f"warning: no artifact for {setup}/{dataset_name}/"
                        f"seed{seed} ({path.name}); skipping",
                        flush=True,
                    )
            if seed_to_artifact:
                dataset_sections[dataset_name] = seed_to_artifact
        if not dataset_sections:
            print(f"No artifacts found for setup {setup!r}; nothing emitted.", flush=True)
            continue
        skeleton = build_config_skeleton(
            setup=setup, dataset_sections=dataset_sections
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        out = config_dir / f"{setup}.json"
        if out.exists() and not overwrite:
            raise RuntimeError(
                f"Config already exists: {out}. Review/merge your hand edits "
                "manually, or pass --overwrite to replace it."
            )
        out.write_text(
            json.dumps(json_safe(skeleton), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune policy hyperparameters per (setup, dataset, seed).",
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated split seeds, applied to all selected setups. "
            "Default: each setup's own SEEDS registry."
        ),
    )
    parser.add_argument("--setups", default=",".join(SETUP_NAMES))
    parser.add_argument(
        "--policies",
        default=None,
        help=(
            "Comma-separated policy class names to tune (subset). Writes a "
            "partial artifact if none exists yet; otherwise merges into the "
            "existing file. Default: all policies."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose artifact already exists.",
    )
    parser.add_argument(
        "--emit-config",
        action="store_true",
        help=(
            "Instead of tuning, draft configs/policies/{setup}.json skeletons "
            "from existing artifacts for manual review."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --emit-config to overwrite an existing config file.",
    )
    args = parser.parse_args()

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    requested_seeds = (
        None
        if args.seeds is None
        else [int(s) for s in args.seeds.split(",") if s.strip()]
    )
    setups = [name.strip() for name in args.setups.split(",") if name.strip()]
    unknown_setups = [name for name in setups if name not in SETUP_SEEDS]
    if unknown_setups:
        raise ValueError(
            f"Unknown setups: {unknown_setups}. Available: {sorted(SETUP_SEEDS)}"
        )
    policy_class_names = (
        None
        if args.policies is None
        else [name.strip() for name in args.policies.split(",") if name.strip()]
    )
    unknown = [name for name in (policy_class_names or []) if name not in POLICY_CLASSES]
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}. Available: {sorted(POLICY_CLASSES)}")
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.emit_config:
        emit_configs(
            setups=setups,
            datasets=datasets,
            requested_seeds=requested_seeds,
            artifacts_dir=output_dir,
            overwrite=args.overwrite,
        )
        return

    total = len(datasets) * sum(
        len(resolve_seeds(setup, requested_seeds)) for setup in setups
    )
    job_index = 0
    for dataset_name in datasets:
        dataset = load_canonical_dataset(dataset_name)
        dataset_filter = MinUserInteractionsFilter(min_interactions=MIN_INTERACTIONS)
        users, items, interactions = dataset_filter.filter(
            users=dataset.load_users(),
            items=dataset.load_items(),
            interactions=dataset.load_interactions(),
        )
        split_cache: dict[int, tuple] = {}
        for setup in setups:
            for seed in resolve_seeds(setup, requested_seeds):
                job_index += 1
                if seed not in split_cache:
                    splitter = RandomFractionSplitter(
                        train_fraction=TRAIN_FRACTION,
                        val_fraction=0.0,
                        test_fraction=TEST_FRACTION,
                        seed=seed,
                    )
                    split_cache[seed] = (splitter, splitter.split(interactions))
                splitter, split = split_cache[seed]
                target_source_column, _ = setup_target(setup, dataset_name)
                artifact_path = (
                    output_dir
                    / artifact_filename(dataset_name, target_source_column, seed)
                )
                if args.resume and policy_class_names is None and artifact_path.exists():
                    print(f"Skipping (artifact exists) {job_index}/{total}: {artifact_path.name}", flush=True)
                    continue
                print(
                    f"Tuning {job_index}/{total}: setup={setup} dataset={dataset_name} "
                    f"seed={seed}",
                    flush=True,
                )
                path = tune_one(
                    setup,
                    dataset_name,
                    seed,
                    filtered_frames=(users, items, split.train),
                    dataset_filter=dataset_filter,
                    splitter=splitter,
                    output_dir=output_dir,
                    policy_class_names=policy_class_names,
                )
                print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
