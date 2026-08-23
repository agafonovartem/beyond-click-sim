"""Tune the Coat policy roster under the re-ranking protocol.

A single job — one dataset, one setup, one seed — so none of the parent
runner's setup/seed/dataset matrix applies. What it does share with
`policy_ranking_agreement_rerank` is the data path: the filter, the splitter and
`POLICY_K` are imported from that runner's task builders rather than restated,
so the frame policies fit on here is the frame they fit on at run time by
construction, not by two constants happening to agree.

Two passes are intended, in this order:

    # 1. diagnostic — select on half the users, measure the winner's curse
    uv run python -m runners.in_distribution.policy_tuning_rerank.run \
        --tuning-user-fraction 0.5

    # 2. production roster — select on all 290
    uv run python -m runners.in_distribution.policy_tuning_rerank.run --print-config

Neither pass calls an LLM or touches the network. The config block from pass 2
is merged into `configs/policies/preference.json` by hand: `emit_configs` in the
parent runner rewrites the whole file and would drop the ml-1m section.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import optuna

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from beyond_click_sim.tasks import MinUserInteractionsFilter  # noqa: E402
from beyond_click_sim.tasks.base import TaskBuilder  # noqa: E402
from beyond_click_sim.tasks.policy_tuning import (  # noqa: E402
    POLICY_CLASSES,
    default_artifacts_dir,
)
from beyond_click_sim.tasks.policy_tuning_rerank import (  # noqa: E402
    artifact_filename,
    tune_policies_for_rerank_split,
)
from beyond_click_sim.tasks.splitters import SourcePartitionSplitter  # noqa: E402
from runners.in_distribution.policy_ranking_agreement.methods.common import (  # noqa: E402
    current_git_commit,
    json_safe,
)
from runners.in_distribution.policy_ranking_agreement_rerank.task_builders import (  # noqa: E402
    DATASET_TARGET_SOURCE_COLUMNS,
    DATASETS,
    MIN_INTERACTIONS,
    POLICY_K,
    SEEDS,
    VERSION,
    load_canonical_dataset,
    repo_root,
)

SETUP = "preference"


def tune(
    dataset_name: str,
    seed: int,
    *,
    tuning_user_fraction: float,
    policy_class_names: list[str] | None,
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, dict]:
    dataset = load_canonical_dataset(dataset_name)
    dataset_filter = MinUserInteractionsFilter(min_interactions=MIN_INTERACTIONS)
    users, items, interactions = dataset_filter.filter(
        users=dataset.load_users(),
        items=dataset.load_items(),
        interactions=dataset.load_interactions(),
    )
    splitter = SourcePartitionSplitter()
    split = splitter.split(interactions)

    target = DATASET_TARGET_SOURCE_COLUMNS[dataset_name]
    policy_classes = (
        None
        if policy_class_names is None
        else tuple(POLICY_CLASSES[name] for name in policy_class_names)
    )
    artifact = tune_policies_for_rerank_split(
        train_interactions=split.train,
        val_interactions=split.test,
        users=users,
        items=items,
        dataset_name=dataset_name,
        dataset_version=VERSION,
        target_source_column=target,
        # The preference setup trains policies on the same signal it scores
        # them by, exactly as `_make_policies` does in the rerank runner.
        positive_feedback_column=target,
        split_seed=seed,
        policy_k=POLICY_K,
        tuning_user_fraction=tuning_user_fraction,
        filter_config=TaskBuilder._component_manifest(dataset_filter),
        # Serialized after `split()` so the splitter's overlap-drop counts are
        # populated rather than null.
        splitter_config=TaskBuilder._component_manifest(splitter),
        policy_classes=policy_classes,
    )
    artifact["setup"] = SETUP
    artifact["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact["git_commit"] = current_git_commit(repo_root())
    artifact["optuna_version"] = optuna.__version__

    n_tuning = artifact["identity"]["leakage"]["n_tuning_users"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / artifact_filename(dataset_name, target, n_tuning, seed)
    if path.exists() and not overwrite:
        raise RuntimeError(f"Artifact already exists: {path}. Pass --overwrite to replace it.")
    path.write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, artifact


def config_block(dataset_name: str, artifact: dict, artifact_path: Path) -> dict:
    """The `datasets.<name>` section to paste into configs/policies/preference.json.

    Printed rather than written: the parent runner's `emit_configs` rebuilds the
    whole config file from artifacts and would drop hand-maintained sections for
    other datasets.
    """

    leakage = artifact["identity"]["leakage"]
    entries = []
    for name in sorted(artifact["policies"]):
        entry = artifact["policies"][name]
        record = {
            "class": name,
            "name": name,
            "params": entry["selected_params"],
            "tuned_from": f"artifacts/{artifact_path.name}",
        }
        if not entry["tuned"]:
            record["comment"] = "baseline (no hyperparameters to tune)"
        else:
            record["comment"] = (
                f"grid search over {entry['grid_size']} points, "
                f"{artifact['identity']['objective']}={entry['val_objective']:.4f}"
            )
        entries.append(record)

    notes = [
        f"Tuned under the re-ranking protocol ({artifact['identity']['protocol']}): "
        f"policies fit on the runner's full train frame "
        f"({artifact['identity']['fit_frame']['n_rows']} rows) and re-rank each "
        f"user's held-out MCAR pool, scored by "
        f"{artifact['identity']['objective']} at k={artifact['identity']['policy_k']}. "
        "This is NOT the ml-1m workflow — see docs/policy_configs.md.",
        "LEAKAGE: hyperparameters were selected on "
        f"{leakage['n_tuning_users']} of the {leakage['n_eval_users_in_run']} users "
        "whose labels also form the reference ordering of the re-ranking run. Real "
        "utilities of tuned policies are optimistically biased; RandomPolicy and "
        "PopularityPolicy have nothing to select and carry no such bias, so a "
        "tuned policy overtaking them is partly an artifact of that selection.",
        f"search_spaces_sha256={artifact['identity']['search_spaces_sha256']}, "
        f"search_space_revision={artifact['identity']['search_space_revision']}. "
        "The boxes were revised once against the diagnostic half after every "
        "policy first selected on a boundary; choosing the grid is a layer of "
        "selection above choosing within it, and the measured curse below covers "
        "only the latter.",
    ]
    flagged = {
        name: entry["boundary_params"]
        for name, entry in sorted(artifact["policies"].items())
        if entry.get("boundary_params")
    }
    if flagged:
        notes.append(
            "Selections still on a grid boundary, i.e. reported at a handicap: "
            + "; ".join(f"{name} {', '.join(params)}" for name, params in flagged.items())
            + ". Every edge points the same way — fewer factors, more smoothing — "
            "so these policies are degenerating toward popularity on a 290x300 "
            "matrix rather than sitting at an interior optimum."
        )
    diagnostics = {
        name: entry["curse_diagnostic"]
        for name, entry in artifact["policies"].items()
        if "curse_diagnostic" in entry
    }
    if diagnostics:
        measured = ", ".join(
            f"{name} {d['excess_over_grid']:+.4f} "
            f"[{d['excess_ci95'][0]:+.4f}, {d['excess_ci95'][1]:+.4f}]"
            for name, d in sorted(diagnostics.items())
        )
        notes.append(
            "Measured winner's curse, 95% CI (selected point's tuned-minus-heldout "
            f"gap, minus that gap averaged over the policy's own grid): {measured}. "
            "Policies with a one-point grid are 0 by construction. Measured for a "
            f"selection on {list(diagnostics.values())[0]['n_tuning_users']} users; "
            "a selection on all of them carries roughly 1/sqrt(2) of it."
        )
    return {dataset_name: {"_review_notes": notes, "default": entries}}


def print_summary(artifact: dict) -> None:
    identity = artifact["identity"]
    print(
        f"\nprotocol={identity['protocol']} k={identity['policy_k']} "
        f"objective={identity['objective']} "
        f"fit_frame={identity['fit_frame']['n_rows']} rows / "
        f"{identity['fit_frame']['n_users']} users "
        f"| tuning users {identity['leakage']['n_tuning_users']}"
        f"/{identity['leakage']['n_eval_users_in_run']}\n"
    )
    header = f"{'policy':<24}{'objective':>10}{'grid':>6}  selected"
    has_diag = any("curse_diagnostic" in e for e in artifact["policies"].values())
    if has_diag:
        header = (
            f"{'policy':<24}{'tuned':>9}{'heldout':>9}{'curse':>9}"
            f"{'ci95':>22}{'grid':>6}  selected"
        )
    print(header)
    print("-" * len(header))
    for name in sorted(artifact["policies"]):
        entry = artifact["policies"][name]
        params = json.dumps(entry["selected_params"], sort_keys=True)
        if has_diag:
            d = entry["curse_diagnostic"]
            ci = f"[{d['excess_ci95'][0]:+.4f}, {d['excess_ci95'][1]:+.4f}]"
            print(
                f"{name:<24}{d['tuning_utility']:>9.4f}{d['holdout_utility']:>9.4f}"
                f"{d['excess_over_grid']:>+9.4f}{ci:>22}{entry['grid_size']:>6}  {params}"
            )
        else:
            print(
                f"{name:<24}{entry['val_objective']:>10.4f}"
                f"{entry['grid_size']:>6}  {params}"
            )
    if has_diag:
        print(
            "\ncurse = selected point's tuned-minus-heldout gap, minus that gap "
            "averaged over the policy's whole grid.\nA one-point grid is 0 by "
            "construction: nothing was selected, so nothing was inflated."
        )
    flagged = {
        name: entry["boundary_params"]
        for name, entry in sorted(artifact["policies"].items())
        if entry.get("boundary_params")
    }
    if flagged:
        print(
            "\nboundary selections (optimum may lie outside the box, so the "
            "policy is reported at a handicap):"
        )
        for name, params in flagged.items():
            print(f"  {name}: {', '.join(params)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASETS[0], choices=list(DATASETS))
    parser.add_argument("--seed", type=int, default=SEEDS[0], choices=list(SEEDS))
    parser.add_argument(
        "--tuning-user-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of evaluation users to select hyperparameters on. 1.0 is "
            "the production roster; a value below 1 additionally scores the "
            "selection on the untuned users and reports the winner's curse."
        ),
    )
    parser.add_argument("--policies", nargs="+", default=None, choices=sorted(POLICY_CLASSES))
    parser.add_argument("--output-dir", type=Path, default=default_artifacts_dir())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    path, artifact = tune(
        args.dataset,
        args.seed,
        tuning_user_fraction=args.tuning_user_fraction,
        policy_class_names=args.policies,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print_summary(artifact)
    print(f"\nWrote {path}")

    if args.print_config:
        block = config_block(args.dataset, artifact, path)
        print(
            f"\n--- paste into configs/policies/{SETUP}.json under \"datasets\" ---\n"
        )
        print(json.dumps(json_safe(block), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
