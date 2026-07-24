from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from beyond_click_sim.tasks.kuairand_policy_shift import _frame_sha256
from runners.ood.kuairand_policy_shift.suite_analysis import (
    CAPTION_METHOD,
    ITEM_CTR_METHOD,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_METRIC,
    DEFAULT_SEED,
    DEFAULT_SIGN_FLIP_SAMPLES,
    MethodArtifact,
    _canonical_targets,
    _input_provenance,
    _load_method_artifact,
    _markdown_table,
    _validate_input_parity,
    _write_csv,
    _write_json,
    analyze_suite,
)
from runners.ood.kuairand_policy_shift.task_builders import (
    build_policy_shift_task,
)


POLICIES = ("standard", "random")
DEFAULT_MIN_POSITIVE_DEGREE = 2
DEFAULT_MIN_ANALYSIS_USERS = 30


def compute_nested_eligibility(
    targets: pd.DataFrame,
    *,
    positive_degrees: Mapping[str, pd.Series],
    min_positive_degree: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[object]]:
    """Audit unchanged c3 rows and return a cross-seed nested user cohort.

    A user-seed pair passes only when all three candidates in both the standard
    and random target c3 have at least ``min_positive_degree`` source users in
    each of the two policy histories. The final cohort is the intersection of
    passing users over every candidate-resample seed.
    """

    if min_positive_degree <= 0:
        raise ValueError("min_positive_degree must be positive")
    required = {
        "task_seed",
        "user_id",
        "candidate_group",
        "target_policy",
        "item_id",
    }
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"Targets are missing columns: {missing}")
    if set(positive_degrees) != set(POLICIES):
        raise ValueError(
            f"positive_degrees must contain exactly {POLICIES}"
        )

    candidates = targets.loc[:, sorted(required)].drop_duplicates().copy()
    duplicate_keys = ["task_seed", "target_policy", "candidate_group", "item_id"]
    if candidates.duplicated(duplicate_keys).any():
        raise ValueError("Targets contain duplicate candidate rows")
    for policy in POLICIES:
        candidates[f"{policy}_positive_source_user_degree"] = (
            candidates["item_id"]
            .map(positive_degrees[policy])
            .fillna(0)
            .astype(int)
        )
    degree_columns = [
        f"{policy}_positive_source_user_degree" for policy in POLICIES
    ]
    candidates["minimum_positive_source_user_degree"] = candidates[
        degree_columns
    ].min(axis=1)
    candidates["passes_degree_threshold"] = candidates[
        "minimum_positive_source_user_degree"
    ].ge(min_positive_degree)

    policy_groups = (
        candidates.groupby(
            ["task_seed", "user_id", "target_policy"],
            sort=False,
        )
        .agg(
            candidate_rows=("item_id", "size"),
            unique_candidate_items=("item_id", "nunique"),
            minimum_positive_source_user_degree=(
                "minimum_positive_source_user_degree",
                "min",
            ),
            passes_degree_threshold=("passes_degree_threshold", "all"),
        )
        .reset_index()
    )
    if not policy_groups["candidate_rows"].eq(3).all():
        raise ValueError("Every user-seed-policy target must contain exactly c3")
    if not policy_groups["unique_candidate_items"].eq(3).all():
        raise ValueError("Every c3 must contain three distinct candidate items")

    user_seed = (
        policy_groups.groupby(["task_seed", "user_id"], sort=False)
        .agg(
            target_policies=("target_policy", "nunique"),
            candidate_rows=("candidate_rows", "sum"),
            minimum_positive_source_user_degree=(
                "minimum_positive_source_user_degree",
                "min",
            ),
            passes_both_target_policies=("passes_degree_threshold", "all"),
        )
        .reset_index()
    )
    if not user_seed["target_policies"].eq(2).all():
        raise ValueError("Every user-seed must have both target policies")
    if not user_seed["candidate_rows"].eq(6).all():
        raise ValueError("Every user-seed must contain two unchanged c3 groups")

    passing_by_seed = {
        int(seed): set(
            rows.loc[
                rows["passes_both_target_policies"],
                "user_id",
            ]
        )
        for seed, rows in user_seed.groupby("task_seed", sort=True)
    }
    common: set[object] | None = None
    for users in passing_by_seed.values():
        common = users if common is None else common & users
    common_users = sorted(common or set(), key=repr)
    user_seed["in_common_cross_seed_cohort"] = user_seed["user_id"].isin(
        common_users
    )
    return candidates, user_seed, common_users


def run_nested_support_audit(
    *,
    gpt_run_dir: Path,
    classical_run_dir: Path,
    output_dir: Path,
    min_positive_degree: int = DEFAULT_MIN_POSITIVE_DEGREE,
    min_analysis_users: int = DEFAULT_MIN_ANALYSIS_USERS,
    metric: str = DEFAULT_METRIC,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    sign_flip_samples: int = DEFAULT_SIGN_FLIP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run a no-inference nested audit on the original k=1 predictions."""

    if min_analysis_users <= 0:
        raise ValueError("min_analysis_users must be positive")
    gpt_run_dir = gpt_run_dir.expanduser().resolve()
    classical_run_dir = classical_run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    artifacts: dict[str, MethodArtifact] = {
        "caption_gpt": _load_method_artifact(
            gpt_run_dir / CAPTION_METHOD,
            expected_method=CAPTION_METHOD,
        ),
        "item_ctr": _load_method_artifact(
            classical_run_dir / ITEM_CTR_METHOD,
            expected_method=ITEM_CTR_METHOD,
        ),
    }
    parity = _validate_input_parity(artifacts)
    _validate_original_k1_protocol(artifacts)

    # All candidate seeds use the exact same source_seed=42 histories. Rebuild
    # once, then verify the hashes against every saved task manifest before
    # using their source labels for the degree audit.
    task = build_policy_shift_task(
        candidate_seed=int(parity["task_seeds"][0]),
        source_seed=42,
        min_positive_source_users_per_policy=1,
    )
    histories = {
        "standard": task.standard_history,
        "random": task.random_history,
    }
    history_hashes = {
        policy: _frame_sha256(history)
        for policy, history in histories.items()
    }
    _validate_source_history_hashes(artifacts, history_hashes=history_hashes)
    positive_degrees = {
        policy: (
            history.loc[history["history_click"].eq(1)]
            .drop_duplicates(["user_id", "item_id"])
            .groupby("item_id", sort=False)["user_id"]
            .nunique()
        )
        for policy, history in histories.items()
    }

    canonical_targets = _canonical_targets(
        artifacts["caption_gpt"].predictions
    )
    candidate_audit, user_seed, common_users = compute_nested_eligibility(
        canonical_targets,
        positive_degrees=positive_degrees,
        min_positive_degree=min_positive_degree,
    )
    per_seed = _per_seed_feasibility(user_seed)
    per_policy_seed = _per_policy_seed_feasibility(candidate_audit)
    analysis_performed = len(common_users) >= min_analysis_users
    comparisons: list[dict[str, Any]] = []
    if analysis_performed:
        comparisons = analyze_suite(
            caption=_filter_users(
                artifacts["caption_gpt"].per_user,
                common_users,
            ),
            categories=None,
            item_ctr=_filter_users(
                artifacts["item_ctr"].per_user,
                common_users,
            ),
            metric=metric,
            bootstrap_samples=bootstrap_samples,
            sign_flip_samples=sign_flip_samples,
            seed=seed,
        )

    nested_targets = canonical_targets[
        canonical_targets["user_id"].isin(common_users)
    ].copy()
    result = {
        "schema_version": 1,
        "analysis": "kuairand_k1_nested_fixed_candidate_support_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "analyzed" if analysis_performed else "feasibility_only",
        "design": {
            "kind": "post_hoc_nested_fixed_candidate_subgroup",
            "source_run": "original min-positive-degree=1 k1 task",
            "candidate_rows": (
                "exact unchanged target rows and saved predictions from k1"
            ),
            "filter": (
                "all candidates in both target-policy c3 must have positive "
                f"source-user degree >= {min_positive_degree} under both "
                "standard and random source histories"
            ),
            "cohort": (
                "user must pass both target policies for every saved "
                "candidate-resample seed"
            ),
            "uses_target_outcomes_for_filtering": False,
            "new_model_calls": 0,
            "new_candidate_rows_used": False,
            "source_history_recovery": (
                "The k1 task builder is rerun only to recover and hash-validate "
                "the exact saved source histories. Target rows reconstructed "
                "as a builder side effect are discarded."
            ),
            "not_equivalent_to_resampled_k2_task": True,
            "distinction_from_resampled_k2": (
                "The resampled k2 task first restricts the candidate pool and "
                "draws new c3 groups. This audit only removes complete "
                "user-seed groups from already evaluated k1 candidates."
            ),
            "post_run_not_preregistered": True,
        },
        "gate": {
            "min_analysis_users": min_analysis_users,
            "common_users": len(common_users),
            "analysis_performed": analysis_performed,
            "reason": (
                None
                if analysis_performed
                else "common cross-seed cohort below analysis gate"
            ),
        },
        "source_support": {
            "minimum_positive_source_users_per_policy": min_positive_degree,
            "degree_unit": "unique positive source users per item",
            "source_seed": 42,
            "history_hashes": history_hashes,
            "history_rows": {
                policy: int(len(history))
                for policy, history in histories.items()
            },
            "items_with_positive_support": {
                policy: int(len(degrees))
                for policy, degrees in positive_degrees.items()
            },
            "items_with_degree_at_least_threshold": {
                policy: int(degrees.ge(min_positive_degree).sum())
                for policy, degrees in positive_degrees.items()
            },
        },
        "input_validation": {
            **parity,
            "original_target_rows_sha256": _frame_sha256(canonical_targets),
            "nested_target_rows": int(len(nested_targets)),
            "nested_target_rows_sha256": _frame_sha256(nested_targets),
            "exact_saved_prediction_targets_used": True,
        },
        "feasibility": {
            "original_users": int(
                canonical_targets["user_id"].nunique()
            ),
            "original_user_seed_pairs": int(len(user_seed)),
            "passing_common_cross_seed_users": len(common_users),
            "common_user_ids_sha256": _values_sha256(common_users),
            "per_seed": per_seed,
            "per_target_policy_and_seed": per_policy_seed,
        },
        "statistical_protocol": (
            {
                "metric": metric,
                "bootstrap_samples": bootstrap_samples,
                "sign_flip_samples": sign_flip_samples,
                "random_seed": seed,
                "holm_family": (
                    "six post-run code-declared primary transfer comparisons"
                ),
            }
            if analysis_performed
            else None
        ),
        "inputs": {
            name: _input_provenance(artifact)
            for name, artifact in artifacts.items()
        },
        "comparisons": comparisons,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "nested_support_audit.json", result)
    candidate_audit.to_parquet(
        output_dir / "candidate_degree_audit.parquet",
        index=False,
    )
    user_seed.to_csv(
        output_dir / "user_seed_eligibility.csv",
        index=False,
    )
    pd.DataFrame({"user_id": common_users}).to_csv(
        output_dir / "common_cross_seed_users.csv",
        index=False,
    )
    if comparisons:
        _write_csv(
            output_dir / "nested_suite_comparisons.csv",
            comparisons,
        )
    (output_dir / "nested_support_report.md").write_text(
        render_nested_report(result),
        encoding="utf-8",
    )
    return result


def render_nested_report(result: Mapping[str, Any]) -> str:
    feasibility = result["feasibility"]
    gate = result["gate"]
    design = result["design"]
    lines = [
        "# KuaiRand nested fixed-candidate support audit",
        "",
        (
            "This is a post-hoc subgroup audit of the **existing k=1 target "
            "rows and predictions**. It does not rebuild candidates and is not "
            "the resampled k=2 task."
        ),
        "",
        (
            "A user-seed pair is retained only when all six unchanged "
            "candidates—both target-policy c3 groups—have at least two "
            "positive source users under each source policy. The final cohort "
            "must pass for every candidate seed."
        ),
        "",
        "## Feasibility",
        "",
        f"- Original users: `{feasibility['original_users']}`.",
        (
            "- Passing users by seed: "
            + ", ".join(
                f"`{row['task_seed']}: {row['passing_users']}`"
                for row in feasibility["per_seed"]
            )
            + "."
        ),
        (
            "- Common users passing all seeds: "
            f"`{feasibility['passing_common_cross_seed_users']}`."
        ),
        (
            f"- Analysis gate: `{gate['min_analysis_users']}` users; status: "
            f"`{result['status']}`."
        ),
        "",
        "Per target policy:",
        "",
        "| Seed | Target policy | Passing users | Total users |",
        "|---:|---|---:|---:|",
    ]
    for row in feasibility["per_target_policy_and_seed"]:
        lines.append(
            f"| {row['task_seed']} | {row['target_policy']} | "
            f"{row['passing_users']} | {row['total_users']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "- The fixed-candidate intersection is much stricter than "
                "resampling from a degree≥2 candidate pool: one user must keep "
                "all 18 original candidate rows across three seeds."
            ),
            (
                "- Therefore this audit can test whether the k=1 result "
                "survives on an unchanged sufficiently-supported subgroup, but "
                "cannot replace the resampled k=2 experiment."
            ),
            (
                "- Cohort filtering uses source support and candidate IDs, not "
                "target click labels."
            ),
            (
                "- The subgroup and analysis gate were defined after model "
                "outputs existed; this is diagnostic, not confirmatory."
            ),
            "",
            "## Input integrity",
            "",
            (
                "- Exact GPT–ItemCTR target parity: "
                f"`{result['input_validation']['exact_target_rows_match']}`."
            ),
            (
                "- Original target SHA-256: "
                f"`{result['input_validation']['original_target_rows_sha256']}`."
            ),
            (
                "- No new model calls or newly constructed candidate rows "
                "used: "
                f"`{design['new_model_calls'] == 0 and not design['new_candidate_rows_used']}`."
            ),
            "",
        ]
    )
    if result["comparisons"]:
        lines.extend(
            [
                "## Nested subgroup comparisons",
                "",
                _markdown_table(
                    [
                        row
                        for row in result["comparisons"]
                        if row["primary"]
                    ],
                    holm=True,
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Statistical analysis",
                "",
                (
                    "Not run: the common fixed-candidate cohort is below the "
                    "declared minimum analysis size. Only feasibility is "
                    "reported."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _validate_original_k1_protocol(
    artifacts: Mapping[str, MethodArtifact],
) -> None:
    for name, artifact in artifacts.items():
        for task_seed, manifest in artifact.manifest["task_manifests"].items():
            actual = int(manifest["min_positive_source_users_per_policy"])
            if actual != 1:
                raise ValueError(
                    f"{name} seed {task_seed} is not the original k1 task"
                )
            requested = int(
                manifest["runner_seed_interface"][
                    "requested_min_positive_source_users_per_policy"
                ]
            )
            if requested != 1:
                raise ValueError(
                    f"{name} seed {task_seed} did not request k1 support"
                )


def _validate_source_history_hashes(
    artifacts: Mapping[str, MethodArtifact],
    *,
    history_hashes: Mapping[str, str],
) -> None:
    for name, artifact in artifacts.items():
        for task_seed, manifest in artifact.manifest["task_manifests"].items():
            saved = manifest["hashes"]["frames"]
            for policy in POLICIES:
                expected = saved[f"{policy}_history"]
                if history_hashes[policy] != expected:
                    raise ValueError(
                        f"Rebuilt {policy} history does not match {name} "
                        f"seed {task_seed}"
                    )


def _filter_users(
    frame: pd.DataFrame,
    user_ids: Sequence[object],
) -> pd.DataFrame:
    filtered = frame[frame["user_id"].isin(user_ids)].copy()
    if filtered["user_id"].nunique() != len(user_ids):
        raise ValueError("Filtered per-user metrics do not cover the cohort")
    return filtered


def _per_seed_feasibility(
    user_seed: pd.DataFrame,
) -> list[dict[str, Any]]:
    records = []
    for task_seed, rows in user_seed.groupby("task_seed", sort=True):
        records.append(
            {
                "task_seed": int(task_seed),
                "total_users": int(rows["user_id"].nunique()),
                "passing_users": int(
                    rows["passes_both_target_policies"].sum()
                ),
                "common_cross_seed_users": int(
                    rows["in_common_cross_seed_cohort"].sum()
                ),
            }
        )
    return records


def _per_policy_seed_feasibility(
    candidate_audit: pd.DataFrame,
) -> list[dict[str, Any]]:
    groups = (
        candidate_audit.groupby(
            ["task_seed", "user_id", "target_policy"],
            sort=False,
        )["passes_degree_threshold"]
        .all()
        .rename("passes")
        .reset_index()
    )
    records = []
    for (task_seed, target_policy), rows in groups.groupby(
        ["task_seed", "target_policy"],
        sort=True,
    ):
        records.append(
            {
                "task_seed": int(task_seed),
                "target_policy": str(target_policy),
                "total_users": int(rows["user_id"].nunique()),
                "passing_users": int(rows["passes"].sum()),
            }
        )
    return records


def _values_sha256(values: Sequence[object]) -> str:
    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit degree>=2 nested support on unchanged KuaiRand k1 targets"
        )
    )
    parser.add_argument("--gpt-run-dir", type=Path, required=True)
    parser.add_argument("--classical-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-positive-degree",
        type=int,
        default=DEFAULT_MIN_POSITIVE_DEGREE,
    )
    parser.add_argument(
        "--min-analysis-users",
        type=int,
        default=DEFAULT_MIN_ANALYSIS_USERS,
    )
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--sign-flip-samples",
        type=int,
        default=DEFAULT_SIGN_FLIP_SAMPLES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_nested_support_audit(
        gpt_run_dir=args.gpt_run_dir,
        classical_run_dir=args.classical_run_dir,
        output_dir=args.output_dir,
        min_positive_degree=args.min_positive_degree,
        min_analysis_users=args.min_analysis_users,
        metric=args.metric,
        bootstrap_samples=args.bootstrap_samples,
        sign_flip_samples=args.sign_flip_samples,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "common_users": result["gate"]["common_users"],
                "analysis_performed": result["gate"]["analysis_performed"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
