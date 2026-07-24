from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks.kuairand_policy_shift import (
    POLICIES,
    SOURCE_END,
    SOURCE_START,
    TAB_VALUE,
    TEST_END,
    TEST_START,
    VAL_END,
    VAL_START,
    _stable_int,
)
from beyond_click_sim.tasks.kuairand_natural_exposure import (
    DEFAULT_CANDIDATE_SIZE,
    DEFAULT_HISTORY_SIZE,
    DEFAULT_MIN_EXPOSURE_USERS_PER_POLICY,
    DEFAULT_MIN_TRAINING_EXPOSURE_USERS_PER_POLICY,
)
from runners.ood.kuairand_policy_shift.task_builders import (
    load_canonical_dataset,
)


DEFAULT_HISTORY_SIZES = (3, 5, 8, 10)
DEFAULT_CANDIDATE_SIZES = (3, 5, 8, 10)
DEFAULT_EXPOSURE_SUPPORTS = (1, 2, 5, 8, 10, 20)
DEFAULT_CANDIDATE_SEEDS = (0, 1, 2)

# Locked after inspecting the no-inference feasibility grid. These values are
# design choices, not parameters selected using model scores.
LOCKED_HISTORY_SIZE = DEFAULT_HISTORY_SIZE
LOCKED_CANDIDATE_SIZE = DEFAULT_CANDIDATE_SIZE
LOCKED_MIN_EXPOSURE_USERS_PER_POLICY = (
    DEFAULT_MIN_EXPOSURE_USERS_PER_POLICY
)
LOCKED_CANDIDATE_SEEDS = DEFAULT_CANDIDATE_SEEDS

SPLITS = {
    "val": (VAL_START, VAL_END),
    "test": (TEST_START, TEST_END),
}
REQUIRED_INTERACTION_COLUMNS = (
    "interaction_id",
    "user_id",
    "item_id",
    "exposure_policy",
    "tab",
    "date",
    "timestamp_ms",
    "target_click",
)
REQUIRED_ITEM_COLUMNS = (
    "item_id",
    "description",
    "category_path",
)


def prepare_semantic_items(items: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return items with both released caption text and category metadata."""

    _require_columns(items, REQUIRED_ITEM_COLUMNS, table="items")
    description = items["description"].astype("string").str.strip()
    category = items["category_path"].astype("string").str.strip()
    eligible = description.notna() & description.ne("")
    eligible &= category.notna() & category.ne("")
    semantic = items.loc[eligible, list(REQUIRED_ITEM_COLUMNS)].copy()
    return semantic, {
        "items_total": int(len(items)),
        "items_with_description": int(
            (description.notna() & description.ne("")).sum()
        ),
        "items_with_category_path": int(
            (category.notna() & category.ne("")).sum()
        ),
        "items_with_both": int(eligible.sum()),
        "items_dropped": int((~eligible).sum()),
        "description_length_chars": _numeric_summary(
            description.loc[eligible].str.len()
        ),
        "category_path_length_chars": _numeric_summary(
            category.loc[eligible].str.len()
        ),
    }


def prepare_interactions(
    interactions: pd.DataFrame,
    *,
    semantic_item_ids: set[object],
) -> pd.DataFrame:
    """Normalize semantically covered rows without consulting response labels."""

    _require_columns(
        interactions,
        REQUIRED_INTERACTION_COLUMNS,
        table="interactions",
    )
    rows = interactions.loc[
        interactions["item_id"].isin(semantic_item_ids)
        & interactions["exposure_policy"].isin(POLICIES),
        list(REQUIRED_INTERACTION_COLUMNS),
    ].copy()
    numeric_click = pd.to_numeric(rows["target_click"], errors="coerce")
    if numeric_click.isna().any() or not set(numeric_click.unique()).issubset(
        {0, 1}
    ):
        raise ValueError("target_click must contain only binary 0/1 values")
    rows["target_click"] = numeric_click.astype("int8")
    rows["date"] = pd.to_numeric(rows["date"], errors="raise").astype(int)
    rows["timestamp_ms"] = pd.to_numeric(
        rows["timestamp_ms"], errors="raise"
    ).astype("int64")
    return rows.sort_values(
        ["timestamp_ms", "interaction_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def prepare_tab1_interactions(
    interactions: pd.DataFrame,
    *,
    semantic_item_ids: set[object],
) -> pd.DataFrame:
    """Normalize semantically covered interactions and retain tab 1."""

    prepared = prepare_interactions(
        interactions,
        semantic_item_ids=semantic_item_ids,
    )
    return prepared.loc[prepared["tab"].eq(TAB_VALUE)].reset_index(drop=True)


def first_policy_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep the chronologically first row of each user-item-policy triple."""

    return (
        rows.sort_values(
            ["timestamp_ms", "interaction_id"],
            kind="mergesort",
        )
        .drop_duplicates(
            ["user_id", "item_id", "exposure_policy"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def build_source_pairs(tab1: pd.DataFrame) -> pd.DataFrame:
    """Build the full label-blind source graph used for support audits."""

    source = tab1.loc[
        tab1["date"].between(SOURCE_START, SOURCE_END)
    ].copy()
    return first_policy_pairs(source)


def source_item_degrees(
    source: pd.DataFrame,
) -> dict[str, dict[str, pd.Series]]:
    """Return exposure and positive-user degree for each source policy."""

    degrees: dict[str, dict[str, pd.Series]] = {}
    for policy in POLICIES:
        policy_rows = source.loc[source["exposure_policy"].eq(policy)]
        exposure = policy_rows.groupby("item_id", sort=False)["user_id"].nunique()
        positive = (
            policy_rows.loc[policy_rows["target_click"].eq(1)]
            .groupby("item_id", sort=False)["user_id"]
            .nunique()
        )
        degrees[policy] = {
            "exposure": exposure,
            "positive": positive,
        }
    return degrees


def common_exposure_supported_items(
    degrees: dict[str, dict[str, pd.Series]],
    *,
    min_exposure_users_per_policy: int,
) -> set[object]:
    """Select source-warm items without using source click outcomes."""

    if min_exposure_users_per_policy <= 0:
        raise ValueError("min_exposure_users_per_policy must be positive")
    supported = {
        policy: set(
            degrees[policy]["exposure"]
            .loc[
                degrees[policy]["exposure"].ge(
                    min_exposure_users_per_policy
                )
            ]
            .index
        )
        for policy in POLICIES
    }
    return supported["standard"] & supported["random"]


def build_pair_novel_target_pool(
    tab1: pd.DataFrame,
    *,
    all_interactions: pd.DataFrame | None = None,
    split_start: int,
    split_end: int,
    allowed_items: set[object],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a natural target pool using no click-dependent row filters."""

    prior_rows = tab1 if all_interactions is None else all_interactions
    prior_pairs = (
        prior_rows.loc[
            prior_rows["date"].lt(split_start),
            ["user_id", "item_id"],
        ]
        .drop_duplicates()
        .assign(_seen_before_target=True)
    )
    target = first_policy_pairs(
        tab1.loc[tab1["date"].between(split_start, split_end)].copy()
    )
    before_item_support = len(target)
    target = target.loc[target["item_id"].isin(allowed_items)].copy()
    after_item_support = len(target)
    target = target.merge(
        prior_pairs,
        on=["user_id", "item_id"],
        how="left",
        validate="many_to_one",
    )
    pair_novel = target.loc[target["_seen_before_target"].isna()].drop(
        columns="_seen_before_target"
    )
    duplicate_pair_policy = pair_novel.duplicated(
        ["user_id", "item_id", "exposure_policy"]
    )
    if duplicate_pair_policy.any():
        raise RuntimeError("Pair-novel target pool contains duplicate policy pairs")
    return pair_novel.reset_index(drop=True), {
        "rows_before_item_support": int(before_item_support),
        "rows_after_item_support": int(after_item_support),
        "rows_after_pair_novelty": int(len(pair_novel)),
        "pair_novelty_violations": 0,
    }


def label_blind_cohort(
    source: pd.DataFrame,
    target_pool: pd.DataFrame,
    *,
    history_size: int,
    candidate_size: int,
) -> list[object]:
    """Select users using only source/target row counts under both policies."""

    if history_size <= 0 or candidate_size <= 0:
        raise ValueError("history_size and candidate_size must be positive")
    source_counts = _policy_counts(source)
    target_counts = _policy_counts(target_pool)
    counts = source_counts.join(
        target_counts,
        how="inner",
        lsuffix="_source",
        rsuffix="_target",
    )
    eligible = counts.loc[
        counts["standard_source"].ge(history_size)
        & counts["random_source"].ge(history_size)
        & counts["standard_target"].ge(candidate_size)
        & counts["random_target"].ge(candidate_size)
    ]
    return sorted(eligible.index.tolist(), key=repr)


def select_recent_histories(
    source: pd.DataFrame,
    *,
    users: list[object],
    history_size: int,
) -> dict[str, pd.DataFrame]:
    """Select each target user's latest H source pairs, without using labels."""

    histories: dict[str, pd.DataFrame] = {}
    for policy in POLICIES:
        policy_rows = source.loc[
            source["exposure_policy"].eq(policy)
            & source["user_id"].isin(users)
        ].copy()
        selected = (
            policy_rows.sort_values(
                ["user_id", "timestamp_ms", "interaction_id"],
                kind="mergesort",
            )
            .groupby("user_id", sort=False)
            .tail(history_size)
            .sort_values(
                ["user_id", "timestamp_ms", "interaction_id"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        counts = selected.groupby("user_id").size()
        if len(counts) != len(users) or not counts.eq(history_size).all():
            raise RuntimeError(
                f"{policy} histories do not contain exactly H rows per user"
            )
        histories[policy] = selected
    return histories


def select_label_blind_candidates(
    target_pool: pd.DataFrame,
    *,
    users: list[object],
    candidate_size: int,
    seed: int,
    split: str,
) -> dict[str, pd.DataFrame]:
    """Select exactly m candidates per user/policy without reading labels."""

    selected_by_policy: dict[str, pd.DataFrame] = {}
    for policy in POLICIES:
        rows = target_pool.loc[
            target_pool["exposure_policy"].eq(policy)
            & target_pool["user_id"].isin(users)
        ].copy()
        rows["_selection_key"] = [
            _stable_int(
                seed,
                "natural-target",
                split,
                policy,
                user_id,
                item_id,
                interaction_id,
            )
            for user_id, item_id, interaction_id in zip(
                rows["user_id"],
                rows["item_id"],
                rows["interaction_id"],
                strict=True,
            )
        ]
        selected = (
            rows.sort_values(
                [
                    "user_id",
                    "_selection_key",
                    "timestamp_ms",
                    "interaction_id",
                ],
                kind="mergesort",
            )
            .groupby("user_id", sort=False)
            .head(candidate_size)
            .sort_values(
                ["user_id", "_selection_key", "interaction_id"],
                kind="mergesort",
            )
            .drop(columns="_selection_key")
            .reset_index(drop=True)
        )
        counts = selected.groupby("user_id").size()
        if len(counts) != len(users) or not counts.eq(candidate_size).all():
            raise RuntimeError(
                f"{policy} targets do not contain exactly m rows per user"
            )
        selected_by_policy[policy] = selected
    return selected_by_policy


def feasibility_grid(
    source: pd.DataFrame,
    target_pool: pd.DataFrame,
    *,
    history_sizes: tuple[int, ...] = DEFAULT_HISTORY_SIZES,
    candidate_sizes: tuple[int, ...] = DEFAULT_CANDIDATE_SIZES,
) -> list[dict[str, int]]:
    """Count the outcome-blind cohort for each H × m combination."""

    rows: list[dict[str, int]] = []
    for history_size in history_sizes:
        for candidate_size in candidate_sizes:
            users = label_blind_cohort(
                source,
                target_pool,
                history_size=history_size,
                candidate_size=candidate_size,
            )
            rows.append(
                {
                    "history_size": history_size,
                    "candidate_size": candidate_size,
                    "users": len(users),
                }
            )
    return rows


def run_audit(
    *,
    dataset: CanonicalDataset,
    output_dir: Path,
    history_sizes: tuple[int, ...] = DEFAULT_HISTORY_SIZES,
    candidate_sizes: tuple[int, ...] = DEFAULT_CANDIDATE_SIZES,
    exposure_supports: tuple[int, ...] = DEFAULT_EXPOSURE_SUPPORTS,
    candidate_seeds: tuple[int, ...] = DEFAULT_CANDIDATE_SEEDS,
) -> dict[str, Any]:
    """Run and persist the no-inference natural-exposure feasibility audit."""

    items = dataset.load_items(columns=list(REQUIRED_ITEM_COLUMNS))
    interactions = dataset.load_interactions(
        columns=list(REQUIRED_INTERACTION_COLUMNS)
    )
    semantic_items, semantic_audit = prepare_semantic_items(items)
    prepared_interactions = prepare_interactions(
        interactions,
        semantic_item_ids=set(semantic_items["item_id"]),
    )
    tab1 = prepared_interactions.loc[
        prepared_interactions["tab"].eq(TAB_VALUE)
    ].reset_index(drop=True)
    source = build_source_pairs(tab1)
    degrees = source_item_degrees(source)
    source_counts = _policy_counts(source)

    support_results: list[dict[str, Any]] = []
    locked_details: dict[str, Any] = {}
    for min_support in exposure_supports:
        allowed_items = common_exposure_supported_items(
            degrees,
            min_exposure_users_per_policy=min_support,
        )
        split_results: dict[str, Any] = {}
        for split, (start, end) in SPLITS.items():
            target_pool, pool_audit = build_pair_novel_target_pool(
                tab1,
                all_interactions=prepared_interactions,
                split_start=start,
                split_end=end,
                allowed_items=allowed_items,
            )
            target_counts = _policy_counts(target_pool)
            grid = feasibility_grid(
                source,
                target_pool,
                history_sizes=history_sizes,
                candidate_sizes=candidate_sizes,
            )
            split_result: dict[str, Any] = {
                "window": [_date_iso(start), _date_iso(end)],
                "target_pool": {
                    **pool_audit,
                    "users_with_rows_under_both_policies": int(
                        (
                            target_counts["standard"].gt(0)
                            & target_counts["random"].gt(0)
                        ).sum()
                    ),
                    "policies": {
                        policy: _response_summary(
                            target_pool.loc[
                                target_pool["exposure_policy"].eq(policy)
                            ]
                        )
                        for policy in POLICIES
                    },
                },
                "grid": grid,
            }
            if min_support == LOCKED_MIN_EXPOSURE_USERS_PER_POLICY:
                users = label_blind_cohort(
                    source,
                    target_pool,
                    history_size=LOCKED_HISTORY_SIZE,
                    candidate_size=LOCKED_CANDIDATE_SIZE,
                )
                histories = select_recent_histories(
                    source,
                    users=users,
                    history_size=LOCKED_HISTORY_SIZE,
                )
                selected_seeds: dict[str, Any] = {}
                for seed in candidate_seeds:
                    selected = select_label_blind_candidates(
                        target_pool,
                        users=users,
                        candidate_size=LOCKED_CANDIDATE_SIZE,
                        seed=seed,
                        split=split,
                    )
                    selected_seeds[str(seed)] = {
                        policy: _selected_target_summary(
                            selected[policy],
                            degrees=degrees,
                            candidate_size=LOCKED_CANDIDATE_SIZE,
                        )
                        for policy in POLICIES
                    }
                locked_details[split] = {
                    "users": len(users),
                    "history": {
                        policy: _response_summary(histories[policy])
                        for policy in POLICIES
                    },
                    "candidate_seeds": selected_seeds,
                }
                split_result["locked_protocol"] = locked_details[split]
            split_results[split] = split_result
        support_results.append(
            {
                "min_exposure_users_per_policy": min_support,
                "common_supported_items": len(allowed_items),
                "splits": split_results,
            }
        )

    result = {
        "schema_version": 1,
        "analysis": "kuairand_natural_exposure_feasibility",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            "root": str(dataset.root),
            "canonical_manifest": dataset.load_manifest(),
        },
        "selection_guarantees": {
            "history_user_filter_uses_labels": False,
            "target_user_filter_uses_labels": False,
            "item_support_filter_uses_labels": False,
            "candidate_selection_uses_labels": False,
            "pair_novelty": (
                "target user-item pair absent from all earlier tabs under "
                "standard and random exposure policies"
            ),
            "history_selection": (
                "latest H source-window unique user-item-policy pairs"
            ),
            "candidate_selection": (
                "seeded stable-hash sample of m pair-novel target-window "
                "exposures, independent of target_click"
            ),
        },
        "semantic_coverage": semantic_audit,
        "tab1_rows_with_full_semantics": len(tab1),
        "source": {
            "window": [_date_iso(SOURCE_START), _date_iso(SOURCE_END)],
            "first_user_item_policy_pairs": len(source),
            "policies": {
                policy: {
                    **_response_summary(
                        source.loc[source["exposure_policy"].eq(policy)]
                    ),
                    "per_user_pair_count": _numeric_summary(
                        source_counts[policy].loc[source_counts[policy].gt(0)]
                    ),
                    "item_exposure_user_degree": _numeric_summary(
                        degrees[policy]["exposure"]
                    ),
                    "item_positive_user_degree": _numeric_summary(
                        degrees[policy]["positive"].reindex(
                            degrees[policy]["exposure"].index,
                            fill_value=0,
                        )
                    ),
                }
                for policy in POLICIES
            },
            "users_with_at_least_h_rows_under_both_policies": {
                str(history_size): int(
                    (
                        source_counts["standard"].ge(history_size)
                        & source_counts["random"].ge(history_size)
                    ).sum()
                )
                for history_size in history_sizes
            },
        },
        "support_scenarios": support_results,
        "locked_protocol": {
            "history_size": LOCKED_HISTORY_SIZE,
            "candidate_size": LOCKED_CANDIDATE_SIZE,
            "min_source_exposure_users_per_item_per_policy": (
                LOCKED_MIN_EXPOSURE_USERS_PER_POLICY
            ),
            "min_training_exposure_users_per_item_per_policy": (
                DEFAULT_MIN_TRAINING_EXPOSURE_USERS_PER_POLICY
            ),
            "candidate_seeds": list(candidate_seeds),
            "history_seed": None,
            "full_square": [
                "standard_history_to_standard_target",
                "random_history_to_standard_target",
                "standard_history_to_random_target",
                "random_history_to_random_target",
            ],
            "selection_rationale": {
                "history_size": (
                    "H=5 supplies five feedback examples from each source "
                    "policy while retaining substantially more users than H=10"
                ),
                "candidate_size": (
                    "m=5 retains enough users and ranking-eligible random "
                    "groups; m=8 sharply reduces the validation cohort"
                ),
                "item_support": (
                    "eight users in each full source-policy log preserves at "
                    "least five users after evaluation-user histories are "
                    "restricted to H=5, without filtering on source clicks"
                ),
            },
            "splits": locked_details,
        },
        "metric_notes": {
            "primary_ranking": (
                "NDCG on groups containing both click labels; report exact "
                "mixed-label coverage and do not drop users from construction"
            ),
            "secondary_ranking": "within-group ROC AUC on the same mixed groups",
            "all_natural_rows": (
                "retain all-zero and all-one groups for probability metrics "
                "once a probability-producing scorer is defined"
            ),
            "absolute_policy_columns": (
                "do not compare raw standard-target and random-target levels; "
                "compare matched/transferred histories within a target column"
            ),
        },
    }

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "feasibility_audit.json", result)
    (output_dir / "feasibility_report.md").write_text(
        render_report(result),
        encoding="utf-8",
    )
    return result


def render_report(result: dict[str, Any]) -> str:
    """Render the audit into a compact human-readable protocol report."""

    lines = [
        "# KuaiRand natural-exposure feasibility audit",
        "",
        "No model or API outputs are used in this audit. User, item, and target "
        "selection are label-blind; click labels are read only for descriptive "
        "prevalence and metric-coverage diagnostics.",
        "",
        "## Locked protocol",
        "",
        "| Parameter | Value |",
        "| --- | ---: |",
        f"| History length per policy (H) | {result['locked_protocol']['history_size']} |",
        f"| Natural candidates per user/policy (m) | {result['locked_protocol']['candidate_size']} |",
        "| Minimum source exposure users per item/policy | "
        f"{result['locked_protocol']['min_source_exposure_users_per_item_per_policy']} |",
        "| Minimum users in actual training graph per item/policy | "
        f"{result['locked_protocol']['min_training_exposure_users_per_item_per_policy']} |",
        "| Candidate seeds | "
        f"{', '.join(map(str, result['locked_protocol']['candidate_seeds']))} |",
        "",
        "The four cells are standard→standard, random→standard, "
        "standard→random, and random→random. Within each target-policy column, "
        "the matched and transferred cells use byte-identical target rows.",
        "",
        "## Feasibility grid",
        "",
    ]
    for scenario in result["support_scenarios"]:
        support = scenario["min_exposure_users_per_policy"]
        lines.extend(
            [
                f"### Minimum exposure support = {support}",
                "",
                f"Common supported items: {scenario['common_supported_items']:,}.",
                "",
            ]
        )
        for split in ("val", "test"):
            split_result = scenario["splits"][split]
            lines.extend(
                [
                    f"#### {split}",
                    "",
                    "| H | m=3 | m=5 | m=8 | m=10 |",
                    "| ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            grid = {
                (row["history_size"], row["candidate_size"]): row["users"]
                for row in split_result["grid"]
            }
            for history_size in DEFAULT_HISTORY_SIZES:
                values = [
                    f"{grid.get((history_size, candidate_size), 0):,}"
                    for candidate_size in DEFAULT_CANDIDATE_SIZES
                ]
                lines.append(
                    f"| {history_size} | " + " | ".join(values) + " |"
                )
            lines.append("")

    lines.extend(["## Locked cohort diagnostics", ""])
    locked = result["locked_protocol"]["splits"]
    for split in ("val", "test"):
        details = locked[split]
        lines.extend(
            [
                f"### {split}: {details['users']:,} users",
                "",
                "| Policy | History CTR | Seed | Target CTR | Mixed groups | All-zero groups |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for policy in POLICIES:
            history_ctr = details["history"][policy]["click_rate"]
            for seed, seed_details in details["candidate_seeds"].items():
                target = seed_details[policy]
                lines.append(
                    f"| {policy} | {history_ctr:.4f} | {seed} | "
                    f"{target['click_rate']:.4f} | "
                    f"{target['mixed_label_group_rate']:.4f} | "
                    f"{target['all_zero_group_rate']:.4f} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Standard and random targets have different natural exposure and "
            "base-rate distributions; raw metric levels across columns are not "
            "a transfer contrast.",
            "- Multiple candidate seeds are correlated resamples of the same "
            "users; available metric-eligible resamples must be averaged within "
            "user before user-level inference.",
            "- All-zero/all-one groups remain in the constructed task. They are "
            "excluded only where a ranking metric is mathematically uninformative.",
            "- Calibration requires probability-producing scores; listwise ranks "
            "alone are not calibrated probabilities.",
            "",
        ]
    )
    return "\n".join(lines)


def _policy_counts(rows: pd.DataFrame) -> pd.DataFrame:
    counts = (
        rows.groupby(["user_id", "exposure_policy"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(POLICIES), fill_value=0)
    )
    return counts.astype(int)


def _response_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "rows": 0,
            "users": 0,
            "items": 0,
            "clicks": 0,
            "click_rate": 0.0,
        }
    return {
        "rows": int(len(rows)),
        "users": int(rows["user_id"].nunique()),
        "items": int(rows["item_id"].nunique()),
        "clicks": int(rows["target_click"].sum()),
        "click_rate": float(rows["target_click"].mean()),
    }


def _selected_target_summary(
    rows: pd.DataFrame,
    *,
    degrees: dict[str, dict[str, pd.Series]],
    candidate_size: int,
) -> dict[str, Any]:
    summary = _response_summary(rows)
    positives = rows.groupby("user_id", sort=False)["target_click"].sum()
    min_exposure_degree = pd.concat(
        [
            rows["item_id"].map(degrees[policy]["exposure"]).fillna(0)
            for policy in POLICIES
        ],
        axis=1,
    ).min(axis=1)
    min_positive_degree = pd.concat(
        [
            rows["item_id"].map(degrees[policy]["positive"]).fillna(0)
            for policy in POLICIES
        ],
        axis=1,
    ).min(axis=1)
    summary.update(
        {
            "groups": int(len(positives)),
            "positive_labels_per_group": _numeric_summary(positives),
            "mixed_label_groups": int(
                (positives.gt(0) & positives.lt(candidate_size)).sum()
            ),
            "mixed_label_group_rate": float(
                (positives.gt(0) & positives.lt(candidate_size)).mean()
            ),
            "all_zero_groups": int(positives.eq(0).sum()),
            "all_zero_group_rate": float(positives.eq(0).mean()),
            "all_one_groups": int(positives.eq(candidate_size).sum()),
            "all_one_group_rate": float(
                positives.eq(candidate_size).mean()
            ),
            "minimum_source_exposure_user_degree": _numeric_summary(
                min_exposure_degree
            ),
            "minimum_source_positive_user_degree": _numeric_summary(
                min_positive_degree
            ),
            "source_positive_support_bins": {
                "zero": int(min_positive_degree.eq(0).sum()),
                "one": int(min_positive_degree.eq(1).sum()),
                "two_to_four": int(
                    min_positive_degree.between(2, 4).sum()
                ),
                "five_or_more": int(min_positive_degree.ge(5).sum()),
            },
        }
    )
    return summary


def _numeric_summary(values: pd.Series) -> dict[str, int | float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0,
        }
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "p95": float(numeric.quantile(0.95)),
        "max": float(numeric.max()),
    }


def _date_iso(value: int) -> str:
    text = f"{int(value):08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    table: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing columns: {missing}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return values


def _parse_seed_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integer seeds"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit label-blind natural KuaiRand target construction without "
            "calling an LLM or fitting a model."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Canonical KuaiRand root; defaults to the shared workspace data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--history-sizes",
        type=_parse_int_tuple,
        default=DEFAULT_HISTORY_SIZES,
    )
    parser.add_argument(
        "--candidate-sizes",
        type=_parse_int_tuple,
        default=DEFAULT_CANDIDATE_SIZES,
    )
    parser.add_argument(
        "--exposure-supports",
        type=_parse_int_tuple,
        default=DEFAULT_EXPOSURE_SUPPORTS,
    )
    parser.add_argument(
        "--candidate-seeds",
        type=_parse_seed_tuple,
        default=DEFAULT_CANDIDATE_SEEDS,
    )
    args = parser.parse_args()
    dataset = load_canonical_dataset(args.dataset_root)
    run_audit(
        dataset=dataset,
        output_dir=args.output_dir,
        history_sizes=args.history_sizes,
        candidate_sizes=args.candidate_sizes,
        exposure_supports=args.exposure_supports,
        candidate_seeds=args.candidate_seeds,
    )


if __name__ == "__main__":
    main()
