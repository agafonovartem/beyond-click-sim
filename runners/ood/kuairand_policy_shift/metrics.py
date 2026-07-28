from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
import pandas as pd

from beyond_click_sim.evaluation import grouped_ranking_metrics


RANKING_KS = (1, 5, 10)
REPORTED_METRICS = (
    "ndcg@5",
    "hit_rate@1",
    "ndcg@10",
    "hit_rate@5",
)
TIE_POLICY = "average"

CELL_COLUMNS = (
    "method",
    "target_pool",
    "target_policy",
    "history_control",
)
GROUP_COLUMNS = (
    "method",
    "task_seed",
    "split",
    "target_pool",
    "target_policy",
    "history_control",
    "user_id",
    "candidate_group",
)


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Evaluate mixed-label candidate groups and report degenerate coverage.

    All-zero and all-positive groups remain in the returned per-group frame,
    but their ranking metrics are missing because any ranking is respectively
    undefined or perfect. Headline cell means first average candidate seeds
    within user and then average users.
    """

    _require_columns(
        predictions,
        [*GROUP_COLUMNS, "item_id", "target", "score"],
    )
    per_group = _per_group_metrics(predictions)
    cells = _aggregate_cells(per_group)
    return cells, per_group


def build_paired_comparisons(
    per_group: pd.DataFrame,
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
) -> list[dict[str, object]]:
    """Compare the two predeclared GPT contrasts on identical target groups."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    _require_columns(
        per_group,
        [
            *GROUP_COLUMNS,
            "ranking_eligible",
            "failed",
            *REPORTED_METRICS,
        ],
    )

    comparisons: list[dict[str, object]] = []
    for target_pool in sorted(per_group["target_pool"].unique()):
        for target_policy in ("standard", "random"):
            scoped = per_group.loc[
                per_group["target_pool"].eq(target_pool)
                & per_group["target_policy"].eq(target_policy)
            ]
            contrasts = (
                (
                    "gpt_history_minus_no_history",
                    _select_method(
                        scoped,
                        method=(
                            "llm_openai_vk_gpt55_listwise_english"
                        ),
                        history_control="observed",
                    ),
                    _select_method(
                        scoped,
                        method=(
                            "llm_openai_vk_gpt55_listwise_english"
                        ),
                        history_control="none",
                    ),
                ),
                (
                    "gpt_history_minus_popularity",
                    _select_method(
                        scoped,
                        method=(
                            "llm_openai_vk_gpt55_listwise_english"
                        ),
                        history_control="observed",
                    ),
                    _select_method(
                        scoped,
                        method="popularity",
                        history_control="observed",
                    ),
                ),
            )
            for name, left, right in contrasts:
                if left.empty or right.empty:
                    continue
                for metric in REPORTED_METRICS:
                    comparisons.append(
                        {
                            "comparison": name,
                            "target_pool": target_pool,
                            "target_policy": target_policy,
                            "metric": metric,
                            "higher_is_better": True,
                            **paired_user_comparison(
                                left,
                                right,
                                metric=metric,
                                bootstrap_samples=bootstrap_samples,
                                bootstrap_seed=bootstrap_seed,
                            ),
                        }
                    )
    return comparisons


def paired_user_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    metric: str,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Return a paired delta after averaging candidate seeds within user."""

    if metric not in REPORTED_METRICS:
        raise ValueError(f"Unsupported paired metric: {metric!r}")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    pair_keys = [
        "task_seed",
        "split",
        "target_pool",
        "target_policy",
        "user_id",
        "candidate_group",
    ]
    left_values = left.loc[
        left["ranking_eligible"] & ~left["failed"],
        [*pair_keys, metric],
    ].rename(columns={metric: "left"})
    right_values = right.loc[
        right["ranking_eligible"] & ~right["failed"],
        [*pair_keys, metric],
    ].rename(columns={metric: "right"})
    paired = left_values.merge(
        right_values,
        on=pair_keys,
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["left", "right"])
    if paired.empty:
        return _empty_comparison()

    paired["delta"] = paired["left"] - paired["right"]
    user_delta = paired.groupby("user_id", sort=False)["delta"].mean()
    seed_counts = paired.groupby("user_id", sort=False)["task_seed"].nunique()
    values = user_delta.to_numpy(dtype=float)
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.choice(
        values,
        size=(bootstrap_samples, len(values)),
        replace=True,
    ).mean(axis=1)
    return {
        "n_users": int(len(values)),
        "n_seed_user_pairs": int(len(paired)),
        "eligible_seeds_per_user_min": int(seed_counts.min()),
        "eligible_seeds_per_user_mean": float(seed_counts.mean()),
        "eligible_seeds_per_user_max": int(seed_counts.max()),
        "mean_delta": float(values.mean()),
        "median_user_delta": float(np.median(values)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_unit": "user_after_mean_over_candidate_seeds",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _per_group_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = predictions.groupby(list(GROUP_COLUMNS), sort=False, dropna=False)
    for key, group in grouped:
        identity = dict(zip(GROUP_COLUMNS, key, strict=True))
        positives = int(group["target"].sum())
        size = int(len(group))
        if positives == 0:
            composition = "all_zero"
        elif positives == size:
            composition = "all_positive"
        else:
            composition = "mixed"
        eligible = composition == "mixed"
        failed = bool(group["score"].isna().any())
        values: dict[str, float] = {
            metric: math.nan for metric in REPORTED_METRICS
        }
        has_ties = False
        if eligible and not failed:
            metrics = grouped_ranking_metrics(
                group["target"],
                group["score"],
                group["candidate_group"],
                ks=RANKING_KS,
                tie_policy=TIE_POLICY,
            )
            values = {
                metric: float(metrics[metric])
                for metric in REPORTED_METRICS
            }
            has_ties = bool(metrics["groups_with_score_ties"])

        records.append(
            {
                **identity,
                "candidate_rows": size,
                "positive_labels": positives,
                "negative_labels": size - positives,
                "label_composition": composition,
                "ranking_eligible": eligible,
                "failed": failed,
                "has_score_ties": has_ties,
                **values,
                **{
                    f"failure_as_zero_{metric}": (
                        0.0 if eligible and failed else values[metric]
                    )
                    for metric in REPORTED_METRICS
                },
            }
        )
    return pd.DataFrame.from_records(records)


def _aggregate_cells(
    per_group: pd.DataFrame,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for key, cell in per_group.groupby(
        list(CELL_COLUMNS),
        sort=False,
        dropna=False,
    ):
        identity = dict(zip(CELL_COLUMNS, key, strict=True))
        eligible = cell["ranking_eligible"]
        successful = eligible & ~cell["failed"]
        cohort_users = pd.Index(cell["user_id"].drop_duplicates())
        eligible_seed_counts = _seed_counts_per_user(
            cell.loc[eligible],
            cohort_users=cohort_users,
        )
        successful_seed_counts = _seed_counts_per_user(
            cell.loc[successful],
            cohort_users=cohort_users,
        )
        cell_record: dict[str, object] = {
            **identity,
            "n_users_cohort": int(len(cohort_users)),
            "n_users_with_mixed_group": int(eligible_seed_counts.gt(0).sum()),
            "n_users_in_metric": int(successful_seed_counts.gt(0).sum()),
            "n_users_in_failure_as_zero_metric": int(
                eligible_seed_counts.gt(0).sum()
            ),
            "users_by_eligible_mixed_seed_count": _count_distribution(
                eligible_seed_counts
            ),
            "users_by_successful_mixed_seed_count": _count_distribution(
                successful_seed_counts
            ),
            "n_groups": int(len(cell)),
            "n_all_zero_groups": int(
                cell["label_composition"].eq("all_zero").sum()
            ),
            "n_all_positive_groups": int(
                cell["label_composition"].eq("all_positive").sum()
            ),
            "n_mixed_groups": int(eligible.sum()),
            "mixed_group_fraction": float(eligible.mean()),
            "n_successful_mixed_groups": int(successful.sum()),
            "n_failed_mixed_groups": int((eligible & cell["failed"]).sum()),
            "mixed_group_score_coverage": (
                float(successful.sum() / eligible.sum())
                if eligible.any()
                else math.nan
            ),
            "n_mixed_groups_with_score_ties": int(
                cell.loc[successful, "has_score_ties"].sum()
            ),
            "tie_policy": TIE_POLICY,
            "aggregation": "mean_seed_metric_per_user_then_mean_users",
        }
        for metric in REPORTED_METRICS:
            cell_record[metric] = _user_then_cell_mean(
                cell.loc[successful],
                metric,
            )
            cell_record[f"failure_as_zero_{metric}"] = _user_then_cell_mean(
                cell.loc[eligible],
                f"failure_as_zero_{metric}",
            )
        records.append(cell_record)
    return records


def _seed_counts_per_user(
    rows: pd.DataFrame,
    *,
    cohort_users: pd.Index,
) -> pd.Series:
    """Count distinct candidate seeds per user, including cohort zeros."""

    counts = rows.groupby("user_id", sort=False)["task_seed"].nunique()
    return counts.reindex(cohort_users, fill_value=0).astype(int)


def _count_distribution(counts: pd.Series) -> dict[str, int]:
    """Map a seed count to the number of cohort users with that count."""

    frequencies = counts.value_counts().sort_index()
    return {
        str(int(seed_count)): int(n_users)
        for seed_count, n_users in frequencies.items()
    }


def _user_then_cell_mean(rows: pd.DataFrame, metric: str) -> float:
    if rows.empty:
        return math.nan
    user_values = rows.groupby("user_id", sort=False)[metric].mean()
    return float(user_values.mean())


def _select_method(
    rows: pd.DataFrame,
    *,
    method: str,
    history_control: str,
) -> pd.DataFrame:
    return rows.loc[
        rows["method"].eq(method)
        & rows["history_control"].eq(history_control)
    ]


def _empty_comparison() -> dict[str, object]:
    return {
        "n_users": 0,
        "n_seed_user_pairs": 0,
        "eligible_seeds_per_user_min": 0,
        "eligible_seeds_per_user_mean": math.nan,
        "eligible_seeds_per_user_max": 0,
        "mean_delta": math.nan,
        "median_user_delta": math.nan,
        "ci95_low": math.nan,
        "ci95_high": math.nan,
        "bootstrap_unit": "user_after_mean_over_candidate_seeds",
    }


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
