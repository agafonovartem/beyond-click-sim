from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from beyond_click_sim.evaluation import grouped_ranking_metrics


DEFAULT_RANKING_K = 3
TIE_POLICY = "average"


def evaluate_prediction_cells(
    predictions: pd.DataFrame,
    *,
    ranking_k: int = DEFAULT_RANKING_K,
    require_mixed_labels: bool = False,
    probability_scores: bool = False,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Return cell aggregates and per-user metrics for paired comparisons."""

    if ranking_k <= 0:
        raise ValueError("ranking_k must be positive")
    ndcg_column = f"ndcg@{ranking_k}"
    strict_ndcg_column = f"failure_as_zero_{ndcg_column}"
    required = {
        "task_seed",
        "user_id",
        "candidate_group",
        "history_policy",
        "target_policy",
        "history_control",
        "target",
        "score",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    cell_columns = [
        "task_seed",
        "history_policy",
        "target_policy",
        "history_control",
        "metadata_condition",
        "matched",
    ]
    records: list[dict[str, object]] = []
    per_user_parts: list[pd.DataFrame] = []
    for cell_key, cell in predictions.groupby(cell_columns, dropna=False, sort=False):
        key = dict(zip(cell_columns, cell_key, strict=True))
        per_user = _per_user_metrics(
            cell,
            ranking_k=ranking_k,
            require_mixed_labels=require_mixed_labels,
            probability_scores=probability_scores,
        )
        per_user_parts.append(per_user.assign(**key))
        eligible = per_user["metric_eligible"]
        successful = per_user[eligible & ~per_user["failed"]]
        eligible_count = int(eligible.sum())
        record = {
                **key,
                "n_users": int(len(per_user)),
                "n_ranking_eligible_users": int(
                    per_user["ranking_eligible"].sum()
                ),
                "ranking_eligible_rate": float(
                    per_user["ranking_eligible"].mean()
                ),
                "n_metric_eligible_users": eligible_count,
                "n_scored_users": int(len(successful)),
                "failed_users": int(
                    (eligible & per_user["failed"]).sum()
                ),
                "score_failure_users_all_groups": int(
                    per_user["failed"].sum()
                ),
                "coverage": (
                    float(len(successful) / eligible_count)
                    if eligible_count
                    else math.nan
                ),
                ndcg_column: _safe_mean(successful[ndcg_column]),
                "hit_rate@1": _safe_mean(successful["hit_rate@1"]),
                strict_ndcg_column: _safe_mean(
                    per_user.loc[eligible, strict_ndcg_column]
                ),
                "failure_as_zero_hit_rate@1": _safe_mean(
                    per_user.loc[eligible, "failure_as_zero_hit_rate@1"]
                ),
                "users_with_score_ties": int(
                    per_user.loc[eligible, "has_score_ties"].sum()
                ),
                "probability_scores": probability_scores,
            }
        if probability_scores:
            valid_rows = cell[cell["score"].notna()]
            pointwise = _pointwise_probability_metrics(valid_rows)
            record.update(
                {
                    "pointwise_rows": int(len(valid_rows)),
                    "pointwise_row_coverage": float(
                        len(valid_rows) / len(cell)
                    ),
                    "log_loss": _safe_mean(per_user["log_loss"]),
                    "brier": _safe_mean(per_user["brier"]),
                    **{
                        f"row_{name}": value
                        for name, value in pointwise.items()
                    },
                }
            )
        records.append(record)
    per_user_frame = pd.concat(per_user_parts, ignore_index=True)
    return records, per_user_frame


def build_paired_comparisons(
    per_user: pd.DataFrame,
    *,
    metrics: Iterable[str] = (
        "ndcg@3",
        "hit_rate@1",
        "failure_as_zero_ndcg@3",
        "failure_as_zero_hit_rate@1",
    ),
    bootstrap_samples: int = 2_000,
    seed: int = 20260724,
) -> list[dict[str, object]]:
    """Build paired, user-level comparisons without treating seeds as users."""

    comparisons: list[dict[str, object]] = []
    target_policies = sorted(per_user["target_policy"].dropna().unique())
    for target_policy in target_policies:
        other_policy = "random" if target_policy == "standard" else "standard"
        selectors = [
            (
                "matched_minus_transferred",
                {
                    "history_policy": target_policy,
                    "history_control": "observed",
                },
                {
                    "history_policy": other_policy,
                    "history_control": "observed",
                },
            ),
            (
                "matched_minus_no_history",
                {
                    "history_policy": target_policy,
                    "history_control": "observed",
                },
                {
                    "history_policy": "none",
                    "history_control": "none",
                },
            ),
            (
                "transferred_minus_no_history",
                {
                    "history_policy": other_policy,
                    "history_control": "observed",
                },
                {
                    "history_policy": "none",
                    "history_control": "none",
                },
            ),
            (
                "matched_original_minus_permuted",
                {
                    "history_policy": target_policy,
                    "history_control": "observed",
                },
                {
                    "history_policy": target_policy,
                    "history_control": "permuted",
                },
            ),
            (
                "transferred_original_minus_permuted",
                {
                    "history_policy": other_policy,
                    "history_control": "observed",
                },
                {
                    "history_policy": other_policy,
                    "history_control": "permuted",
                },
            ),
        ]
        target_rows = per_user[per_user["target_policy"].eq(target_policy)]
        for comparison_name, left_selector, right_selector in selectors:
            left = _select_cell(target_rows, left_selector)
            right = _select_cell(target_rows, right_selector)
            if left.empty or right.empty:
                continue
            for metric in metrics:
                comparison = paired_user_comparison(
                    left,
                    right,
                    metric=metric,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                )
                comparisons.append(
                    {
                        "comparison": comparison_name,
                        "target_policy": target_policy,
                        "metric": metric,
                        "failure_policy": (
                            "failed_group_zero"
                            if metric.startswith("failure_as_zero_")
                            else "complete_case_pair"
                        ),
                        "higher_is_better": metric not in {"log_loss", "brier"},
                        "left": left_selector,
                        "right": right_selector,
                        **comparison,
                    }
                )
    return comparisons


def paired_user_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    metric: str,
    bootstrap_samples: int = 2_000,
    seed: int = 20260724,
) -> dict[str, object]:
    """Compare cells pairwise, averaging task seeds within each user first."""

    keys = ["task_seed", "user_id"]
    left_values = left.loc[:, [*keys, metric]].rename(
        columns={metric: "left"}
    )
    right_values = right.loc[:, [*keys, metric]].rename(
        columns={metric: "right"}
    )
    paired = left_values.merge(right_values, on=keys, validate="one_to_one")
    paired = paired.dropna(subset=["left", "right"])
    if paired.empty:
        return {
            "n_users": 0,
            "n_seed_user_pairs": 0,
            "metric_eligible_seeds_per_user_min": 0,
            "metric_eligible_seeds_per_user_mean": math.nan,
            "metric_eligible_seeds_per_user_max": 0,
            "users_with_all_metric_eligible_seeds": 0,
            "available_task_seed_count": 0,
            "mean_delta": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    paired["delta"] = paired["left"] - paired["right"]
    user_delta = paired.groupby("user_id", sort=False)["delta"].mean()
    seeds_per_user = paired.groupby("user_id", sort=False)[
        "task_seed"
    ].nunique()
    available_seed_count = int(paired["task_seed"].nunique())
    rng = np.random.default_rng(seed)
    values = user_delta.to_numpy(dtype=float)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    draws = rng.choice(
        values,
        size=(bootstrap_samples, len(values)),
        replace=True,
    ).mean(axis=1)
    return {
        "n_users": int(len(values)),
        "n_seed_user_pairs": int(len(paired)),
        "metric_eligible_seeds_per_user_min": int(seeds_per_user.min()),
        "metric_eligible_seeds_per_user_mean": float(
            seeds_per_user.mean()
        ),
        "metric_eligible_seeds_per_user_max": int(seeds_per_user.max()),
        "users_with_all_metric_eligible_seeds": int(
            seeds_per_user.eq(available_seed_count).sum()
        ),
        "available_task_seed_count": available_seed_count,
        "mean_delta": float(values.mean()),
        "median_user_delta": float(np.median(values)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_unit": "user_after_mean_over_task_seeds",
        "bootstrap_samples": bootstrap_samples,
    }


def _per_user_metrics(
    cell: pd.DataFrame,
    *,
    ranking_k: int,
    require_mixed_labels: bool,
    probability_scores: bool,
) -> pd.DataFrame:
    ndcg_column = f"ndcg@{ranking_k}"
    strict_ndcg_column = f"failure_as_zero_{ndcg_column}"
    records: list[dict[str, object]] = []
    for group_id, group in cell.groupby("candidate_group", sort=False):
        user_ids = group["user_id"].drop_duplicates()
        if len(user_ids) != 1:
            raise ValueError("Each candidate group must contain one user")
        positives = int(group["target"].sum())
        ranking_eligible = 0 < positives < len(group)
        metric_eligible = ranking_eligible if require_mixed_labels else True
        failed = bool(group["score"].isna().any())
        if probability_scores and not failed:
            pointwise = _pointwise_probability_metrics(group)
            user_log_loss = pointwise["log_loss"]
            user_brier = pointwise["brier"]
        else:
            user_log_loss = math.nan
            user_brier = math.nan
        if not metric_eligible:
            ndcg = math.nan
            hit = math.nan
            strict_ndcg = math.nan
            strict_hit = math.nan
            ties = False
        elif failed:
            ndcg = math.nan
            hit = math.nan
            strict_ndcg = 0.0
            strict_hit = 0.0
            ties = False
        else:
            metrics = grouped_ranking_metrics(
                group["target"],
                group["score"],
                group["candidate_group"],
                ks=tuple(sorted({1, ranking_k})),
                tie_policy=TIE_POLICY,
            )
            ndcg = float(metrics[ndcg_column])
            hit = float(metrics["hit_rate@1"])
            strict_ndcg = ndcg
            strict_hit = hit
            ties = bool(metrics["groups_with_score_ties"])
        records.append(
            {
                "user_id": user_ids.iloc[0],
                "candidate_group": group_id,
                "candidate_rows": int(len(group)),
                "positive_labels": positives,
                "negative_labels": int(len(group) - positives),
                "ranking_eligible": ranking_eligible,
                "metric_eligible": metric_eligible,
                ndcg_column: ndcg,
                "hit_rate@1": hit,
                strict_ndcg_column: strict_ndcg,
                "failure_as_zero_hit_rate@1": strict_hit,
                "failed": failed,
                "has_score_ties": ties,
                "log_loss": user_log_loss,
                "brier": user_brier,
            }
        )
    return pd.DataFrame.from_records(records)


def _select_cell(
    rows: pd.DataFrame,
    selector: dict[str, str],
) -> pd.DataFrame:
    selected = rows
    for column, value in selector.items():
        selected = selected[selected[column].eq(value)]
    return selected


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else math.nan


def _pointwise_probability_metrics(
    rows: pd.DataFrame,
) -> dict[str, float]:
    if rows.empty:
        return {
            "log_loss": math.nan,
            "brier": math.nan,
            "roc_auc": math.nan,
            "average_precision": math.nan,
            "observed_click_rate": math.nan,
            "mean_predicted_probability": math.nan,
            "calibration_intercept": math.nan,
            "calibration_slope": math.nan,
        }
    y_true = rows["target"].astype(int).to_numpy()
    probability = rows["score"].astype(float).to_numpy()
    if not np.isfinite(probability).all() or (
        (probability < 0.0) | (probability > 1.0)
    ).any():
        raise ValueError("Probability scores must be finite and in [0, 1]")
    epsilon = 1e-6
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    both_classes = len(np.unique(y_true)) == 2
    intercept, slope = _calibration_fit(
        y_true,
        clipped,
        both_classes=both_classes,
    )
    return {
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, clipped)),
        "roc_auc": (
            float(roc_auc_score(y_true, clipped))
            if both_classes
            else math.nan
        ),
        "average_precision": (
            float(average_precision_score(y_true, clipped))
            if y_true.sum() > 0
            else math.nan
        ),
        "observed_click_rate": float(y_true.mean()),
        "mean_predicted_probability": float(clipped.mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _calibration_fit(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    both_classes: bool,
) -> tuple[float, float]:
    if not both_classes or len(np.unique(probability)) < 2:
        return math.nan, math.nan
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=1_000,
    )
    model.fit(logit, y_true)
    return float(model.intercept_[0]), float(model.coef_[0, 0])
