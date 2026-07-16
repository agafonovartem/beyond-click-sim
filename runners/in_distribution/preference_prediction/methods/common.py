from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from beyond_click_sim.evaluation import (
    binary_classification_metrics,
    grouped_binary_classification_metrics,
    grouped_ranking_metrics,
    user_grouped_binary_classification_metrics,
    user_grouped_ranking_metrics,
)
from beyond_click_sim.tasks import Task, split_xy
from runners.in_distribution.preference_prediction.metrics import (
    RANKING_KS,
    RANKING_TIE_POLICY,
)


def task_xy(task: Task) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    target_column = task.schema.target_column
    return {
        "train": split_xy(task.train, target_column=target_column),
        "val": split_xy(task.val, target_column=target_column),
        "test": split_xy(task.test, target_column=target_column),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def current_git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def prediction_frame(
    *,
    split: str,
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    frame = X.copy()
    frame.insert(0, "split", split)
    frame["target"] = y
    frame["score"] = scores
    frame["prediction"] = predictions
    return frame


def score_frame(
    *,
    split: str,
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
) -> pd.DataFrame:
    frame = X.copy()
    frame.insert(0, "split", split)
    frame["target"] = y
    frame["score"] = scores
    return frame


def ranking_metrics_for_split(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
    candidate_group_column: str,
    ks: tuple[int, ...] = RANKING_KS,
    tie_policy: str = RANKING_TIE_POLICY,
) -> dict[str, dict[str, float | int | str]]:
    return {
        "macro_by_group": grouped_ranking_metrics(
            y,
            scores,
            X[candidate_group_column],
            ks=ks,
            tie_policy=tie_policy,  # type: ignore[arg-type]
        ),
        "macro_by_user_group_mean": user_grouped_ranking_metrics(
            y,
            scores,
            X[candidate_group_column],
            X["user_id"],
            ks=ks,
            tie_policy=tie_policy,  # type: ignore[arg-type]
        ),
    }


def pointwise_metrics_for_split(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    predictions: pd.Series,
    candidate_group_column: str,
) -> dict[str, dict[str, float | int]]:
    """Compute the standard pointwise metrics payload for one split."""

    return {
        "macro_by_group": grouped_binary_classification_metrics(
            y,
            predictions,
            X[candidate_group_column],
        ),
        "macro_by_user_group_mean": user_grouped_binary_classification_metrics(
            y,
            predictions,
            X[candidate_group_column],
            X["user_id"],
        ),
        "micro": binary_classification_metrics(y, predictions),
    }


def failure_as_negative_pointwise_metrics(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
    candidate_group_column: str,
) -> dict[str, dict[str, float | int]]:
    """Compute pointwise metrics with failed LLM rows counted as `no`.

    LLM yes/no methods leave rows from failed candidate groups as NaN. The
    parsed-only metrics intentionally exclude those rows; this stricter variant
    treats every missing decision as a negative prediction.
    """

    strict_predictions = scores.fillna(0.0).astype(bool).rename("prediction")
    return pointwise_metrics_for_split(
        X=X,
        y=y,
        predictions=strict_predictions,
        candidate_group_column=candidate_group_column,
    )


def ranking_metrics_with_failed_groups_as_zero(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
    candidate_group_column: str,
    ks: tuple[int, ...] = RANKING_KS,
    tie_policy: str = RANKING_TIE_POLICY,
) -> dict[str, dict[str, float | int | str]]:
    """Compute ranking metrics where failed candidate groups get zero utility.

    Filling failed groups with score=0 would evaluate them as random ties, which
    is too generous for "LLM did not answer". This helper instead assigns zero
    hit-rate/NDCG to any candidate group containing missing scores, then macro
    averages those zero-valued groups with successfully scored groups.
    """

    records: list[dict[str, Any]] = []
    metric_keys = [f"{name}@{k}" for k in ks for name in ("hit_rate", "ndcg")]
    for group_id, group in X.groupby(candidate_group_column, sort=False):
        group_scores = scores.loc[group.index]
        group_y = y.loc[group.index]
        user_ids = group["user_id"].drop_duplicates()
        if len(user_ids) != 1:
            raise ValueError("Each candidate group must contain exactly one user")

        failed = bool(group_scores.isna().any())
        if failed:
            metric_values = {key: 0.0 for key in metric_keys}
            has_score_ties = False
        else:
            group_metrics = grouped_ranking_metrics(
                group_y,
                group_scores,
                group[candidate_group_column],
                ks=ks,
                tie_policy=tie_policy,  # type: ignore[arg-type]
            )
            metric_values = {key: float(group_metrics[key]) for key in metric_keys}
            has_score_ties = bool(group_metrics["groups_with_score_ties"])

        records.append(
            {
                "group": group_id,
                "user_id": user_ids.iloc[0],
                "n": int(len(group)),
                "n_positive": int(group_y.astype(bool).sum()),
                "failed": failed,
                "has_score_ties": has_score_ties,
                **metric_values,
            }
        )

    if not records:
        raise ValueError("Cannot compute ranking metrics on empty inputs")

    per_group = pd.DataFrame.from_records(records)
    return {
        "macro_by_group": _aggregate_failure_as_zero_ranking_records(
            per_group,
            ks=ks,
            tie_policy=tie_policy,
        ),
        "macro_by_user_group_mean": _aggregate_failure_as_zero_ranking_records(
            per_group,
            ks=ks,
            tie_policy=tie_policy,
            group_by_user=True,
        ),
    }


def score_coverage_summary(scores: pd.Series) -> dict[str, int | float]:
    """Return row-level coverage diagnostics for LLM score vectors."""

    requested_rows = int(len(scores))
    scored_rows = int(scores.notna().sum())
    failed_rows = requested_rows - scored_rows
    return {
        "requested_rows": requested_rows,
        "scored_rows": scored_rows,
        "failed_rows": failed_rows,
        "scored_fraction": (
            float(scored_rows / requested_rows) if requested_rows else 0.0
        ),
        "failed_fraction": (
            float(failed_rows / requested_rows) if requested_rows else 0.0
        ),
    }


def _aggregate_failure_as_zero_ranking_records(
    per_group: pd.DataFrame,
    *,
    ks: tuple[int, ...],
    tie_policy: str,
    group_by_user: bool = False,
) -> dict[str, float | int | str]:
    metric_keys = [f"{name}@{k}" for k in ks for name in ("hit_rate", "ndcg")]
    if group_by_user:
        metrics = {
            key: float(per_group.groupby("user_id", sort=False)[key].mean().mean())
            for key in metric_keys
        }
    else:
        metrics = {key: float(per_group[key].mean()) for key in metric_keys}

    n_groups = int(len(per_group))
    diagnostics: dict[str, float | int | str] = {
        "n_groups": n_groups,
        "n": int(per_group["n"].sum()),
        "n_positive": int(per_group["n_positive"].sum()),
        "groups_without_positive": int(per_group["n_positive"].eq(0).sum()),
        "groups_with_score_ties": int(per_group["has_score_ties"].sum()),
        "groups_with_score_ties_fraction": float(per_group["has_score_ties"].mean()),
        "failed_groups": int(per_group["failed"].sum()),
        "failed_groups_fraction": float(per_group["failed"].mean()),
        "scored_groups": int((~per_group["failed"]).sum()),
        "failure_policy": "failed_group_zero",
        "tie_policy": tie_policy,
    }
    for k in ks:
        diagnostics[f"groups_with_size_lte@{k}"] = int(per_group["n"].le(k).sum())
    if group_by_user:
        diagnostics["n_users"] = int(per_group["user_id"].nunique())

    return {**metrics, **diagnostics}


def candidate_group_summary(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    candidate_group_column: str,
    sampled_column: str | None,
) -> dict[str, int | float]:
    frame = pd.DataFrame(
        {
            "group": X[candidate_group_column].to_numpy(),
            "target": y.astype(int).to_numpy(),
        }
    )
    if sampled_column is not None and sampled_column in X.columns:
        frame["sampled"] = X[sampled_column].astype(bool).to_numpy().astype(int)
    else:
        frame["sampled"] = 0

    grouped = frame.groupby("group", sort=False).agg(
        rows=("target", "size"),
        positives=("target", "sum"),
        sampled=("sampled", "sum"),
    )
    if grouped.empty:
        return {
            "groups": 0,
            "rows_min": 0,
            "rows_mean": 0.0,
            "rows_max": 0,
            "positives_min": 0,
            "positives_mean": 0.0,
            "positives_max": 0,
            "sampled_min": 0,
            "sampled_mean": 0.0,
            "sampled_max": 0,
        }
    return {
        "groups": int(len(grouped)),
        "rows_min": int(grouped["rows"].min()),
        "rows_mean": float(grouped["rows"].mean()),
        "rows_max": int(grouped["rows"].max()),
        "positives_min": int(grouped["positives"].min()),
        "positives_mean": float(grouped["positives"].mean()),
        "positives_max": int(grouped["positives"].max()),
        "sampled_min": int(grouped["sampled"].min()),
        "sampled_mean": float(grouped["sampled"].mean()),
        "sampled_max": int(grouped["sampled"].max()),
    }


def limit_candidate_groups(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    candidate_group_column: str,
    max_candidate_groups: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_candidate_groups is None:
        return X, y
    groups = X[candidate_group_column].drop_duplicates().head(max_candidate_groups)
    mask = X[candidate_group_column].isin(groups)
    return X.loc[mask].copy(), y.loc[mask].copy()
