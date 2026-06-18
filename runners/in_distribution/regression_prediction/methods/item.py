from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from beyond_click_sim.scorers import ItemMeanRegressor, ItemModeRegressor
from beyond_click_sim.tasks import Task
from runners.in_distribution.regression_prediction.methods.common import (
    current_git_commit,
    regression_metrics_for_split,
    score_frame,
    task_xy,
    write_json,
)
from runners.in_distribution.regression_prediction.metrics import (
    REGRESSION_MAIN_METRIC,
    REGRESSION_METRICS_FILENAME,
)
from runners.in_distribution.regression_prediction.task_builders import repo_root


ITEM_MEAN_METHOD_NAME = "item_mean_regressor"
ITEM_MODE_METHOD_NAME = "item_mode_regressor"


def run_item_mean(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=ITEM_MEAN_METHOD_NAME,
        scorer_class=ItemMeanRegressor,
    )


def run_item_mode(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=ITEM_MODE_METHOD_NAME,
        scorer_class=ItemModeRegressor,
    )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    scorer_class: type[ItemMeanRegressor] | type[ItemModeRegressor],
) -> dict[str, object]:
    """Run a per-item train-target statistic baseline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    if task.schema.candidate_group_column is not None:
        raise ValueError("Regression method requires observed-only task rows")

    item_column = task.schema.id_columns[1]
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    _record_stage(stage_times, method_name, "prepare_xy", stage_start)

    stage_start = perf_counter()
    scorer = scorer_class(item_column=item_column).fit(X_train, y_train)
    _record_stage(stage_times, method_name, "fit", stage_start)

    stage_start = perf_counter()
    val_scores = scorer.score(X_val)
    _record_stage(stage_times, method_name, "score_val", stage_start)

    stage_start = perf_counter()
    test_scores = scorer.score(X_test)
    _record_stage(stage_times, method_name, "score_test", stage_start)

    stage_start = perf_counter()
    val_metrics = regression_metrics_for_split(X=X_val, y=y_val, scores=val_scores)
    test_metrics = regression_metrics_for_split(X=X_test, y=y_test, scores=test_scores)
    cold_item_rows = {
        "val": scorer.cold_item_rows(X_val),
        "test": scorer.cold_item_rows(X_test),
    }
    _record_stage(stage_times, method_name, "compute_metrics", stage_start)

    stage_start = perf_counter()
    predictions = pd.concat(
        [
            score_frame(split="val", X=X_val, y=y_val, scores=val_scores),
            score_frame(split="test", X=X_test, y=y_test, scores=test_scores),
        ],
        ignore_index=True,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _record_stage(stage_times, method_name, "write_predictions", stage_start)

    root = repo_root()
    scorer_manifest = _scorer_manifest(scorer=scorer)
    manifest = {
        "method": method_name,
        "protocol": "regression",
        "scorer": scorer_manifest,
        "regression_evaluation": {
            "main_metric": REGRESSION_MAIN_METRIC,
            "aggregations": ["micro", "macro_by_user_mean"],
            "metrics": ["mae", "rmse"],
        },
        "evaluated_splits": ["val", "test"],
        "cold_item_rows": cold_item_rows,
        "task": {
            "name": task.name,
            "manifest": task.manifest,
        },
        "stage_times_seconds": stage_times,
        "git_commit": current_git_commit(root),
    }
    metrics = {
        "method": method_name,
        "task": task.name,
        "protocol": "regression",
        "main_metric": REGRESSION_MAIN_METRIC,
        "regression_evaluation": {
            "aggregations": ["micro", "macro_by_user_mean"],
            "metrics": ["mae", "rmse"],
        },
        "evaluated_splits": ["val", "test"],
        "val": val_metrics,
        "test": test_metrics,
        "cold_item_rows": cold_item_rows,
        "stage_times_seconds": stage_times,
    }
    stage_start = perf_counter()
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / REGRESSION_METRICS_FILENAME, metrics)
    _record_stage(stage_times, method_name, "write_metadata", stage_start)
    return metrics


def _scorer_manifest(
    *,
    scorer: ItemMeanRegressor | ItemModeRegressor,
) -> dict[str, Any]:
    fallback_scorer = scorer.fallback_scorer_
    manifest: dict[str, Any] = {
        "class": scorer.__class__.__name__,
        "item_column": scorer.item_column,
        "stat_source": scorer.stat_source,
        "n_items_seen_train": len(scorer.item_count_by_item_ or {}),
        "cold_item_policy": scorer.cold_item_policy,
        "fallback": {
            "scorer": None if fallback_scorer is None else fallback_scorer.__class__.__name__,
            "value": scorer.fallback_,
        },
    }
    if isinstance(scorer, ItemModeRegressor):
        manifest["tie_break"] = scorer.tie_break
    return manifest


def _record_stage(
    stage_times: dict[str, float],
    method_name: str,
    stage: str,
    start: float,
) -> None:
    seconds = round(perf_counter() - start, 3)
    stage_times[stage] = seconds
    print(f"[{method_name}] {stage}: {seconds}s", flush=True)
