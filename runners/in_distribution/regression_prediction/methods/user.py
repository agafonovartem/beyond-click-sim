from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from beyond_click_sim.scorers import UserMeanRegressor, UserModeRegressor
from beyond_click_sim.tasks import Task
from runners.in_distribution.regression_prediction.config import (
    DATASET_TARGET_REGRESSION_CONFIG,
    MAX_HISTORY_ITEMS,
)
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


USER_MEAN_METHOD_NAME = "user_mean_regressor"
USER_MODE_METHOD_NAME = "user_mode_regressor"
USER_MODE_FULL_HISTORY_METHOD_NAME = "user_mode_full_history_regressor"


def run_user_mean(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=USER_MEAN_METHOD_NAME,
        scorer_class=UserMeanRegressor,
    )


def run_user_mode(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=USER_MODE_METHOD_NAME,
        scorer_class=UserModeRegressor,
    )


def run_user_mode_full_history(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=USER_MODE_FULL_HISTORY_METHOD_NAME,
        scorer_class=UserModeRegressor,
        max_history_items=None,
    )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    scorer_class: type[UserMeanRegressor] | type[UserModeRegressor],
    max_history_items: int | None = MAX_HISTORY_ITEMS,
) -> dict[str, object]:
    """Run a per-user baseline over the same train-history window shown to the LLM."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    if task.schema.candidate_group_column is not None:
        raise ValueError("Regression method requires observed-only task rows")

    dataset_name = str(task.manifest["dataset"])
    target_source_column = str(task.manifest["target_source_column"])
    target_config = DATASET_TARGET_REGRESSION_CONFIG[dataset_name][target_source_column]
    history_value_column = str(target_config["history_value_column"])

    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    _record_stage(stage_times, method_name, "prepare_xy", stage_start)

    stage_start = perf_counter()
    scorer = scorer_class(
        history_value_column=history_value_column,
        max_history_items=max_history_items,
    ).fit(X_train, y_train)
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
    scorer_manifest = _scorer_manifest(
        scorer=scorer,
        history_value_column=history_value_column,
        target_config=target_config,
        max_history_items=max_history_items,
    )
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
        "stage_times_seconds": stage_times,
    }
    stage_start = perf_counter()
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / REGRESSION_METRICS_FILENAME, metrics)
    _record_stage(stage_times, method_name, "write_metadata", stage_start)
    return metrics


def _scorer_manifest(
    *,
    scorer: UserMeanRegressor | UserModeRegressor,
    history_value_column: str,
    target_config: dict[str, Any],
    max_history_items: int | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "class": scorer.__class__.__name__,
        "history_value_column": history_value_column,
        "history_window": {
            "source": "train_rows",
            "selection": scorer.history_selection,
            "max_history_items": max_history_items,
            "matched_llm_regressor_history": max_history_items == MAX_HISTORY_ITEMS,
        },
        "prompt_columns": {
            "history_description_columns": target_config["history_description_columns"],
            "candidate_description_columns": target_config["candidate_description_columns"],
        },
    }
    if isinstance(scorer, UserModeRegressor):
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
