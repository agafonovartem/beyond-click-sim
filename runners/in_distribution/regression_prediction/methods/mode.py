from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from beyond_click_sim.scorers import ModeRegressor
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


METHOD_NAME = "mode_regressor"


def run(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the train-mode baseline for discrete regression prediction."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    if task.schema.candidate_group_column is not None:
        raise ValueError("Regression method requires observed-only task rows")
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    _record_stage(stage_times, "prepare_xy", stage_start)

    stage_start = perf_counter()
    scorer = ModeRegressor().fit(X_train, y_train)
    _record_stage(stage_times, "fit", stage_start)

    stage_start = perf_counter()
    val_scores = scorer.score(X_val)
    _record_stage(stage_times, "score_val", stage_start)

    stage_start = perf_counter()
    test_scores = scorer.score(X_test)
    _record_stage(stage_times, "score_test", stage_start)

    stage_start = perf_counter()
    val_metrics = regression_metrics_for_split(X=X_val, y=y_val, scores=val_scores)
    test_metrics = regression_metrics_for_split(X=X_test, y=y_test, scores=test_scores)
    _record_stage(stage_times, "compute_metrics", stage_start)

    stage_start = perf_counter()
    predictions = pd.concat(
        [
            score_frame(split="val", X=X_val, y=y_val, scores=val_scores),
            score_frame(split="test", X=X_test, y=y_test, scores=test_scores),
        ],
        ignore_index=True,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _record_stage(stage_times, "write_predictions", stage_start)

    root = repo_root()
    manifest = {
        "method": METHOD_NAME,
        "protocol": "regression",
        "scorer": {
            "class": "ModeRegressor",
            "mode": scorer.mode_,
            "tie_break": scorer.tie_break,
        },
        "regression_evaluation": {
            "main_metric": REGRESSION_MAIN_METRIC,
            "aggregations": ["micro", "macro_by_user_mean"],
            "metrics": ["mae", "rmse"],
        },
        "task": {
            "name": task.name,
            "manifest": task.manifest,
        },
        "stage_times_seconds": stage_times,
        "git_commit": current_git_commit(root),
    }
    metrics = {
        "method": METHOD_NAME,
        "task": task.name,
        "protocol": "regression",
        "main_metric": REGRESSION_MAIN_METRIC,
        "regression_evaluation": {
            "aggregations": ["micro", "macro_by_user_mean"],
            "metrics": ["mae", "rmse"],
        },
        "val": val_metrics,
        "test": test_metrics,
        "stage_times_seconds": stage_times,
    }
    stage_start = perf_counter()
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / REGRESSION_METRICS_FILENAME, metrics)
    _record_stage(stage_times, "write_metadata", stage_start)
    return metrics


def _record_stage(stage_times: dict[str, float], stage: str, start: float) -> None:
    seconds = round(perf_counter() - start, 3)
    stage_times[stage] = seconds
    print(f"[mode_regressor] {stage}: {seconds}s", flush=True)
