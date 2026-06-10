from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from beyond_click_sim.evaluation import (
    apply_threshold,
    binary_classification_metrics,
    find_best_user_group_threshold,
    grouped_binary_classification_metrics,
    user_grouped_binary_classification_metrics,
)
from beyond_click_sim.scorers import PopularityScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods.common import (
    candidate_group_summary,
    current_git_commit,
    prediction_frame,
    task_xy,
    write_json,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root


METHOD_NAME = "popularity_f1_threshold"
THRESHOLD_METRIC = "macro_by_user_group_mean_f1"


def run(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the deterministic item-popularity baseline for pointwise alignment."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("Popularity alignment method requires candidate_group_column")
    _record_stage(stage_times, "prepare_xy", stage_start)

    stage_start = perf_counter()
    scorer = PopularityScorer(item_column="item_id").fit(X_train, y_train)
    _record_stage(stage_times, "fit", stage_start)

    stage_start = perf_counter()
    val_scores = scorer.score(X_val)
    _record_stage(stage_times, "score_val", stage_start)

    stage_start = perf_counter()
    test_scores = scorer.score(X_test)
    _record_stage(stage_times, "score_test", stage_start)

    stage_start = perf_counter()
    threshold_selection = find_best_user_group_threshold(
        y_val,
        val_scores,
        X_val[candidate_group_column],
        X_val["user_id"],
        metric="f1",
    )
    threshold = float(threshold_selection["threshold"])
    _record_stage(stage_times, "select_threshold", stage_start)

    stage_start = perf_counter()
    val_predictions = apply_threshold(val_scores, threshold)
    test_predictions = apply_threshold(test_scores, threshold)
    _record_stage(stage_times, "apply_threshold", stage_start)

    stage_start = perf_counter()
    val_macro_metrics = grouped_binary_classification_metrics(
        y_val,
        val_predictions,
        X_val[candidate_group_column],
    )
    val_user_group_metrics = user_grouped_binary_classification_metrics(
        y_val,
        val_predictions,
        X_val[candidate_group_column],
        X_val["user_id"],
    )
    test_macro_metrics = grouped_binary_classification_metrics(
        y_test,
        test_predictions,
        X_test[candidate_group_column],
    )
    test_user_group_metrics = user_grouped_binary_classification_metrics(
        y_test,
        test_predictions,
        X_test[candidate_group_column],
        X_test["user_id"],
    )
    val_micro_metrics = binary_classification_metrics(y_val, val_predictions)
    test_micro_metrics = binary_classification_metrics(y_test, test_predictions)
    _record_stage(stage_times, "compute_metrics", stage_start)

    stage_start = perf_counter()
    val_candidate_groups = candidate_group_summary(
        X_val,
        y_val,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )
    test_candidate_groups = candidate_group_summary(
        X_test,
        y_test,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )
    _record_stage(stage_times, "candidate_group_summary", stage_start)

    stage_start = perf_counter()
    predictions = pd.concat(
        [
            prediction_frame(
                split="val",
                X=X_val,
                y=y_val,
                scores=val_scores,
                predictions=val_predictions,
            ),
            prediction_frame(
                split="test",
                X=X_test,
                y=y_test,
                scores=test_scores,
                predictions=test_predictions,
            ),
        ],
        ignore_index=True,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _record_stage(stage_times, "write_predictions", stage_start)

    root = repo_root()
    manifest = {
        "method": METHOD_NAME,
        "scorer": {
            "class": "PopularityScorer",
            "item_column": "item_id",
        },
        "decision_rule": {
            "kind": "threshold_on_validation",
            "selection_metric": THRESHOLD_METRIC,
            **threshold_selection,
        },
        "task": {
            "name": task.name,
            "manifest": task.manifest,
        },
        "candidate_groups": {
            "val": val_candidate_groups,
            "test": test_candidate_groups,
        },
        "stage_times_seconds": stage_times,
        "git_commit": current_git_commit(root),
    }
    metrics = {
        "method": METHOD_NAME,
        "task": task.name,
        "main_metric": "test.macro_by_user_group_mean.f1",
        "val": {
            "macro_by_group": val_macro_metrics,
            "macro_by_user_group_mean": val_user_group_metrics,
            "micro": val_micro_metrics,
        },
        "test": {
            "macro_by_group": test_macro_metrics,
            "macro_by_user_group_mean": test_user_group_metrics,
            "micro": test_micro_metrics,
        },
        "threshold": threshold,
        "threshold_metric": THRESHOLD_METRIC,
        "candidate_groups": {
            "val": val_candidate_groups,
            "test": test_candidate_groups,
        },
        "stage_times_seconds": stage_times,
    }
    stage_start = perf_counter()
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "metrics.json", metrics)
    _record_stage(stage_times, "write_metadata", stage_start)
    return metrics


def _record_stage(stage_times: dict[str, float], stage: str, start: float) -> None:
    seconds = round(perf_counter() - start, 3)
    stage_times[stage] = seconds
    print(f"[popularity] {stage}: {seconds}s", flush=True)
