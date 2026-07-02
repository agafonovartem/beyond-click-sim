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
from beyond_click_sim.scorers import ColdItemKNNScorer
from beyond_click_sim.tasks import split_xy
from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.interaction_prediction.methods.common import (
    candidate_group_summary,
    current_git_commit,
    limit_candidate_groups,
    prediction_frame,
    ranking_metrics_for_split,
    score_frame,
    task_xy,
    write_json,
)
from runners.in_distribution.cold_start.metrics import (
    POINTWISE_MAIN_METRIC,
    POINTWISE_METRICS_FILENAME,
    RANKING_KS,
    RANKING_MAIN_METRIC,
    RANKING_METRICS_FILENAME,
    RANKING_TIE_POLICY,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root


N_NEIGHBORS = 20
AGGREGATION = "mean"
METHOD_NAME = "item_knn_cold_start"
RANKING_METHOD_NAME = "item_knn_cold_start_ranking"
THRESHOLD_METRIC = "macro_by_user_group_mean_f1"


def run(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    """Run the ItemKNN baseline for cold-start pointwise alignment."""
    return _run_method(task, output_dir, max_candidate_groups=None)


def run_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    """Run ItemKNN on the first 25 candidate groups (smoke test)."""
    return _run_method(task, output_dir, max_candidate_groups=25)


def run_ranking(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    """Run the ItemKNN baseline for cold-start ranking evaluation."""
    return _run_ranking_method(task, output_dir, max_candidate_groups=None)


def run_ranking_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    """Run ItemKNN ranking on the first 25 candidate groups (smoke test)."""
    return _run_ranking_method(task, output_dir, max_candidate_groups=25)


def _run_method(
    task: ColdStartTask,
    output_dir: Path,
    max_candidate_groups: int | None,
) -> dict[str, object]:
    """ItemKNN cold-start runner with threshold calibration on validation.

    Fit sequence:
      1. fit_train(X_train, y_train) — builds item-item cosine similarity from warm train rows.
      2. fit(X_history, y_history)  — stores each cold user's k-item profile from
                                       task.online_session_history (same frame the LLM scorer uses).
    Threshold is selected on val cold users, then applied to test cold users.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    X_history, y_history = split_xy(
        task.online_session_history,
        target_column=task.schema.target_column,
    )
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("ItemKNN cold-start method requires candidate_group_column")
    X_val, y_val = limit_candidate_groups(
        X_val, y_val,
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    X_test, y_test = limit_candidate_groups(
        X_test, y_test,
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    _record_stage(stage_times, "prepare_xy", stage_start)

    stage_start = perf_counter()
    scorer = (
        ColdItemKNNScorer(
            n_neighbors=N_NEIGHBORS,
            aggregation=AGGREGATION,
            item_column="item_id",
            user_column="user_id",
        )
        .fit_train(X_train, y_train)
        .fit(X_history, y_history)
    )
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
        y_val, val_predictions, X_val[candidate_group_column],
    )
    val_user_group_metrics = user_grouped_binary_classification_metrics(
        y_val, val_predictions, X_val[candidate_group_column], X_val["user_id"],
    )
    test_macro_metrics = grouped_binary_classification_metrics(
        y_test, test_predictions, X_test[candidate_group_column],
    )
    test_user_group_metrics = user_grouped_binary_classification_metrics(
        y_test, test_predictions, X_test[candidate_group_column], X_test["user_id"],
    )
    val_micro_metrics = binary_classification_metrics(y_val, val_predictions)
    test_micro_metrics = binary_classification_metrics(y_test, test_predictions)
    _record_stage(stage_times, "compute_metrics", stage_start)

    stage_start = perf_counter()
    val_candidate_groups = candidate_group_summary(
        X_val, y_val,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )
    test_candidate_groups = candidate_group_summary(
        X_test, y_test,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )
    _record_stage(stage_times, "candidate_group_summary", stage_start)

    stage_start = perf_counter()
    predictions = pd.concat(
        [
            prediction_frame(
                split="val", X=X_val, y=y_val,
                scores=val_scores, predictions=val_predictions,
            ),
            prediction_frame(
                split="test", X=X_test, y=y_test,
                scores=test_scores, predictions=test_predictions,
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
            "class": "ColdItemKNNScorer",
            "fit_on": "online_session_history",
            "k": task.k,
            "n_neighbors": N_NEIGHBORS,
            "aggregation": AGGREGATION,
            "item_column": "item_id",
            "user_column": "user_id",
        },
        "decision_rule": {
            "kind": "threshold_on_validation",
            "selection_metric": THRESHOLD_METRIC,
            **threshold_selection,
        },
        "limits": {
            "max_candidate_groups": max_candidate_groups,
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
        "main_metric": POINTWISE_MAIN_METRIC,
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
    write_json(output_dir / POINTWISE_METRICS_FILENAME, metrics)
    _record_stage(stage_times, "write_metadata", stage_start)
    return metrics


def _run_ranking_method(
    task: ColdStartTask,
    output_dir: Path,
    max_candidate_groups: int | None,
) -> dict[str, object]:
    """ItemKNN cold-start runner for raw-score ranking evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    X_history, y_history = split_xy(
        task.online_session_history,
        target_column=task.schema.target_column,
    )
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("ItemKNN cold-start ranking method requires candidate_group_column")
    X_val, y_val = limit_candidate_groups(
        X_val, y_val,
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    X_test, y_test = limit_candidate_groups(
        X_test, y_test,
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    _record_stage(stage_times, "prepare_xy", stage_start)

    stage_start = perf_counter()
    scorer = (
        ColdItemKNNScorer(
            n_neighbors=N_NEIGHBORS,
            aggregation=AGGREGATION,
            item_column="item_id",
            user_column="user_id",
        )
        .fit_train(X_train, y_train)
        .fit(X_history, y_history)
    )
    _record_stage(stage_times, "fit", stage_start)

    stage_start = perf_counter()
    val_scores = scorer.score(X_val)
    _record_stage(stage_times, "score_val", stage_start)

    stage_start = perf_counter()
    test_scores = scorer.score(X_test)
    _record_stage(stage_times, "score_test", stage_start)

    stage_start = perf_counter()
    val_ranking_metrics = ranking_metrics_for_split(
        X=X_val, y=y_val, scores=val_scores,
        candidate_group_column=candidate_group_column,
    )
    test_ranking_metrics = ranking_metrics_for_split(
        X=X_test, y=y_test, scores=test_scores,
        candidate_group_column=candidate_group_column,
    )
    _record_stage(stage_times, "compute_ranking_metrics", stage_start)

    stage_start = perf_counter()
    val_candidate_groups = candidate_group_summary(
        X_val, y_val,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )
    test_candidate_groups = candidate_group_summary(
        X_test, y_test,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )
    _record_stage(stage_times, "candidate_group_summary", stage_start)

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
        "method": RANKING_METHOD_NAME,
        "protocol": "ranking",
        "scorer": {
            "class": "ColdItemKNNScorer",
            "fit_on": "online_session_history",
            "k": task.k,
            "n_neighbors": N_NEIGHBORS,
            "aggregation": AGGREGATION,
            "item_column": "item_id",
            "user_column": "user_id",
        },
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
        },
        "limits": {
            "max_candidate_groups": max_candidate_groups,
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
        "method": RANKING_METHOD_NAME,
        "task": task.name,
        "protocol": "ranking",
        "main_metric": RANKING_MAIN_METRIC,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
        },
        "val": val_ranking_metrics,
        "test": test_ranking_metrics,
        "candidate_groups": {
            "val": val_candidate_groups,
            "test": test_candidate_groups,
        },
        "stage_times_seconds": stage_times,
    }
    stage_start = perf_counter()
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / RANKING_METRICS_FILENAME, metrics)
    _record_stage(stage_times, "write_metadata", stage_start)
    return metrics


def _record_stage(stage_times: dict[str, float], stage: str, start: float) -> None:
    seconds = round(perf_counter() - start, 3)
    stage_times[stage] = seconds
    print(f"[item_knn] {stage}: {seconds}s", flush=True)
