"""Generic runner for classical scorer baselines on the interaction task.

Any scorer that satisfies the ``Scorer`` contract (``fit(X, y) -> score(X)``)
and is fit on the positives-only train split can be evaluated here. The bodies
are the same proven pipeline used by ``methods/popularity.py`` — pointwise
F1-thresholded decisions and raw-score ranking — parameterized by the scorer
instance and its manifest description so ItemKNN/ALS/BPR/LightGCN reuse one code
path instead of copying ~300 lines each.

``PopularityScorer`` deliberately keeps its own ``methods/popularity.py`` runner
(a stable, already-reported baseline); this module is for the added classical
collaborative-filtering scorers.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from beyond_click_sim.evaluation import (
    apply_threshold,
    binary_classification_metrics,
    find_best_user_group_threshold,
    grouped_binary_classification_metrics,
    user_grouped_binary_classification_metrics,
)
from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods.common import (
    candidate_group_summary,
    current_git_commit,
    prediction_frame,
    ranking_metrics_for_split,
    score_frame,
    task_xy,
    write_json,
)
from runners.in_distribution.interaction_prediction.metrics import (
    POINTWISE_MAIN_METRIC,
    POINTWISE_METRICS_FILENAME,
    RANKING_KS,
    RANKING_MAIN_METRIC,
    RANKING_METRICS_FILENAME,
    RANKING_TIE_POLICY,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root


THRESHOLD_METRIC = "macro_by_user_group_mean_f1"


def coverage_summary(
    *,
    X_split: pd.DataFrame,
    y_split: pd.Series,
    scores: pd.Series,
    X_train: pd.DataFrame,
    user_column: str = "user_id",
    item_column: str = "item_id",
) -> dict[str, int | float]:
    """Diagnose how much of a split a train-fit CF scorer can actually score.

    Collaborative baselines return 0 for users or items unseen in training (and
    for warm items with no similar profile item). LLM/popularity scorers score
    every row, so this asymmetry is part of the "evaluation-setup-specific"
    story and is recorded rather than hidden. ``nonzero_score_*`` counts rows
    that received a signal; ``positive_nonzero_score_fraction`` restricts that to
    true positives (the rows the baseline most needs to score to rank well).
    """

    train_users = set(X_train[user_column])
    train_items = set(X_train[item_column])
    n = int(len(X_split))
    cold_user = ~X_split[user_column].isin(train_users)
    cold_item = ~X_split[item_column].isin(train_items)
    nonzero = scores.to_numpy() != 0
    positive = y_split.to_numpy().astype(bool)
    n_positive = int(positive.sum())
    return {
        "rows": n,
        "cold_user_rows": int(cold_user.sum()),
        "cold_item_rows": int(cold_item.sum()),
        "nonzero_score_rows": int(nonzero.sum()),
        "nonzero_score_fraction": float(nonzero.mean()) if n else 0.0,
        "positive_rows": n_positive,
        "positive_nonzero_score_fraction": (
            float((nonzero & positive).sum() / n_positive) if n_positive else 0.0
        ),
    }


def run_classical_pointwise(
    task: Task,
    output_dir: Path,
    *,
    scorer: Scorer,
    method_name: str,
    scorer_manifest: dict[str, Any],
    log_tag: str,
) -> dict[str, object]:
    """Fit a classical scorer on train, threshold on val, evaluate pointwise."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError(f"{method_name} method requires candidate_group_column")
    _record_stage(stage_times, "prepare_xy", stage_start, log_tag)

    stage_start = perf_counter()
    scorer = scorer.fit(X_train, y_train)
    _record_stage(stage_times, "fit", stage_start, log_tag)

    stage_start = perf_counter()
    val_scores = scorer.score(X_val)
    _record_stage(stage_times, "score_val", stage_start, log_tag)

    stage_start = perf_counter()
    test_scores = scorer.score(X_test)
    _record_stage(stage_times, "score_test", stage_start, log_tag)

    stage_start = perf_counter()
    threshold_selection = find_best_user_group_threshold(
        y_val,
        val_scores,
        X_val[candidate_group_column],
        X_val["user_id"],
        metric="f1",
    )
    threshold = float(threshold_selection["threshold"])
    _record_stage(stage_times, "select_threshold", stage_start, log_tag)

    stage_start = perf_counter()
    val_predictions = apply_threshold(val_scores, threshold)
    test_predictions = apply_threshold(test_scores, threshold)
    _record_stage(stage_times, "apply_threshold", stage_start, log_tag)

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
    _record_stage(stage_times, "compute_metrics", stage_start, log_tag)

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
    val_coverage = coverage_summary(
        X_split=X_val, y_split=y_val, scores=val_scores, X_train=X_train,
    )
    test_coverage = coverage_summary(
        X_split=X_test, y_split=y_test, scores=test_scores, X_train=X_train,
    )
    _record_stage(stage_times, "candidate_group_summary", stage_start, log_tag)

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
    _record_stage(stage_times, "write_predictions", stage_start, log_tag)

    root = repo_root()
    manifest = {
        "method": method_name,
        "scorer": scorer_manifest,
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
        "coverage": {
            "val": val_coverage,
            "test": test_coverage,
        },
        "stage_times_seconds": stage_times,
        "git_commit": current_git_commit(root),
    }
    metrics = {
        "method": method_name,
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
        "coverage": {
            "val": val_coverage,
            "test": test_coverage,
        },
        "stage_times_seconds": stage_times,
    }
    stage_start = perf_counter()
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / POINTWISE_METRICS_FILENAME, metrics)
    _record_stage(stage_times, "write_metadata", stage_start, log_tag)
    return metrics


def run_classical_ranking(
    task: Task,
    output_dir: Path,
    *,
    scorer: Scorer,
    method_name: str,
    scorer_manifest: dict[str, Any],
    log_tag: str,
) -> dict[str, object]:
    """Fit a classical scorer on train, evaluate raw-score ranking per group."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError(f"{method_name} method requires candidate_group_column")
    _record_stage(stage_times, "prepare_xy", stage_start, log_tag)

    stage_start = perf_counter()
    scorer = scorer.fit(X_train, y_train)
    _record_stage(stage_times, "fit", stage_start, log_tag)

    stage_start = perf_counter()
    val_scores = scorer.score(X_val)
    _record_stage(stage_times, "score_val", stage_start, log_tag)

    stage_start = perf_counter()
    test_scores = scorer.score(X_test)
    _record_stage(stage_times, "score_test", stage_start, log_tag)

    stage_start = perf_counter()
    val_ranking_metrics = ranking_metrics_for_split(
        X=X_val, y=y_val, scores=val_scores,
        candidate_group_column=candidate_group_column,
    )
    test_ranking_metrics = ranking_metrics_for_split(
        X=X_test, y=y_test, scores=test_scores,
        candidate_group_column=candidate_group_column,
    )
    _record_stage(stage_times, "compute_ranking_metrics", stage_start, log_tag)

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
    val_coverage = coverage_summary(
        X_split=X_val, y_split=y_val, scores=val_scores, X_train=X_train,
    )
    test_coverage = coverage_summary(
        X_split=X_test, y_split=y_test, scores=test_scores, X_train=X_train,
    )
    _record_stage(stage_times, "candidate_group_summary", stage_start, log_tag)

    stage_start = perf_counter()
    predictions = pd.concat(
        [
            score_frame(split="val", X=X_val, y=y_val, scores=val_scores),
            score_frame(split="test", X=X_test, y=y_test, scores=test_scores),
        ],
        ignore_index=True,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _record_stage(stage_times, "write_predictions", stage_start, log_tag)

    root = repo_root()
    manifest = {
        "method": method_name,
        "protocol": "ranking",
        "scorer": scorer_manifest,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
        },
        "task": {
            "name": task.name,
            "manifest": task.manifest,
        },
        "candidate_groups": {
            "val": val_candidate_groups,
            "test": test_candidate_groups,
        },
        "coverage": {
            "val": val_coverage,
            "test": test_coverage,
        },
        "stage_times_seconds": stage_times,
        "git_commit": current_git_commit(root),
    }
    metrics = {
        "method": method_name,
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
        "coverage": {
            "val": val_coverage,
            "test": test_coverage,
        },
        "stage_times_seconds": stage_times,
    }
    stage_start = perf_counter()
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / RANKING_METRICS_FILENAME, metrics)
    _record_stage(stage_times, "write_metadata", stage_start, log_tag)
    return metrics


def _record_stage(
    stage_times: dict[str, float], stage: str, start: float, log_tag: str
) -> None:
    seconds = round(perf_counter() - start, 3)
    stage_times[stage] = seconds
    print(f"[{log_tag}] {stage}: {seconds}s", flush=True)
