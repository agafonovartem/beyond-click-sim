"""Preference-local direct-listwise evaluation protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from beyond_click_sim.evaluation import apply_threshold, find_best_user_group_threshold
from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.preference_prediction.methods._grouped_llm_yes_no import (
    _score_groups,
)
from runners.in_distribution.preference_prediction.methods.common import (
    candidate_group_summary,
    current_git_commit,
    pointwise_metrics_for_split,
    prediction_frame,
    ranking_metrics_for_split,
    ranking_metrics_with_failed_groups_as_zero,
    score_coverage_summary,
    write_json,
)
from runners.in_distribution.preference_prediction.metrics import (
    POINTWISE_MAIN_METRIC,
    POINTWISE_METRICS_FILENAME,
    RANKING_KS,
    RANKING_MAIN_METRIC,
    RANKING_METRICS_FILENAME,
    RANKING_TIE_POLICY,
)


THRESHOLD_METRIC = "macro_by_user_group_mean_f1"
SCORE_MAPPING = "score = n_candidates - rank"


def evaluate_listwise_scorer(
    *,
    task: Task,
    output_dir: Path,
    method_name: str,
    scorer: Scorer,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    candidate_group_column: str,
    max_candidate_groups: int | None,
    max_llm_attempts: int,
    max_workers: int,
    scorer_manifest: dict[str, object],
    repo_root: Path,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate direct listwise scores and a validation-selected threshold."""

    val_scores, val_errors = _score_groups(
        scorer,
        X_val,
        candidate_group_column=candidate_group_column,
        max_attempts=max_llm_attempts,
        max_workers=max_workers,
    )
    test_scores, test_errors = _score_groups(
        scorer,
        X_test,
        candidate_group_column=candidate_group_column,
        max_attempts=max_llm_attempts,
        max_workers=max_workers,
    )
    _write_errors(
        output_dir / "llm_errors.jsonl",
        [
            *({"split": "val", **error} for error in val_errors),
            *({"split": "test", **error} for error in test_errors),
        ],
    )
    val_valid = val_scores.notna()
    test_valid = test_scores.notna()
    if not val_valid.any():
        raise RuntimeError("Listwise scorer did not produce valid validation scores")
    if not test_valid.any():
        raise RuntimeError("Listwise scorer did not produce valid test scores")

    threshold_selection = find_best_user_group_threshold(
        y_val.loc[val_valid],
        val_scores.loc[val_valid],
        X_val.loc[val_valid, candidate_group_column],
        X_val.loc[val_valid, "user_id"],
        metric="f1",
    )
    threshold = float(threshold_selection["threshold"])
    val_predictions = _nullable_threshold_predictions(val_scores, threshold)
    test_predictions = _nullable_threshold_predictions(test_scores, threshold)

    pd.concat(
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
    ).to_parquet(output_dir / "predictions.parquet", index=False)
    val_pointwise = _pointwise_evaluation(
        X=X_val,
        y=y_val,
        predictions=val_predictions,
        candidate_group_column=candidate_group_column,
    )
    test_pointwise = _pointwise_evaluation(
        X=X_test,
        y=y_test,
        predictions=test_predictions,
        candidate_group_column=candidate_group_column,
    )
    val_ranking = _ranking_evaluation(
        X=X_val,
        y=y_val,
        scores=val_scores,
        candidate_group_column=candidate_group_column,
    )
    test_ranking = _ranking_evaluation(
        X=X_test,
        y=y_test,
        scores=test_scores,
        candidate_group_column=candidate_group_column,
    )
    candidate_groups = {
        "val": _candidate_group_evaluation(
            task=task,
            X=X_val,
            y=y_val,
            valid=val_valid,
            candidate_group_column=candidate_group_column,
        ),
        "test": _candidate_group_evaluation(
            task=task,
            X=X_test,
            y=y_test,
            valid=test_valid,
            candidate_group_column=candidate_group_column,
        ),
    }
    coverage = {
        "val": score_coverage_summary(val_scores),
        "test": score_coverage_summary(test_scores),
    }
    llm_errors = {"val": len(val_errors), "test": len(test_errors)}

    manifest: dict[str, object] = {
        "method": method_name,
        "protocol": "listwise_ranking_with_validation_threshold",
        "scorer": scorer_manifest,
        "decision_rule": {
            "ranking": {
                "kind": "complete_listwise_rank_parser",
                "score_mapping": SCORE_MAPPING,
            },
            "pointwise": {
                "kind": "threshold_on_validation",
                "selection_metric": THRESHOLD_METRIC,
                **threshold_selection,
            },
        },
        "limits": {
            "max_candidate_groups_per_split": max_candidate_groups,
            "max_llm_attempts": max_llm_attempts,
            "max_workers": max_workers,
        },
        "llm_errors": llm_errors,
        "candidate_groups": candidate_groups,
        "task": {"name": task.name, "manifest": task.manifest},
        "git_commit": current_git_commit(repo_root),
    }
    if source_metadata is not None:
        manifest["source"] = source_metadata

    pointwise_result = {
        "method": method_name,
        "task": task.name,
        "protocol": "listwise_scores_thresholded_on_validation",
        "main_metric": POINTWISE_MAIN_METRIC,
        "val": val_pointwise["parsed_only"],
        "val_failure_as_negative": val_pointwise["failure_as_negative"],
        "test": test_pointwise["parsed_only"],
        "test_failure_as_negative": test_pointwise["failure_as_negative"],
        "threshold": threshold,
        "threshold_metric": THRESHOLD_METRIC,
        "coverage": coverage,
        "llm_errors": llm_errors,
        "candidate_groups": candidate_groups,
    }
    ranking_result = {
        "method": method_name,
        "task": task.name,
        "protocol": "direct_listwise_ranking",
        "main_metric": RANKING_MAIN_METRIC,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
            "score_mapping": SCORE_MAPPING,
        },
        "val": val_ranking["parsed_only"],
        "val_failure_as_zero_group": val_ranking["failure_as_zero_group"],
        "test": test_ranking["parsed_only"],
        "test_failure_as_zero_group": test_ranking["failure_as_zero_group"],
        "pointwise_threshold": threshold,
        "coverage": coverage,
        "llm_errors": llm_errors,
        "candidate_groups": candidate_groups,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / POINTWISE_METRICS_FILENAME, pointwise_result)
    write_json(output_dir / RANKING_METRICS_FILENAME, ranking_result)
    return ranking_result


def _nullable_threshold_predictions(
    scores: pd.Series,
    threshold: float,
) -> pd.Series:
    return apply_threshold(scores, threshold).where(scores.notna()).astype("boolean")


def _pointwise_evaluation(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    predictions: pd.Series,
    candidate_group_column: str,
) -> dict[str, object]:
    valid = predictions.notna()
    return {
        "parsed_only": pointwise_metrics_for_split(
            X=X.loc[valid],
            y=y.loc[valid],
            predictions=predictions.loc[valid].astype(bool),
            candidate_group_column=candidate_group_column,
        ),
        "failure_as_negative": pointwise_metrics_for_split(
            X=X,
            y=y,
            predictions=predictions.fillna(False).astype(bool),
            candidate_group_column=candidate_group_column,
        ),
    }


def _ranking_evaluation(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
    candidate_group_column: str,
) -> dict[str, object]:
    valid = scores.notna()
    return {
        "parsed_only": ranking_metrics_for_split(
            X=X.loc[valid],
            y=y.loc[valid],
            scores=scores.loc[valid],
            candidate_group_column=candidate_group_column,
        ),
        "failure_as_zero_group": ranking_metrics_with_failed_groups_as_zero(
            X=X,
            y=y,
            scores=scores,
            candidate_group_column=candidate_group_column,
        ),
    }


def _candidate_group_evaluation(
    *,
    task: Task,
    X: pd.DataFrame,
    y: pd.Series,
    valid: pd.Series,
    candidate_group_column: str,
) -> dict[str, object]:
    return {
        "requested": candidate_group_summary(
            X,
            y,
            candidate_group_column=candidate_group_column,
            sampled_column=task.schema.sampled_column,
        ),
        "scored": candidate_group_summary(
            X.loc[valid],
            y.loc[valid],
            candidate_group_column=candidate_group_column,
            sampled_column=task.schema.sampled_column,
        ),
    }


def _write_errors(path: Path, errors: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, sort_keys=True) + "\n")
