"""Shared ranking-only evaluation for cold-start listwise scorers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.cold_start.metrics import (
    LISTWISE_MAIN_METRIC,
    RANKING_KS,
    RANKING_METRICS_FILENAME,
    RANKING_TIE_POLICY,
)
from runners.in_distribution.cold_start.methods.llm_yes_no import _score_groups
from runners.in_distribution.interaction_prediction.methods.common import (
    candidate_group_summary,
    cold_start_metadata,
    current_git_commit,
    ranking_metrics_for_split,
    ranking_metrics_with_failed_groups_as_zero,
    score_coverage_summary,
    score_frame,
    write_json,
)
from runners.in_distribution.llm_error_budget import LLMErrorRateExceededError


SCORE_MAPPING = "score = n_candidates - rank"
DEFAULT_PROTOCOL = "direct_listwise_preference_ranking"


def evaluate_listwise_scorer(
    *,
    task: ColdStartTask,
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
    max_error_rate: float,
    min_groups_before_check: int,
    scorer_manifest: dict[str, object],
    repo_root: Path,
    protocol: str = DEFAULT_PROTOCOL,
) -> dict[str, object]:
    """Evaluate a fitted listwise scorer without pointwise thresholding."""

    all_errors: list[dict[str, object]] = []
    try:
        val_scores, val_errors = _score_groups(
            scorer,
            X_val,
            candidate_group_column=candidate_group_column,
            max_attempts=max_llm_attempts,
            max_workers=max_workers,
            method_name=method_name,
            task_name=f"{task.name}:val",
            max_error_rate=max_error_rate,
            min_groups_before_check=min_groups_before_check,
        )
    except LLMErrorRateExceededError as error:
        _write_errors(
            output_dir / "llm_errors.jsonl",
            _label_split_errors("val", error.errors),
        )
        raise
    all_errors.extend(_label_split_errors("val", val_errors))

    try:
        test_scores, test_errors = _score_groups(
            scorer,
            X_test,
            candidate_group_column=candidate_group_column,
            max_attempts=max_llm_attempts,
            max_workers=max_workers,
            method_name=method_name,
            task_name=f"{task.name}:test",
            max_error_rate=max_error_rate,
            min_groups_before_check=min_groups_before_check,
        )
    except LLMErrorRateExceededError as error:
        all_errors.extend(_label_split_errors("test", error.errors))
        _write_errors(output_dir / "llm_errors.jsonl", all_errors)
        raise
    all_errors.extend(_label_split_errors("test", test_errors))

    _write_errors(output_dir / "llm_errors.jsonl", all_errors)
    val_valid = val_scores.notna()
    test_valid = test_scores.notna()
    if not val_valid.any():
        raise RuntimeError("Listwise scorer did not produce valid validation scores")
    if not test_valid.any():
        raise RuntimeError("Listwise scorer did not produce valid test scores")

    pd.concat(
        [
            score_frame(
                split="val",
                X=X_val,
                y=y_val,
                scores=val_scores,
                metadata=cold_start_metadata(task),
            ),
            score_frame(
                split="test",
                X=X_test,
                y=y_test,
                scores=test_scores,
                metadata=cold_start_metadata(task),
            ),
        ],
        ignore_index=True,
    ).to_parquet(output_dir / "predictions.parquet", index=False)

    val_ranking = _ranking_metrics(
        X=X_val,
        y=y_val,
        scores=val_scores,
        candidate_group_column=candidate_group_column,
    )
    test_ranking = _ranking_metrics(
        X=X_test,
        y=y_test,
        scores=test_scores,
        candidate_group_column=candidate_group_column,
    )
    candidate_groups = {
        "val": _candidate_groups(
            task=task,
            X=X_val,
            y=y_val,
            valid=val_valid,
            candidate_group_column=candidate_group_column,
        ),
        "test": _candidate_groups(
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

    manifest = {
        "method": method_name,
        "protocol": protocol,
        "scorer": scorer_manifest,
        "decision_rule": {
            "ranking": {
                "kind": "complete_listwise_rank_parser",
                "score_mapping": SCORE_MAPPING,
            }
        },
        "limits": {
            "max_candidate_groups_per_split": max_candidate_groups,
            "max_llm_attempts": max_llm_attempts,
            "max_workers": max_workers,
            "max_error_rate": max_error_rate,
            "min_groups_before_check": min_groups_before_check,
        },
        "llm_errors": llm_errors,
        "candidate_groups": candidate_groups,
        "task": {"name": task.name, "manifest": task.manifest},
        "git_commit": current_git_commit(repo_root),
    }
    ranking_result = {
        "method": method_name,
        "task": task.name,
        "protocol": protocol,
        "main_metric": LISTWISE_MAIN_METRIC,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
            "score_mapping": SCORE_MAPPING,
        },
        "val": val_ranking["parsed_only"],
        "val_failure_as_zero_group": val_ranking["failure_as_zero_group"],
        "test": test_ranking["parsed_only"],
        "test_failure_as_zero_group": test_ranking["failure_as_zero_group"],
        "coverage": coverage,
        "llm_errors": llm_errors,
        "candidate_groups": candidate_groups,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / RANKING_METRICS_FILENAME, ranking_result)
    return ranking_result


def _ranking_metrics(
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


def _candidate_groups(
    *,
    task: ColdStartTask,
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


def _label_split_errors(
    split: str, errors: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [{"split": split, **error} for error in errors]


def _write_errors(path: Path, errors: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, sort_keys=True) + "\n")
