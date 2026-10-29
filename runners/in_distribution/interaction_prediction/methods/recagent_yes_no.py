"""Isolated MovieLens RecAgent runner: one yes/no call supplies both evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers.recagent import MemoryConfig, RecAgentYesNoScorer
from beyond_click_sim.scorers.recagent.profiles import (
    CANDIDATE_COLUMNS,
    HISTORY_COLUMNS,
)
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods._llmsrec_syn_data import (
    prepare_training,
    require_temporal_task,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    candidate_group_summary,
    current_git_commit,
    failure_as_negative_pointwise_metrics,
    limit_candidate_groups,
    pointwise_metrics_for_split,
    prediction_frame,
    ranking_metrics_for_split,
    ranking_metrics_with_failed_groups_as_zero,
    score_coverage_summary,
    task_xy,
    write_json,
)
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    _score_groups,
    _serving_metadata,
)
from runners.in_distribution.interaction_prediction.metrics import (
    POINTWISE_MAIN_METRIC,
    RANKING_MAIN_METRIC,
    RANKING_KS,
    RANKING_TIE_POLICY,
)
from runners.in_distribution.interaction_prediction.task_builders import (
    load_canonical_dataset,
    repo_root,
)
from runners.in_distribution.item_summaries import task_item_summary_metadata
from runners.in_distribution.llm_item_stats import maybe_add_item_rating_prompt_columns


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str = "recagent_yes_no",
    client_name: str = "litellm_local",
    model: str = "Qwen/Qwen3-8B",
    model_revision: str | None = None,
    canonical_dataset=None,
    client=None,
    embedding_provider=None,
    embedding_revision: str = "main",
    memory_config: MemoryConfig | None = None,
    use_demographics: bool = True,
    use_item_stats: bool = False,
    candidate_item_summaries: bool = True,
    max_history_items: int | None = None,
    max_candidate_groups: int | None = None,
    max_llm_attempts: int = 5,
    max_workers: int = 1,
    seed: int = 2023,
    temperature: float = 0.0,
    max_tokens: int = 256,
    importance_max_tokens: int = 16,
    extra_body: dict | None = None,
    cache_dir: Path | None = None,
    preflight_only: bool = False,
) -> dict:
    if task.manifest.get("dataset") != "ml-1m":
        raise ValueError("RecAgent supports MovieLens-1M interaction tasks only")
    require_temporal_task(task)
    if max_llm_attempts < 1 or max_workers < 1:
        raise ValueError("Attempts and workers must be positive")
    prepared = prepare_training(
        task, canonical_dataset or load_canonical_dataset("ml-1m")
    )
    summaries = task_item_summary_metadata(task, candidate=candidate_item_summaries)
    columns = maybe_add_item_rating_prompt_columns(
        "ml-1m",
        {
            "history_description_columns": HISTORY_COLUMNS,
            "candidate_description_columns": (
                *CANDIDATE_COLUMNS,
                *(("item_summary",) if candidate_item_summaries else ()),
            ),
        },
        use_item_stats=use_item_stats,
    )
    if (
        use_item_stats
        and (task.manifest.get("item_feature_builder") or {}).get("class")
        != "TrainItemRatingStatistics"
    ):
        raise ValueError("RecAgent item statistics must have training-only provenance")
    group_column = task.schema.candidate_group_column
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = limit_candidate_groups(
        *xy["test"],
        candidate_group_column=group_column,
        max_candidate_groups=max_candidate_groups,
    )
    if X_test.empty:
        raise ValueError("RecAgent requires at least one test candidate group")
    for column in (
        "user_id",
        "item_id",
        group_column,
        *columns["candidate_description_columns"],
    ):
        if column not in X_test or X_test[column].isna().any():
            raise ValueError(f"Missing test identity or candidate metadata: {column}")
    output_dir = Path(output_dir)
    cache_dir = (
        Path(cache_dir) if cache_dir is not None else output_dir / "recagent_cache"
    )
    model_revision = model_revision or os.environ.get(
        "BEYOND_CLICK_SIM_RECAGENT_MODEL_REVISION"
    )
    scorer = RecAgentYesNoScorer(
        client=client,
        model=model,
        model_revision=model_revision,
        training_events=prepared.events,
        candidate_group_column=group_column,
        **columns,
        embedding_provider=embedding_provider,
        embedding_revision=embedding_revision,
        memory_config=memory_config,
        use_demographics=use_demographics,
        max_history_items=max_history_items,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        importance_max_tokens=importance_max_tokens,
        max_initialization_attempts=max_llm_attempts,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        if extra_body is None
        else extra_body,
        cache_dir=cache_dir,
    ).fit(X_train, y_train, profile_user_ids=X_test.user_id.drop_duplicates().tolist())
    preflight = {
        **scorer.preflight(candidate_groups=X_test[group_column].nunique()),
        "provenance": prepared.provenance,
        "item_summaries": summaries,
        "candidate_rows": len(X_test),
        "method": method_name,
        "task": task.name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "preflight.json", preflight)
    if preflight_only:
        return preflight
    if client is None:
        scorer.client = make_llm_client(client_name)
    # Append immediately so an interrupted initialization still has a stage trace.
    with (output_dir / "recagent_trace.jsonl").open("a", encoding="utf-8") as trace:

        def emit(record):
            trace.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            trace.flush()

        scorer.diagnostic_sink = emit
        scorer.build_agents()
        scores, errors = _score_groups(
            scorer,
            X_test,
            candidate_group_column=group_column,
            max_attempts=max_llm_attempts,
            max_workers=max_workers,
        )
        initialization = scorer.initialization_metadata()
    valid = scores.notna()
    predictions = scores.astype("boolean").rename("prediction")
    prediction_frame(
        split="test", X=X_test, y=y_test, scores=scores, predictions=predictions
    ).to_parquet(output_dir / "predictions.parquet", index=False)
    with (output_dir / "llm_errors.jsonl").open("w", encoding="utf-8") as file:
        for user, error in scorer.initialization_errors.items():
            file.write(
                json.dumps({"stage": "initialization", "user_id": user, "error": error})
                + "\n"
            )
        for error in errors:
            file.write(json.dumps({"stage": "group", **error}) + "\n")
    coverage = score_coverage_summary(scores)
    groups = {
        "requested": candidate_group_summary(
            X_test,
            y_test,
            candidate_group_column=group_column,
            sampled_column=task.schema.sampled_column,
        ),
        "scored": candidate_group_summary(
            X_test.loc[valid],
            y_test.loc[valid],
            candidate_group_column=group_column,
            sampled_column=task.schema.sampled_column,
        ),
    }
    parsed = (
        pointwise_metrics_for_split(
            X=X_test.loc[valid],
            y=y_test.loc[valid],
            predictions=predictions.loc[valid].astype(bool),
            candidate_group_column=group_column,
        )
        if valid.any()
        else None
    )
    ranking = (
        ranking_metrics_for_split(
            X=X_test.loc[valid],
            y=y_test.loc[valid],
            scores=scores.loc[valid],
            candidate_group_column=group_column,
        )
        if valid.any()
        else None
    )
    common = {
        "method": method_name,
        "task": task.name,
        "coverage": coverage,
        "candidate_groups": groups,
        "llm_errors": len(errors),
        "scored_rows": int(valid.sum()),
        "requested_rows": len(X_test),
        "max_candidate_groups": max_candidate_groups,
        "max_workers": max_workers,
    }
    result = {
        **common,
        "main_metric": POINTWISE_MAIN_METRIC,
        "test": parsed,
        "test_failure_as_negative": failure_as_negative_pointwise_metrics(
            X=X_test, y=y_test, scores=scores, candidate_group_column=group_column
        ),
    }
    ranking_result = {
        **common,
        "protocol": "ranking",
        "main_metric": RANKING_MAIN_METRIC,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
        },
        "test": ranking,
        "test_failure_as_zero_group": ranking_metrics_with_failed_groups_as_zero(
            X=X_test, y=y_test, scores=scores, candidate_group_column=group_column
        ),
    }
    manifest = {
        "method": method_name,
        "status": "complete" if valid.any() else "all_groups_failed",
        "scorer": {
            "class": "RecAgentYesNoScorer",
            "client_name": client_name,
            **scorer.configuration,
            "item_summaries": summaries,
            "uses_item_stats": use_item_stats,
            "serving": _serving_metadata() if client_name == "litellm_local" else None,
        },
        "initialization": initialization,
        "training_provenance": prepared.provenance,
        "task": {"name": task.name, "manifest": task.manifest},
        "decision_rule": {"kind": "hard_binary_yes_no_parser", "threshold": None},
        "ranking_scores": "same_binary_scores_average_ties",
        "validation": "unused_fixed_configuration_no_threshold_search",
        "limits": {
            "max_candidate_groups": max_candidate_groups,
            "max_workers": max_workers,
            "max_llm_attempts": max_llm_attempts,
        },
        "coverage": coverage,
        "candidate_groups": groups,
        "git_commit": current_git_commit(repo_root()),
        "claim": "Reimplementation of RecAgent's profile–memory–action response architecture using available author-code details, adapted to offline interaction prediction.",
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "metrics.json", result)
    write_json(output_dir / "metrics_ranking.json", ranking_result)
    if not valid.any():
        raise RuntimeError(
            "RecAgent produced no valid scores; predictions, diagnostics and failure-inclusive metrics were saved"
        )
    return result


def run_qwen3_8b_smoke(task: Task, output_dir: Path) -> dict:
    return run_method(
        task,
        output_dir,
        method_name="recagent_yes_no_litellm_qwen3_8b_smoke",
        max_candidate_groups=5,
    )


def run_qwen3_8b_full(task: Task, output_dir: Path) -> dict:
    return run_method(
        task, output_dir, method_name="recagent_yes_no_litellm_qwen3_8b_full"
    )


def run_qwen36_27b_smoke(task: Task, output_dir: Path) -> dict:
    return run_method(
        task,
        output_dir,
        method_name="recagent_yes_no_litellm_qwen36_27b_smoke",
        model="Qwen/Qwen3.6-27B",
        max_candidate_groups=5,
    )


def run_qwen36_27b_full(task: Task, output_dir: Path) -> dict:
    return run_method(
        task,
        output_dir,
        method_name="recagent_yes_no_litellm_qwen36_27b_full",
        model="Qwen/Qwen3.6-27B",
    )


def main():
    from runners.in_distribution.interaction_prediction.task_builders import (
        TASK_BUILDERS,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_BUILDERS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate and count only; no inference or model loading",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model-revision")
    parser.add_argument("--embedding-revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-history-items", type=int)
    parser.add_argument("--no-demographics", action="store_true")
    parser.add_argument("--no-candidate-summaries", action="store_true")
    args = parser.parse_args()
    result = run_method(
        TASK_BUILDERS[args.task](),
        args.output_dir,
        model=args.model,
        model_revision=args.model_revision,
        embedding_revision=args.embedding_revision,
        cache_dir=args.cache_dir,
        max_candidate_groups=args.max_groups,
        max_history_items=args.max_history_items,
        use_demographics=not args.no_demographics,
        candidate_item_summaries=not args.no_candidate_summaries,
        preflight_only=args.preflight,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
