"""MovieLens SimUSER(zero, text-only), with combined validation/test outputs."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import pandas as pd

from beyond_click_sim.data.canonical import sha256_file
from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers.simuser import SimUSERConfig, SimUSERYesNoScorer
from beyond_click_sim.scorers.simuser.profiles import STATIC_COLUMNS
from beyond_click_sim.scorers.simuser.state import identity
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


def _catalog(dataset, prepared):
    items = dataset.load_items()
    if (
        sha256_file(dataset.items_path)
        != prepared.provenance["canonical_file_sha256"]["items"]
    ):
        raise ValueError("Catalog changed after training verification")
    permitted = {identity(v) for v in prepared.catalog.item_id}
    catalog = items.loc[items.item_id.map(identity).isin(permitted)].rename(
        columns={
            "title": "item_title",
            "genres": "item_genres",
            "summary": "item_summary",
        }
    )
    required = ["item_id", *STATIC_COLUMNS]
    if not set(required) <= set(catalog) or len(catalog) != len(permitted):
        raise ValueError(
            "Canonical catalog must supply every permitted item's title, genres and summary"
        )
    source = (
        json.loads(dataset.manifest_path.read_text())
        .get("item_enrichment", {})
        .get("movie_summaries")
    )
    return catalog[required].copy(), {
        "kind": "frozen_canonical_catalog",
        "canonical_items_sha256": prepared.provenance["canonical_file_sha256"]["items"],
        "summary_artifact": source,
        "genres": "canonical",
        "quality": "mean observed rating over complete permitted training partition",
        "popularity": "observed training interaction count",
    }


def _enrich(X, catalog):
    """ID-checked static metadata on a private copy; discard no supplied rows."""
    result = X.copy(deep=True)
    lookup = catalog.set_index(catalog.item_id.map(identity))
    keys = X.item_id.map(identity)
    if not keys.isin(lookup.index).all():
        raise ValueError("Task item absent from permitted catalog")
    for column in STATIC_COLUMNS:
        result[column] = keys.map(lookup[column]).to_numpy()
    return result


def run_method(
    task,
    output_dir,
    *,
    method_name="simuser_zero_text_only",
    client_name="litellm_local",
    model="Qwen/Qwen3-8B",
    model_revision=None,
    client=None,
    embedding_client=None,
    embedding_client_name="openai",
    embedding_model="text-embedding-3-small",
    embedding_identity=None,
    canonical_dataset=None,
    config=None,
    seed=None,
    max_candidate_groups=None,
    max_llm_attempts=5,
    max_workers=1,
    splits=("val", "test"),
    extra_body=None,
    cache_dir=None,
    preflight_only=False,
):
    if task.manifest.get("dataset") != "ml-1m":
        raise ValueError("SimUSER text-only zero supports MovieLens-1M only")
    require_temporal_task(task)
    if not 1 <= max_llm_attempts <= 5 or max_workers < 1:
        raise ValueError("Use one to five group attempts and positive workers")
    if (
        not splits
        or len(set(splits)) != len(splits)
        or not set(splits) <= {"val", "test"}
    ):
        raise ValueError("Requested splits must be distinct val/test entries")
    dataset = canonical_dataset or load_canonical_dataset("ml-1m")
    prepared = prepare_training(task, dataset)
    catalog, metadata = _catalog(dataset, prepared)
    group_column = task.schema.candidate_group_column
    xy = task_xy(task)
    selected = {
        s: limit_candidate_groups(
            *xy[s],
            candidate_group_column=group_column,
            max_candidate_groups=max_candidate_groups,
        )
        for s in splits
    }
    if any(X.empty for X, _ in selected.values()):
        raise ValueError(
            "Each requested split must contain at least one candidate group"
        )
    # Preserve originals for artifacts/evaluation; pass a strict identity allowlist to score.
    inputs = {s: _enrich(X, catalog) for s, (X, _) in selected.items()}
    users = pd.concat([X.user_id for X in inputs.values()]).drop_duplicates().tolist()
    X_train, y_train = xy["train"]
    if X_train[task.schema.sampled_column].fillna(False).any():
        raise ValueError("Sampled training rows cannot initialize SimUSER")
    if config is not None and seed is not None and config.seed != seed:
        raise ValueError("Explicit seed disagrees with SimUSERConfig.seed")
    if seed is None:
        seed = task.manifest.get("sampler", {}).get(
            "seed", task.manifest["splitter"].get("seed", 2023)
        )
    config = config or SimUSERConfig(seed=seed)
    context_env = os.environ.get("BEYOND_CLICK_SIM_VLLM_MAX_MODEL_LEN")
    if (
        config.context_tokens is None
        and client_name.startswith(("litellm", "vllm"))
        and context_env
    ):
        config = replace(config, context_tokens=int(context_env))
    model_revision = model_revision or os.environ.get(
        "BEYOND_CLICK_SIM_SIMUSER_MODEL_REVISION"
    )
    scorer = SimUSERYesNoScorer(
        client=client,
        model=model,
        model_revision=model_revision,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        embedding_identity=embedding_identity
        or f"{embedding_client_name}:{embedding_model}",
        config=config,
        training_events=prepared.events,
        catalog=catalog,
        candidate_group_column=group_column,
        cache_dir=Path(cache_dir)
        if cache_dir is not None
        else Path(output_dir) / "simuser_cache",
        extra_body=(
            {"chat_template_kwargs": {"enable_thinking": False}}
            if extra_body is None and "qwen" in model.lower()
            else extra_body
        ),
    ).fit(_enrich(X_train, catalog), y_train, profile_user_ids=users)
    group_count, max_candidates = 0, 0
    for X in inputs.values():
        for _, group in X.groupby(group_column, sort=False):
            scorer.validate_candidates(group)
            group_count += 1
            max_candidates = max(max_candidates, len(group))
    preflight = {
        **scorer.preflight(candidate_groups=group_count, max_candidates=max_candidates),
        "training_provenance": prepared.provenance,
        "metadata": metadata,
        "splits": {
            s: {"rows": len(X), "groups": X[group_column].nunique()}
            for s, X in inputs.items()
        },
        "method": method_name,
        "task": task.name,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "preflight.json", preflight)
    if preflight_only:
        return preflight
    if client is None:
        scorer.client = make_llm_client(client_name)
    if embedding_client is None:
        scorer.embedding_client = make_llm_client(embedding_client_name)
    pointwise = {
        "method": method_name,
        "task": task.name,
        "main_metric": POINTWISE_MAIN_METRIC,
    }
    ranking = {
        "method": method_name,
        "task": task.name,
        "protocol": "ranking",
        "main_metric": RANKING_MAIN_METRIC,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
        },
    }
    frames, errors, coverage, groups = [], [], {}, {}
    try:
        with (output_dir / "simuser_trace.jsonl").open("a", encoding="utf-8") as trace:

            def emit(record):
                trace.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                trace.flush()

            scorer.diagnostic_sink = emit
            scorer.build_agents()
            for split in splits:
                X, y = selected[split]
                allowed = inputs[split][["user_id", "item_id", group_column]]
                scores, failures = _score_groups(
                    scorer,
                    allowed,
                    candidate_group_column=group_column,
                    max_attempts=max_llm_attempts,
                    max_workers=max_workers,
                )
                errors.extend({"split": split, **r} for r in failures)
                valid = scores.notna()
                predictions = scores.astype("boolean").rename("prediction")
                frame = prediction_frame(
                    split=split, X=X, y=y, scores=scores, predictions=predictions
                )
                frame["task_row_index"] = X.index
                frames.append(frame)
                coverage[split] = score_coverage_summary(scores)
                groups[split] = {
                    "requested": candidate_group_summary(
                        X,
                        y,
                        candidate_group_column=group_column,
                        sampled_column=task.schema.sampled_column,
                    ),
                    "scored": candidate_group_summary(
                        X.loc[valid],
                        y.loc[valid],
                        candidate_group_column=group_column,
                        sampled_column=task.schema.sampled_column,
                    ),
                }
                pointwise[split] = (
                    pointwise_metrics_for_split(
                        X=X.loc[valid],
                        y=y.loc[valid],
                        predictions=predictions.loc[valid].astype(bool),
                        candidate_group_column=group_column,
                    )
                    if valid.any()
                    else None
                )
                ranking[split] = (
                    ranking_metrics_for_split(
                        X=X.loc[valid],
                        y=y.loc[valid],
                        scores=scores.loc[valid],
                        candidate_group_column=group_column,
                    )
                    if valid.any()
                    else None
                )
                pointwise[f"{split}_failure_as_negative"] = (
                    failure_as_negative_pointwise_metrics(
                        X=X, y=y, scores=scores, candidate_group_column=group_column
                    )
                )
                ranking[f"{split}_failure_as_zero_group"] = (
                    ranking_metrics_with_failed_groups_as_zero(
                        X=X, y=y, scores=scores, candidate_group_column=group_column
                    )
                )
            initialization = scorer.initialization_metadata()
    finally:
        scorer.close()
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(output_dir / "predictions.parquet", index=False)
    with (output_dir / "llm_errors.jsonl").open("w", encoding="utf-8") as file:
        for user, error in scorer.initialization_errors.items():
            file.write(
                json.dumps({"stage": "initialization", "user_id": user, "error": error})
                + "\n"
            )
        for error in errors:
            file.write(json.dumps({"stage": "group", **error}) + "\n")
    for result in (pointwise, ranking):
        result.update(
            coverage=coverage, candidate_groups=groups, llm_errors=len(errors)
        )
    complete = bool(combined.score.notna().all())
    manifest = {
        "method": method_name,
        "status": "complete"
        if complete
        else "partial_failure"
        if combined.score.notna().any()
        else "all_groups_failed",
        "scorer": {
            "class": "SimUSERYesNoScorer",
            **scorer.configuration,
            "client_name": client_name,
            "embedding_client_name": embedding_client_name,
            "serving": _serving_metadata() if client_name == "litellm_local" else None,
        },
        "initialization": initialization,
        "cache_directory": str(scorer.cache_dir.resolve()),
        "training_provenance": prepared.provenance,
        "metadata": metadata,
        "task": {"name": task.name, "manifest": task.manifest},
        "decision_rule": {"kind": "native_watch_skip", "threshold": None},
        "ranking_scores": "same_binary_scores_average_ties",
        "validation": "exposed_for_external_per_seed_selection_on_macro_by_user_group_mean_ndcg@5; no search controller",
        "successful_metrics_equal_strict": complete,
        "coverage": coverage,
        "candidate_groups": groups,
        "limits": {
            "max_candidate_groups_per_split": max_candidate_groups,
            "max_workers": max_workers,
            "max_llm_attempts": max_llm_attempts,
        },
        "git_commit": current_git_commit(repo_root()),
        "runner_sha256": sha256_file(Path(__file__)),
        "claim": "Text-only, history-initialized adaptation of SimUSER persona matching, dual-memory retrieval and iterative watch/skip response; underspecified details reconstructed.",
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "metrics.json", pointwise)
    write_json(output_dir / "metrics_ranking.json", ranking)
    if not combined.score.notna().any():
        raise RuntimeError(
            "SimUSER produced no valid scores; missing outputs and failure-inclusive artifacts saved"
        )
    return pointwise


def run_qwen3_8b_smoke(task, output_dir):
    return run_method(
        task,
        output_dir,
        method_name="simuser_zero_text_only_litellm_qwen3_8b_smoke",
        max_candidate_groups=5,
    )


def run_qwen3_8b_full(task, output_dir):
    return run_method(
        task, output_dir, method_name="simuser_zero_text_only_litellm_qwen3_8b_full"
    )


def run_qwen36_27b_smoke(task, output_dir):
    return run_method(
        task,
        output_dir,
        method_name="simuser_zero_text_only_litellm_qwen36_27b_smoke",
        model="Qwen/Qwen3.6-27B",
        max_candidate_groups=5,
    )


def run_qwen36_27b_full(task, output_dir):
    return run_method(
        task,
        output_dir,
        method_name="simuser_zero_text_only_litellm_qwen36_27b_full",
        model="Qwen/Qwen3.6-27B",
    )


def main():
    from runners.in_distribution.interaction_prediction.task_builders import (
        TASK_BUILDERS,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_BUILDERS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model-revision")
    parser.add_argument("--client", default="litellm_local")
    parser.add_argument("--embedding-client", default="openai")
    parser.add_argument("--embedding-identity")
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--seed", type=int, help="Defaults to the materialized task's sampling seed"
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("val", "test"), default=["val", "test"]
    )
    args = parser.parse_args()
    task = TASK_BUILDERS[args.task]()
    seed = (
        args.seed
        if args.seed is not None
        else task.manifest.get("sampler", {}).get(
            "seed", task.manifest["splitter"].get("seed", 2023)
        )
    )
    result = run_method(
        task,
        args.output_dir,
        client_name=args.client,
        model=args.model,
        model_revision=args.model_revision,
        embedding_client_name=args.embedding_client,
        embedding_identity=args.embedding_identity,
        config=SimUSERConfig(seed=seed, context_tokens=args.context_tokens),
        cache_dir=args.cache_dir,
        max_candidate_groups=args.max_groups,
        max_workers=args.max_workers,
        splits=tuple(args.splits),
        preflight_only=args.preflight,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
