"""Opt-in DyTA4Rec: one ensemble for ranking and strict binary evaluation."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from beyond_click_sim.evaluation.binary import _user_group_threshold_metric_values
from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers.dyta4rec import DyTA4RecConfig, DyTA4RecScorer
from beyond_click_sim.scorers.dyta4rec.state import atomic_json, digest
from ._llmsrec_syn_data import prepare_training, require_temporal_task
from .common import (
    candidate_group_summary,
    current_git_commit,
    limit_candidate_groups,
    pointwise_metrics_for_split,
    prediction_frame,
    ranking_metrics_for_split,
    ranking_metrics_with_failed_groups_as_zero,
    score_coverage_summary,
    task_xy,
    write_json,
)
from .llm_yes_no import _score_groups, _serving_metadata
from ..metrics import (
    POINTWISE_MAIN_METRIC,
    RANKING_MAIN_METRIC,
    RANKING_KS,
    RANKING_TIE_POLICY,
)
from ..task_builders import load_canonical_dataset, repo_root


def select_threshold(X, y, scores, group_column):
    """Keep every group in the unchanged metric sweep's denominator.

    A -inf internal sentinel fixes failed rows to negative at every permitted
    threshold. Its transition is excluded. It is never saved or ranked.
    """
    if not scores.notna().any():
        return {
            "threshold": None,
            "metric": "macro_by_user_group_mean_f1",
            "metric_value": None,
            "status": "no_valid_validation_groups",
        }
    if not np.isfinite(scores.dropna()).all():
        raise ValueError("DyTA4Rec scores must be finite")
    thresholds, values = _user_group_threshold_metric_values(
        y, scores.fillna(-np.inf), X[group_column], X.user_id, metric="f1"
    )
    valid = np.isfinite(thresholds)
    thresholds, values = thresholds[valid], values[valid]
    best = int(np.argmax(values))
    return {
        "threshold": float(thresholds[best]),
        "metric": "macro_by_user_group_mean_f1",
        "metric_value": float(values[best]),
        "status": "selected",
        "population": "all_requested_validation_groups; missing_outputs_all_negative",
        "tie_convention": "all_negative_endpoint_first_then_ascending_score_transitions",
        "candidate_thresholds": len(thresholds),
    }


def _predictions(scores, threshold):
    if threshold is None:
        return pd.Series(pd.NA, index=scores.index, dtype="boolean", name="prediction")
    return (
        (scores >= threshold)
        .where(scores.notna())
        .astype("boolean")
        .rename("prediction")
    )


def run_method(
    task,
    output_dir,
    *,
    method_name=None,
    client_name="litellm_local",
    model="Qwen/Qwen3-8B",
    model_revision=None,
    canonical_dataset=None,
    client=None,
    embedding_provider=None,
    embedding_revision="main",
    config=None,
    max_candidate_groups=None,
    max_llm_attempts=5,
    max_workers=1,
    extra_body=None,
    cache_dir=None,
    preflight_only=False,
    serving_metadata=None,
):
    if task.manifest.get("dataset") != "ml-1m":
        raise ValueError("DyTA4Rec supports only MovieLens-1M interaction tasks")
    require_temporal_task(task)
    if not 1 <= max_llm_attempts <= 5 or max_workers < 1:
        raise ValueError("Require 1–5 whole-group attempts and positive workers")
    config = config or DyTA4RecConfig()
    method_name = method_name or f"dyta4rec_offline_{config.fusion}"
    prepared = prepare_training(
        task, canonical_dataset or load_canonical_dataset("ml-1m")
    )
    group_column = task.schema.candidate_group_column
    xy = task_xy(task)
    pairs = {
        split: limit_candidate_groups(
            *xy[split],
            candidate_group_column=group_column,
            max_candidate_groups=max_candidate_groups,
        )
        for split in ("val", "test")
    }
    if any(X.empty for X, _ in pairs.values()):
        raise ValueError(
            "DyTA4Rec requires nonempty supplied validation and test groups"
        )
    requested = (
        pd.concat([X.user_id for X, _ in pairs.values()]).drop_duplicates().tolist()
    )
    output_dir = Path(output_dir)
    cache_dir = (
        Path(cache_dir) if cache_dir is not None else output_dir / "dyta4rec_cache"
    )
    scorer = DyTA4RecScorer(
        client=client,
        model=model,
        model_revision=model_revision
        or os.environ.get("BEYOND_CLICK_SIM_DYTA4REC_MODEL_REVISION"),
        config=config,
        training_events=prepared.events,
        candidate_group_column=group_column,
        sampled_column=task.schema.sampled_column,
        embedding_provider=embedding_provider,
        embedding_revision=embedding_revision,
        extra_body=extra_body,
        cache_dir=cache_dir,
        serving_metadata={
            "client_name": client_name,
            **(
                serving_metadata
                if serving_metadata is not None
                else (_serving_metadata() if client_name == "litellm_local" else {})
            ),
        },
    ).fit(*xy["train"], profile_user_ids=requested)
    # Explicit allowlist: no held-out outcomes or evaluator-owned flags reach score.
    inputs = {
        s: X[["user_id", "item_id", group_column, "item_title", "item_genres"]].copy()
        for s, (X, _) in pairs.items()
    }
    for X in inputs.values():
        scorer.validate_candidates(X)
    preflight = {
        **scorer.preflight(
            candidate_groups=sum(X[group_column].nunique() for X in inputs.values()),
            candidate_frames=inputs.values(),
        ),
        "task": task.name,
        "method": method_name,
        "training_provenance": prepared.provenance,
        "requested_groups": {
            s: int(X[group_column].nunique()) for s, X in inputs.items()
        },
        "requested_rows": {s: len(X) for s, X in inputs.items()},
        "max_group_attempts": max_llm_attempts,
        "max_workers": max_workers,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "preflight.json", preflight)
    if preflight_only:
        return preflight
    identity = digest(
        {
            "configuration": scorer.configuration,
            "task": task.manifest,
            "training": prepared.provenance,
            "inputs": {s: X.to_json(orient="split") for s, X in inputs.items()},
            "validation_labels": pairs["val"][1].tolist(),
            "test_labels": pairs["test"][1].tolist(),
            "method": method_name,
            "retry_limit": max_llm_attempts,
        }
    )
    identity_path = output_dir / "evaluation_identity.json"
    if (
        identity_path.exists()
        and json.loads(identity_path.read_text())["identity"] != identity
    ):
        raise ValueError(
            "Output directory contains a different evaluation; use a new directory with a shared inference cache"
        )
    atomic_json(identity_path, {"identity": identity})
    if client is None:
        scorer.client = make_llm_client(client_name)
    errors, scores = [], {}
    with (output_dir / "dyta4rec_trace.jsonl").open("a", encoding="utf-8") as trace:

        def emit(record):
            trace.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            trace.flush()

        scorer.diagnostic_sink = emit
        scorer.build_agents()
        atomic_json(
            output_dir / "dyta4rec_snapshots.json",
            {u: s.to_dict() for u, s in scorer.agent_snapshots.items()},
        )
        for user, error in scorer.initialization_errors.items():
            errors.append({"stage": "initialization", "user_id": user, "error": error})
        for split, X in inputs.items():
            scorer._emit(event="split_started", split=split)
            scores[split], failures = _score_groups(
                scorer,
                X,
                candidate_group_column=group_column,
                max_attempts=max_llm_attempts,
                max_workers=max_workers,
            )
            errors.extend({"split": split, "stage": "group", **e} for e in failures)
    with (output_dir / "llm_errors.jsonl").open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error) + "\n")
    selection = select_threshold(*pairs["val"], scores["val"], group_column)
    threshold = selection["threshold"]
    frames, coverage, conversion_coverage, groups = [], {}, {}, {}
    pointwise, ranking = {}, {}
    for split, (X, y) in pairs.items():
        s = scores[split]
        valid = s.notna()
        prediction = _predictions(s, threshold)
        frame = prediction_frame(
            split=split, X=X, y=y, scores=s, predictions=prediction
        )
        frame["source_row_index"] = X.index
        frame["status"] = np.where(
            ~valid,
            "score_failed",
            "threshold_unavailable" if threshold is None else "ok",
        )
        frames.append(frame)
        coverage[split] = score_coverage_summary(s)
        conversion_coverage[split] = score_coverage_summary(prediction)
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
                predictions=prediction.loc[valid].astype(bool),
                candidate_group_column=group_column,
            )
            if valid.any() and threshold is not None
            else None
        )
        pointwise[f"{split}_failure_as_negative"] = (
            pointwise_metrics_for_split(
                X=X,
                y=y,
                predictions=prediction.fillna(False).astype(bool),
                candidate_group_column=group_column,
            )
            if threshold is not None
            else None
        )
        ranking[split] = (
            ranking_metrics_for_split(
                X=X.loc[valid],
                y=y.loc[valid],
                scores=s.loc[valid],
                candidate_group_column=group_column,
            )
            if valid.any()
            else None
        )
        ranking[f"{split}_failure_as_zero_group"] = (
            ranking_metrics_with_failed_groups_as_zero(
                X=X, y=y, scores=s, candidate_group_column=group_column
            )
        )
    pd.concat(frames, ignore_index=True).to_parquet(
        output_dir / "predictions.parquet", index=False
    )
    common = {
        "method": method_name,
        "task": task.name,
        "coverage": coverage,
        "classification_coverage": conversion_coverage,
        "candidate_groups": groups,
        "llm_errors": {s: sum(e.get("split") == s for e in errors) for s in pairs},
        "headline_population": "fixed_requested_groups_including_failures",
    }
    metrics = {
        **common,
        **pointwise,
        "protocol": "fused_scores_thresholded_on_validation",
        "main_metric": POINTWISE_MAIN_METRIC,
        "threshold": threshold,
        "threshold_selection": selection,
        "threshold_metric": selection["metric"],
    }
    result = {
        **common,
        **ranking,
        "protocol": "adaptive_rank_fusion",
        "main_metric": RANKING_MAIN_METRIC,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
            "scores": "original_unnormalized_fusion",
        },
        "pointwise_threshold": threshold,
    }
    root = repo_root()
    sources = [
        Path(__file__),
        root
        / "runners/in_distribution/interaction_prediction/methods/_llmsrec_syn_data.py",
        root / "runners/in_distribution/interaction_prediction/methods/common.py",
        root / "src/beyond_click_sim/evaluation/binary.py",
        root / "src/beyond_click_sim/evaluation/ranking.py",
    ]
    manifest = {
        **common,
        "evaluation_identity": identity,
        "status": "complete"
        if all(s.notna().all() for s in scores.values())
        else "completed_with_failures",
        "scorer": {
            "class": "DyTA4RecScorer",
            "client_name": client_name,
            **scorer.configuration,
        },
        "initialization_and_inference": scorer.initialization_metadata(),
        "training_provenance": prepared.provenance,
        "task": {"name": task.name, "manifest": task.manifest},
        "decision_rule": selection,
        "configuration_selection": "fixed_configuration; external_per_seed_validation_ndcg@5_selection",
        "inference_calls_for_classification": 0,
        "git_commit": current_git_commit(root),
        "runner_source_sha256": {
            str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "limits": {
            "max_groups_per_split": max_candidate_groups,
            "max_group_attempts": max_llm_attempts,
            "max_workers": max_workers,
        },
        "claim": "Temporal ranking/fusion adaptation with upstream-derived two-level memory reconstruction; not exact reproduction",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "metrics_ranking.json", result)
    write_json(output_dir / "manifest.json", manifest)
    return result


def run_qwen3_8b_smoke(task, output_dir):
    return run_method(
        task,
        output_dir,
        method_name="dyta4rec_offline_rrf_litellm_qwen3_8b_smoke",
        max_candidate_groups=5,
    )


def run_qwen3_8b_full(task, output_dir):
    return run_method(
        task, output_dir, method_name="dyta4rec_offline_rrf_litellm_qwen3_8b_full"
    )


def run_qwen36_27b_smoke(task, output_dir):
    return run_method(
        task,
        output_dir,
        method_name="dyta4rec_offline_rrf_litellm_qwen36_27b_smoke",
        model="Qwen/Qwen3.6-27B",
        max_candidate_groups=5,
    )


def run_qwen36_27b_full(task, output_dir):
    return run_method(
        task,
        output_dir,
        method_name="dyta4rec_offline_rrf_litellm_qwen36_27b_full",
        model="Qwen/Qwen3.6-27B",
    )


def main():
    from ..task_builders import TASK_BUILDERS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_BUILDERS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--client", default="litellm_local")
    parser.add_argument("--model-revision")
    parser.add_argument("--embedding-revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fusion", choices=("rrf", "bc"), default="rrf")
    parser.add_argument("--history-items", type=int, default=10)
    parser.add_argument("--icl-examples", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    args = parser.parse_args()
    config = DyTA4RecConfig(
        fusion=args.fusion,
        seed=args.seed,
        history_items=args.history_items,
        icl_examples=args.icl_examples,
        context_tokens=args.context_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result = run_method(
        TASK_BUILDERS[args.task](),
        args.output_dir,
        model=args.model,
        client_name=args.client,
        model_revision=args.model_revision,
        embedding_revision=args.embedding_revision,
        cache_dir=args.cache_dir,
        max_candidate_groups=args.max_groups,
        max_workers=args.max_workers,
        config=config,
        extra_body={} if args.model == "gpt-4o-mini" else None,
        preflight_only=args.preflight,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
