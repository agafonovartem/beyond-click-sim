"""Isolated LLMSRec-Syn interaction runner; tuning never evaluates test data."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers.history.llmsrec_syn import (
    LLMSRecSynScorer,
    positive_integer,
)
from beyond_click_sim.scorers.history.llmsrec_syn_prompts import AUTHOR_REVISION
from beyond_click_sim.scorers.history.llmsrec_syn_retrieval import (
    CachedEmbeddings,
    digest,
    identity,
)
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods._listwise import (
    evaluate_listwise_scorer,
)
from runners.in_distribution.interaction_prediction.methods._llmsrec_syn_data import (
    prepare_training,
    require_temporal_task,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    limit_candidate_groups,
    ranking_metrics_with_failed_groups_as_zero,
    task_xy,
    write_json,
)
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    _score_groups,
)
from runners.in_distribution.interaction_prediction.task_builders import (
    load_canonical_dataset,
    repo_root,
)


SELECTION_METRIC = "macro_by_user_group_mean_ndcg@5_failed_group_zero"


def select_k(
    scorer: LLMSRecSynScorer,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    k_values: Sequence[int],
    max_attempts: int,
    max_workers: int,
    after_scored: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Use fixed validation groups, including every failure, for all choices."""
    values = sorted({positive_integer(k, "k") for k in k_values})
    if not values or values[-1] > 20:
        raise ValueError(
            "k_values must be nonempty and contain integers from 1 through 20"
        )
    if X_val.empty or not X_val.index.is_unique or not X_val.index.equals(y_val.index):
        raise ValueError(
            "K selection requires nonempty validation rows aligned to labels"
        )
    group_column = scorer.candidate_group_column
    fingerprint = digest(
        [
            [
                identity(row.user_id),
                identity(getattr(row, group_column)),
                identity(row.item_id),
            ]
            for row in X_val.itertuples(index=False)
        ]
    )
    records = []
    for k in values:
        scorer.k = k
        scores, errors = _score_groups(
            scorer,
            X_val,
            candidate_group_column=group_column,
            max_attempts=max_attempts,
            max_workers=max_workers,
        )
        if after_scored is not None:
            after_scored(f"selection_k{k}")
        metrics = ranking_metrics_with_failed_groups_as_zero(
            X=X_val,
            y=y_val,
            scores=scores,
            candidate_group_column=group_column,
            ks=(5,),
        )["macro_by_user_group_mean"]
        records.append(
            {
                "k": k,
                "selection_score": metrics["ndcg@5"],
                "requested_groups": metrics["n_groups"],
                "successful_groups": metrics["scored_groups"],
                "failed_groups": metrics["failed_groups"],
                "errors": errors,
                "metrics": metrics,
            }
        )
    # Values are ascending: max retains the smaller K for equal scores.
    selected = max(records, key=lambda row: row["selection_score"])
    result = {
        "metric": SELECTION_METRIC,
        "tie_break": "smaller_k",
        "validation_group_fingerprint": fingerprint,
        "validation_rows": len(X_val),
        "validation_users": int(X_val.user_id.nunique()),
        "selected_k": selected["k"],
        "candidates": records,
    }
    scorer.k = selected["k"]
    return result


def _request_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [r for r in records if r["request_issued"]]
    usages = [r["usage"] for r in requests]
    return {
        "group_attempts": len(records),
        "completed_groups": sum(r["completed"] for r in records),
        "scorer_issued_requests": len(requests),
        "usage_complete": bool(usages)
        and all(u is not None and "total_tokens" in u for u in usages),
        "reported_tokens": {
            field: sum(u[field] for u in usages if u and field in u)
            if any(u and field in u for u in usages)
            else None
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "elapsed_group_seconds": sum(r["elapsed_seconds"] for r in records),
    }


def _provenance(root: Path) -> dict[str, Any]:
    paths = [
        "src/beyond_click_sim/scorers/history/llmsrec_syn.py",
        "src/beyond_click_sim/scorers/history/llmsrec_syn_retrieval.py",
        "src/beyond_click_sim/scorers/history/llmsrec_syn_prompts.py",
        "src/beyond_click_sim/scorers/history/listwise.py",
        "src/beyond_click_sim/tasks/splitters.py",
        "runners/in_distribution/interaction_prediction/methods/llmsrec_syn.py",
        "runners/in_distribution/interaction_prediction/methods/_llmsrec_syn_data.py",
        "runners/in_distribution/interaction_prediction/methods/_listwise.py",
        "runners/in_distribution/interaction_prediction/methods/common.py",
    ]
    return {
        "paper": {
            "arxiv": "2403.10135v1",
            "implementation_kind": "architecture_reimplementation_with_protocol_adaptations",
        },
        "author_code": {
            "repository": "https://github.com/demoleiwang/LLMSRec_Syn",
            "revision": AUTHOR_REVISION,
            "manuscript_precedence": True,
            "published_version_correspondence_verified": False,
        },
        "source_file_sha256": {
            path: sha256((root / path).read_bytes()).hexdigest() for path in paths
        },
    }


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str = "llmsrec_syn",
    client_name: str = "openai",
    model: str = "gpt-3.5-turbo",
    embedding_client_name: str = "openai",
    embedding_model: str = "text-embedding-ada-002",
    embedding_model_identity: str | None = None,
    embedding_cache_path: Path | None = None,
    embedding_batch_size: int = 64,
    canonical_dataset: CanonicalDataset | None = None,
    k_values: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    max_candidate_groups: int | None = None,
    max_history_items: int = 50,
    aggregate_history_items: int = 50,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_llm_attempts: int = 5,
    max_workers: int = 1,
    seed: int = 0,
    extra_body: dict | None = None,
) -> dict[str, Any]:
    require_temporal_task(task)
    max_workers = positive_integer(max_workers, "max_workers")
    max_llm_attempts = positive_integer(max_llm_attempts, "max_llm_attempts")
    values = sorted({positive_integer(k, "k") for k in k_values})
    if not values or values[-1] > 20:
        raise ValueError("k_values must contain integers from 1 through 20")
    prepared = prepare_training(
        task,
        canonical_dataset or load_canonical_dataset(str(task.manifest["dataset"])),
    )
    # Test rows/labels are not read by the configuration-selection path.
    target_column = task.schema.target_column
    group_column = task.schema.candidate_group_column
    X_val, y_val = limit_candidate_groups(
        task.val.drop(columns=target_column).copy(),
        task.val[target_column].copy(),
        candidate_group_column=group_column,
        max_candidate_groups=max_candidate_groups,
    )
    if X_val.empty:
        raise ValueError("LLMSRec-Syn requires nonempty validation groups")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        "manifest.json",
        "predictions.parquet",
        "llmsrec_syn_attempts.jsonl",
        "llmsrec_syn_selection.json",
    ]
    if any((output_dir / name).exists() for name in artifacts):
        raise FileExistsError("Use a fresh output directory for LLMSRec-Syn results")
    llm_client = make_llm_client(client_name)
    embedding_client = make_llm_client(embedding_client_name)
    # The caller can include a deployment revision when the endpoint uses aliases.
    model_identity = (
        embedding_model_identity
        or f"{embedding_client_name}:{getattr(embedding_client, 'base_url', '')}:{embedding_model}"
    )
    embeddings = CachedEmbeddings(
        embedding_client,
        embedding_model,
        model_identity=model_identity,
        training_fingerprint=prepared.provenance["training_fingerprint"],
        cache_path=embedding_cache_path
        or output_dir / "llmsrec_syn_embeddings.sqlite3",
        batch_size=embedding_batch_size,
    )
    statistics: dict[str, Any] = {}
    diagnostics_path = output_dir / "llmsrec_syn_attempts.jsonl"
    try:
        scorer = LLMSRecSynScorer(
            llm_client,
            model,
            embeddings=embeddings,
            dataset=str(task.manifest["dataset"]),
            k=values[0],
            max_history_items=max_history_items,
            aggregate_history_items=aggregate_history_items,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            extra_body=extra_body,
            candidate_group_column=group_column,
        ).fit(
            task.train.drop(columns=target_column),
            task.train[target_column],
            train_event_metadata=prepared.events,
            catalog=prepared.catalog,
        )

        def flush(phase: str) -> None:
            records = scorer.drain_diagnostics()
            with diagnostics_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            {"phase": phase, **record},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
            statistics[phase] = _request_summary(records)
            statistics["embeddings"] = embeddings.statistics()
            write_json(output_dir / "llmsrec_syn_request_summary.json", statistics)

        selection = select_k(
            scorer,
            X_val,
            y_val,
            k_values=values,
            max_attempts=max_llm_attempts,
            max_workers=max_workers,
            after_scored=flush,
        )
        write_json(output_dir / "llmsrec_syn_selection.json", selection)
        if not any(row["successful_groups"] for row in selection["candidates"]):
            raise RuntimeError(
                "LLMSRec-Syn produced no valid rankings for any validation configuration"
            )
        # Only after freezing K do we materialize test model inputs and labels.
        xy = task_xy(task)
        X_test, y_test = limit_candidate_groups(
            *xy["test"],
            candidate_group_column=group_column,
            max_candidate_groups=max_candidate_groups,
        )
        return evaluate_listwise_scorer(
            task=task,
            output_dir=output_dir,
            method_name=method_name,
            scorer=scorer,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            candidate_group_column=group_column,
            max_candidate_groups=max_candidate_groups,
            max_llm_attempts=max_llm_attempts,
            max_workers=max_workers,
            repo_root=repo_root(),
            source_metadata=_provenance(repo_root()),
            score_mapping=scorer.score_mapping,
            after_split_scored=flush,
            scorer_manifest={
                "class": "LLMSRecSynScorer",
                "client_name": client_name,
                "model": model,
                "embedding_client_name": embedding_client_name,
                "embedding_model": embedding_model,
                "embedding_model_identity": model_identity,
                "k": scorer.k,
                "selection": selection,
                "prompt": scorer.prompt_metadata,
                "max_history_items": max_history_items,
                "aggregate_history_items": aggregate_history_items,
                "synthetic_candidate_count": 20,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": extra_body,
                "seed": seed,
                "uses_item_stats": False,
                "metadata": "titles_only",
                "cutoff_rule": "other_user_train_timestamp <= max_target_train_timestamp",
                "label_rule": "last_event_after_cutoff_filtering",
                "training_provenance": prepared.provenance,
                "score_limitations": "ordinal; fixed positive count per group size at a fixed threshold; not absolute interaction willingness",
                "diagnostics": diagnostics_path.name,
                "request_statistics": statistics,
                "validation_rescored_after_k_selection": True,
            },
        )
    finally:
        embeddings.close()


def run_openai_smoke(task: Task, output_dir: Path) -> dict[str, Any]:
    return run_method(
        task,
        output_dir,
        method_name="llmsrec_syn_openai_smoke",
        max_candidate_groups=25,
    )


def run_openai_full(task: Task, output_dir: Path) -> dict[str, Any]:
    return run_method(task, output_dir, method_name="llmsrec_syn_openai_full")


def _run_qwen(
    task: Task,
    output_dir: Path,
    *,
    model: str,
    suffix: str,
    limit: int | None,
    workers: int,
) -> dict[str, Any]:
    return run_method(
        task,
        output_dir,
        method_name=f"llmsrec_syn_litellm_{suffix}",
        client_name="litellm_local",
        model=model,
        max_candidate_groups=limit,
        max_workers=workers,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def run_qwen3_8b_smoke(task: Task, output_dir: Path) -> dict[str, Any]:
    return _run_qwen(
        task,
        output_dir,
        model="Qwen/Qwen3-8B",
        suffix="qwen3_8b_smoke",
        limit=25,
        workers=64,
    )


def run_qwen3_8b_full(task: Task, output_dir: Path) -> dict[str, Any]:
    return _run_qwen(
        task,
        output_dir,
        model="Qwen/Qwen3-8B",
        suffix="qwen3_8b_full",
        limit=None,
        workers=64,
    )


def run_qwen36_27b_smoke(task: Task, output_dir: Path) -> dict[str, Any]:
    return _run_qwen(
        task,
        output_dir,
        model="Qwen/Qwen3.6-27B",
        suffix="qwen36_27b_smoke",
        limit=25,
        workers=32,
    )


def run_qwen36_27b_full(task: Task, output_dir: Path) -> dict[str, Any]:
    return _run_qwen(
        task,
        output_dir,
        model="Qwen/Qwen3.6-27B",
        suffix="qwen36_27b_full",
        limit=None,
        workers=32,
    )
