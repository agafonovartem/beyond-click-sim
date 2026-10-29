from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMRankScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods._listwise import (
    evaluate_listwise_scorer,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    limit_candidate_groups,
    task_xy,
    write_json,
)
from runners.in_distribution.interaction_prediction.methods.llm_listwise_ranking import (
    LITELLM_CLIENT_NAME,
    QWEN3_8B_MODEL,
    QWEN3_8B_MAX_WORKERS,
    QWEN36_27B_MODEL,
    QWEN36_27B_MAX_WORKERS,
    _serving_metadata,
    _source_metadata,
)
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    DATASET_JSON_LIST_COLUMNS,
    DATASET_PROMPT_COLUMNS,
    QWEN_EXTRA_BODY,
    prompt_column_labels,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root
from runners.in_distribution.llm_item_stats import maybe_add_item_rating_prompt_columns


QWEN3_8B_METHOD_NAME = "llmrank_litellm_qwen3_8b"
QWEN36_27B_METHOD_NAME = "llmrank_litellm_qwen36_27b"
MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 512
MAX_LLM_ATTEMPTS = 5


def run_qwen3_8b_with_item_stats_smoke(
    task: Task, output_dir: Path
) -> dict[str, object]:
    return _run_qwen(
        task,
        output_dir,
        model=QWEN3_8B_MODEL,
        method_name=f"{QWEN3_8B_METHOD_NAME}_with_item_stats_smoke",
        max_candidate_groups=25,
        max_workers=QWEN3_8B_MAX_WORKERS,
    )


def run_qwen3_8b_with_item_stats_full(
    task: Task, output_dir: Path
) -> dict[str, object]:
    return _run_qwen(
        task,
        output_dir,
        model=QWEN3_8B_MODEL,
        method_name=f"{QWEN3_8B_METHOD_NAME}_with_item_stats_full",
        max_candidate_groups=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
    )


def run_qwen36_27b_with_item_stats_smoke(
    task: Task, output_dir: Path
) -> dict[str, object]:
    return _run_qwen(
        task,
        output_dir,
        model=QWEN36_27B_MODEL,
        method_name=f"{QWEN36_27B_METHOD_NAME}_with_item_stats_smoke",
        max_candidate_groups=25,
        max_workers=QWEN36_27B_MAX_WORKERS,
    )


def run_qwen36_27b_with_item_stats_full(
    task: Task, output_dir: Path
) -> dict[str, object]:
    return _run_qwen(
        task,
        output_dir,
        model=QWEN36_27B_MODEL,
        method_name=f"{QWEN36_27B_METHOD_NAME}_with_item_stats_full",
        max_candidate_groups=None,
        max_workers=QWEN36_27B_MAX_WORKERS,
    )


def _run_qwen(
    task: Task,
    output_dir: Path,
    *,
    model: str,
    method_name: str,
    max_candidate_groups: int | None,
    max_workers: int,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=method_name,
        client_name=LITELLM_CLIENT_NAME,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=max_workers,
        use_item_stats=True,
        extra_body=QWEN_EXTRA_BODY,
        serving_metadata=_serving_metadata(),
        source_metadata=_source_metadata(),
    )


def require_temporal_task(task: Task) -> None:
    """Reject unsupported or legacy tasks before constructing an LLM client."""

    if task.manifest.get("dataset") != "ml-1m":
        raise ValueError(
            "LLMRank interaction runner currently supports only temporal ml-1m tasks"
        )
    splitter = task.manifest.get("splitter", {})
    if (
        splitter.get("class") != "RandomFractionSplitter"
        or splitter.get("shuffle") is not False
        or splitter.get("group_column") != "user_id"
        or not isinstance(splitter.get("timestamp_column"), str)
        or not splitter["timestamp_column"]
    ):
        raise ValueError(
            "LLMRank requires an explicit per-user temporal splitter manifest"
        )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
    max_history_items: int = MAX_HISTORY_ITEMS,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_llm_attempts: int = MAX_LLM_ATTEMPTS,
    max_workers: int = 1,
    use_item_stats: bool = False,
    bootstrap_seed: int = 0,
    extra_body: dict | None = None,
    serving_metadata: dict[str, object] | None = None,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate the final three-round LLMRank adaptation on MovieLens tasks."""

    require_temporal_task(task)
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("LLMRank requires candidate_group_column")
    if max_llm_attempts < 1 or max_workers < 1:
        raise ValueError("max_llm_attempts and max_workers must be positive")
    dataset_name = str(task.manifest["dataset"])
    prompt_columns = maybe_add_item_rating_prompt_columns(
        dataset_name,
        DATASET_PROMPT_COLUMNS[dataset_name],
        use_item_stats=use_item_stats,
    )
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_val, y_val = limit_candidate_groups(
        *xy["val"],
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    X_test, y_test = limit_candidate_groups(
        *xy["test"],
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    history_user_ids = pd.concat(
        [X_val["user_id"], X_test["user_id"]]
    ).drop_duplicates()
    column_labels = prompt_column_labels(dataset_name, use_item_stats=use_item_stats)
    scorer = LLMRankScorer(
        client=make_llm_client(client_name),
        model=model,
        user_column="user_id",
        candidate_group_column=candidate_group_column,
        history_description_columns=prompt_columns["history_description_columns"],
        candidate_description_columns=prompt_columns["candidate_description_columns"],
        column_labels=column_labels,
        json_list_columns=DATASET_JSON_LIST_COLUMNS[dataset_name],
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        bootstrap_seed=bootstrap_seed,
        extra_body=extra_body,
    ).fit(X_train, y_train, history_user_ids=history_user_ids.tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "llmrank_rounds.jsonl"
    diagnostics_path.write_text("", encoding="utf-8")
    statistics: dict[str, Any] = {}

    def flush_diagnostics(split: str) -> None:
        records = scorer.drain_diagnostics()
        with diagnostics_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps({"split": split, **record}, ensure_ascii=False) + "\n"
                )
        statistics[split] = _request_summary(records)
        write_json(output_dir / "llmrank_request_summary.json", statistics)

    root = repo_root()
    provenance = {**(source_metadata or {}), **_method_provenance(root)}
    return evaluate_listwise_scorer(
        task=task,
        output_dir=output_dir,
        method_name=method_name,
        scorer=scorer,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
        max_llm_attempts=max_llm_attempts,
        max_workers=max_workers,
        repo_root=root,
        source_metadata=provenance,
        score_mapping=scorer.score_mapping,
        after_split_scored=flush_diagnostics,
        scorer_manifest={
            "class": scorer.__class__.__name__,
            "client_name": client_name,
            "model": model,
            "max_history_items": max_history_items,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_columns": prompt_columns,
            "column_labels": column_labels,
            "json_list_columns": list(DATASET_JSON_LIST_COLUMNS[dataset_name]),
            "uses_item_stats": use_item_stats,
            "extra_body": extra_body,
            "prompt": scorer.prompt_metadata,
            "serving": serving_metadata,
            "history_order": "upstream_temporal_train_order",
            "bootstrap": {
                "rounds": scorer.bootstrap_rounds,
                "seed": scorer.bootstrap_seed,
                "seed_algorithm": scorer.seed_algorithm,
                "aggregation": scorer.score_mapping,
                "retry_policy": "whole_ensemble_same_permutations_no_partial_scores",
                "successful_requests_per_group": scorer.bootstrap_rounds,
                "max_scorer_requests_per_group": scorer.bootstrap_rounds
                * max_llm_attempts,
                "excludes_client_retries": True,
                "diagnostics": diagnostics_path.name,
                "request_statistics": statistics,
            },
        },
    )


def _request_summary(records: list[dict[str, Any]]) -> dict[str, object]:
    rounds = [round_record for record in records for round_record in record["rounds"]]
    usage = [record["usage"] for record in rounds if record["usage"] is not None]
    return {
        "ensemble_attempts": len(records),
        "completed_ensembles": sum(record["completed"] for record in records),
        "scorer_issued_requests": len(rounds),
        "valid_rankings": sum("ranked_labels" in record for record in rounds),
        "requests_with_usage": len(usage),
        "requests_without_usage": len(rounds) - len(usage),
        "reported_token_totals": {
            field: sum(values) if values else None
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            for values in [[entry[field] for entry in usage if field in entry]]
        },
        "usage_complete": bool(rounds)
        and all(
            record["usage"] is not None
            and all(
                field in record["usage"]
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            for record in rounds
        ),
        "elapsed_request_seconds": sum(record["elapsed_seconds"] for record in rounds),
    }


def _method_provenance(root: Path) -> dict[str, object]:
    source_files = [
        "src/beyond_click_sim/scorers/history/llmrank.py",
        "src/beyond_click_sim/scorers/history/listwise.py",
        "src/beyond_click_sim/scorers/history/prompts.py",
        "src/beyond_click_sim/scorers/history/selection.py",
        "src/beyond_click_sim/scorers/constant.py",
        "src/beyond_click_sim/tasks/splitters.py",
        "src/beyond_click_sim/tasks/alignment.py",
        "src/beyond_click_sim/tasks/base.py",
        "runners/in_distribution/interaction_prediction/methods/llmrank.py",
        "runners/in_distribution/interaction_prediction/methods/_listwise.py",
        "runners/in_distribution/interaction_prediction/methods/llm_yes_no.py",
        "runners/in_distribution/interaction_prediction/task_builders.py",
    ]
    return {
        "paper": {
            "url": "https://arxiv.org/abs/2305.08845v2",
            "sections": ["2.2", "3.2", "3.3"],
            "implementation_kind": "controlled_adaptation",
        },
        "author_code": {
            "url": "https://github.com/RUCAIBox/LLMRank",
            "retrieved": "2026-10-25",
            "commit": None,
            "published_version_correspondence_verified": False,
        },
        "source_file_sha256": {
            name: sha256((root / name).read_bytes()).hexdigest()
            for name in source_files
        },
    }
