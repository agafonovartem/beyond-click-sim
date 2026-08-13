from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMInteractionYesNoScorer
from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.tasks import split_xy
from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.llm_error_budget import (
    DEFAULT_MAX_ERROR_RATE,
    DEFAULT_MIN_GROUPS_BEFORE_CHECK,
    LLMErrorRateExceededError,
    check_error_budget,
)
from runners.in_distribution.item_summaries import (
    ITEM_SUMMARY_COLUMN,
    ITEM_SUMMARY_COLUMN_LABEL,
    SummaryVisibility,
    maybe_add_item_summary_prompt_columns,
    resolve_item_summary_visibility,
    task_item_summary_metadata,
)
from runners.in_distribution.llm_item_stats import (
    item_rating_column_labels,
    maybe_add_item_rating_prompt_columns,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    candidate_group_summary,
    cold_start_metadata,
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
from runners.in_distribution.cold_start.metrics import (
    POINTWISE_MAIN_METRIC,
    POINTWISE_METRICS_FILENAME,
    RANKING_KS,
    RANKING_MAIN_METRIC,
    RANKING_METRICS_FILENAME,
    RANKING_TIE_POLICY,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root

OLLAMA_LLAMA31_8B_METHOD_NAME = "llm_yes_no_ollama_llama31_8b"
OLLAMA_LLAMA31_8B_CLIENT = "ollama_local"
OLLAMA_LLAMA31_8B_MODEL = "llama3.1:8b"
VLLM_LLAMA33_70B_METHOD_NAME = "llm_yes_no_vllm_llama33_70b"
VLLM_LLAMA33_70B_CLIENT = "vllm_local"
VLLM_LLAMA33_70B_MODEL = "llama-3.3-70b-instruct"
VLLM_QWEN36_27B_METHOD_NAME = "llm_yes_no_vllm_qwen36_27b"
VLLM_QWEN36_27B_CLIENT = "vllm_local"
VLLM_QWEN36_27B_MODEL = "Qwen/Qwen3.6-27B"
VLLM_QWEN36_35B_A3B_METHOD_NAME = "llm_yes_no_vllm_qwen36_35b_a3b"
VLLM_QWEN36_35B_A3B_CLIENT = "vllm_local"
VLLM_QWEN36_35B_A3B_MODEL = "Qwen/Qwen3.6-35B-A3B"
QWEN_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}
MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_LLM_ATTEMPTS = 5
OLLAMA_MAX_WORKERS = 1
# VLLM_MAX_WORKERS = 128
VLLM_MAX_WORKERS = 240

DATASET_PROMPT_COLUMNS = {
    "ml-1m": {
        "history_description_columns": ("item_title", "item_genres", "rating"),
        "candidate_description_columns": ("item_title", "item_genres"),
    },
}


def run_llama31_8b_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_smoke",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=25,
        max_workers=OLLAMA_MAX_WORKERS,
        scoring="batch",
    )


def run_llama31_8b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_full",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=None,
        max_workers=OLLAMA_MAX_WORKERS,
        scoring="batch",
    )


def run_llama31_8b_itemwise_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_itemwise_smoke",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=25,
        max_workers=OLLAMA_MAX_WORKERS,
        scoring="itemwise",
    )


def run_llama31_8b_itemwise_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_itemwise_full",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=None,
        max_workers=OLLAMA_MAX_WORKERS,
        scoring="itemwise",
    )


def run_llama33_70b_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_smoke",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        scoring="batch",
    )


def run_llama33_70b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_full",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        scoring="batch",
    )


def run_llama33_70b_itemwise_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_itemwise_smoke",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        scoring="itemwise",
    )


def run_llama33_70b_itemwise_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_itemwise_full",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        scoring="itemwise",
    )


def run_qwen36_27b_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        scoring="batch",
    )


def run_qwen36_27b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        scoring="batch",
    )


def run_qwen36_27b_itemwise_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_itemwise_smoke",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        scoring="itemwise",
    )


def run_qwen36_27b_itemwise_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_itemwise_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        scoring="itemwise",
    )


def run_qwen36_35b_a3b_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        scoring="batch",
    )


def run_qwen36_35b_a3b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        scoring="batch",
    )


def run_qwen36_35b_a3b_itemwise_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_itemwise_smoke",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        scoring="itemwise",
    )


def run_qwen36_35b_a3b_itemwise_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_itemwise_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        scoring="itemwise",
    )


def _sumboth_kwargs(scoring: str, *, qwen: bool) -> dict[str, object]:
    return {
        "summary_visibility": "both",
        "scoring": scoring,
        **({"extra_body": QWEN_EXTRA_BODY} if qwen else {}),
    }


def run_llama31_8b_sumboth_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_sumboth_smoke",
        client_name=OLLAMA_LLAMA31_8B_CLIENT, model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=25, max_workers=OLLAMA_MAX_WORKERS,
        **_sumboth_kwargs("batch", qwen=False),
    )


def run_llama31_8b_sumboth_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_sumboth_full",
        client_name=OLLAMA_LLAMA31_8B_CLIENT, model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=None, max_workers=OLLAMA_MAX_WORKERS,
        **_sumboth_kwargs("batch", qwen=False),
    )


def run_llama31_8b_itemwise_sumboth_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_itemwise_sumboth_smoke",
        client_name=OLLAMA_LLAMA31_8B_CLIENT, model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=25, max_workers=OLLAMA_MAX_WORKERS,
        **_sumboth_kwargs("itemwise", qwen=False),
    )


def run_llama31_8b_itemwise_sumboth_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_itemwise_sumboth_full",
        client_name=OLLAMA_LLAMA31_8B_CLIENT, model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=None, max_workers=OLLAMA_MAX_WORKERS,
        **_sumboth_kwargs("itemwise", qwen=False),
    )


def run_llama33_70b_sumboth_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_sumboth_smoke",
        client_name=VLLM_LLAMA33_70B_CLIENT, model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=25, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("batch", qwen=False),
    )


def run_llama33_70b_sumboth_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_sumboth_full",
        client_name=VLLM_LLAMA33_70B_CLIENT, model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("batch", qwen=False),
    )


def run_llama33_70b_itemwise_sumboth_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_itemwise_sumboth_smoke",
        client_name=VLLM_LLAMA33_70B_CLIENT, model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=25, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("itemwise", qwen=False),
    )


def run_llama33_70b_itemwise_sumboth_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_itemwise_sumboth_full",
        client_name=VLLM_LLAMA33_70B_CLIENT, model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("itemwise", qwen=False),
    )


def run_qwen36_27b_sumboth_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumboth_smoke",
        client_name=VLLM_QWEN36_27B_CLIENT, model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("batch", qwen=True),
    )


def run_qwen36_27b_sumboth_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumboth_full",
        client_name=VLLM_QWEN36_27B_CLIENT, model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("batch", qwen=True),
    )


def run_qwen36_27b_itemwise_sumboth_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_itemwise_sumboth_smoke",
        client_name=VLLM_QWEN36_27B_CLIENT, model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("itemwise", qwen=True),
    )


def run_qwen36_27b_itemwise_sumboth_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_itemwise_sumboth_full",
        client_name=VLLM_QWEN36_27B_CLIENT, model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("itemwise", qwen=True),
    )


def run_qwen36_35b_a3b_sumboth_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumboth_smoke",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT, model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("batch", qwen=True),
    )


def run_qwen36_35b_a3b_sumboth_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumboth_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT, model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("batch", qwen=True),
    )


def run_qwen36_35b_a3b_itemwise_sumboth_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_itemwise_sumboth_smoke",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT, model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("itemwise", qwen=True),
    )


def run_qwen36_35b_a3b_itemwise_sumboth_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task, output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_itemwise_sumboth_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT, model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None, max_workers=VLLM_MAX_WORKERS,
        **_sumboth_kwargs("itemwise", qwen=True),
    )


def run_method(
    task: ColdStartTask,
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
    extra_body: dict | None = None,
    summary_visibility: SummaryVisibility = "none",
    scoring: str = "itemwise",
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
    min_groups_before_check: int = DEFAULT_MIN_GROUPS_BEFORE_CHECK,
) -> dict[str, object]:
    """Run the yes/no LLM scorer for cold-start interaction alignment.

    The scorer is fitted on task.online_session_history — the cold user's k earliest
    temporal interactions — not on task.train (which contains only warm-user rows and
    would give the LLM zero per-cold-user context).

    `scoring="itemwise"` (default) issues one LLM call per candidate. `scoring="batch"`
    issues one LLM call per full candidate group (all positives + negatives together).

    `summary_visibility` controls movie summaries (ml-1m only): "history" appends
    summaries to profile items (online_session_history only — never post-k items),
    "candidate" appends them to candidate items, "both" does both.
    """

    if scoring not in ("batch", "itemwise"):
        raise ValueError(f"scoring must be 'batch' or 'itemwise', got {scoring!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    resolved_summary_visibility = resolve_item_summary_visibility(summary_visibility)
    prompt_columns = maybe_add_item_rating_prompt_columns(
        dataset_name,
        DATASET_PROMPT_COLUMNS[dataset_name],
        use_item_stats=use_item_stats,
    )
    prompt_columns = maybe_add_item_summary_prompt_columns(
        dataset_name,
        prompt_columns,
        history_item_summaries=resolved_summary_visibility["history"],
        candidate_item_summaries=resolved_summary_visibility["candidate"],
    )
    column_labels = item_rating_column_labels(
        dataset_name,
        use_item_stats=use_item_stats,
    )
    if resolved_summary_visibility["any"]:
        column_labels = {
            **column_labels,
            ITEM_SUMMARY_COLUMN: ITEM_SUMMARY_COLUMN_LABEL,
        }
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("LLM yes/no method requires candidate_group_column")

    # The cold user's k-item profile is the LLM's per-user context window.
    # task.train contains only warm-user rows — using it here would leave the LLM
    # with no history for any cold user.
    X_history, y_history = split_xy(
        task.online_session_history,
        target_column=task.schema.target_column,
    )

    xy = task_xy(task)
    X_test, y_test = limit_candidate_groups(
        *xy["test"],
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    item_summary_metadata = task_item_summary_metadata(
        task,
        history=resolved_summary_visibility["history"],
        candidate=resolved_summary_visibility["candidate"],
        extra_frames={"online_session_history": task.online_session_history},
    )

    scorer = LLMInteractionYesNoScorer(
        client=make_llm_client(client_name),
        model=model,
        history_description_columns=prompt_columns["history_description_columns"],
        candidate_description_columns=prompt_columns["candidate_description_columns"],
        column_labels=column_labels,
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        prompt_style=scoring,
    ).fit(X_history, y_history)

    X_scoring, score_group_column = _assign_itemwise_groups(
        X_test, candidate_group_column=candidate_group_column
    ) if scoring == "itemwise" else (X_test, candidate_group_column)

    try:
        scores, errors = _score_groups(
            scorer,
            X_scoring,
            candidate_group_column=score_group_column,
            max_attempts=max_llm_attempts,
            max_workers=max_workers,
            method_name=method_name,
            task_name=task.name,
            max_error_rate=max_error_rate,
            min_groups_before_check=min_groups_before_check,
        )
    except LLMErrorRateExceededError as error:
        _write_errors(output_dir / "llm_errors.jsonl", error.errors)
        raise
    valid = scores.notna()
    predictions = scores.astype("boolean").rename("prediction")
    prediction_frame(
        split="test",
        X=X_test,
        y=y_test,
        scores=scores,
        predictions=predictions,
        metadata=cold_start_metadata(task),
    ).to_parquet(output_dir / "predictions.parquet", index=False)
    _write_errors(output_dir / "llm_errors.jsonl", errors)

    if not valid.any():
        raise RuntimeError("LLM scorer did not produce any valid scores")

    valid_scores = scores.loc[valid]
    valid_X = X_test.loc[valid].copy()
    valid_y = y_test.loc[valid].copy()
    valid_predictions = valid_scores.astype(bool).rename("prediction")
    parsed_only_metrics = pointwise_metrics_for_split(
        X=valid_X,
        y=valid_y,
        predictions=valid_predictions,
        candidate_group_column=candidate_group_column,
    )
    failure_as_negative_metrics = failure_as_negative_pointwise_metrics(
        X=X_test,
        y=y_test,
        scores=scores,
        candidate_group_column=candidate_group_column,
    )
    ranking_metrics = ranking_metrics_for_split(
        X=valid_X,
        y=valid_y,
        scores=valid_scores,
        candidate_group_column=candidate_group_column,
    )
    failure_as_zero_ranking_metrics = ranking_metrics_with_failed_groups_as_zero(
        X=X_test,
        y=y_test,
        scores=scores,
        candidate_group_column=candidate_group_column,
    )
    coverage = score_coverage_summary(scores)
    requested_candidate_groups = candidate_group_summary(
        X_test,
        y_test,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )
    scored_candidate_groups = candidate_group_summary(
        valid_X,
        valid_y,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )

    root = repo_root()
    manifest = {
        "method": method_name,
        "scorer": {
            "class": "LLMInteractionYesNoScorer",
            "fit_on": "online_session_history",
            "k": task.k,
            "client_name": client_name,
            "model": model,
            "max_history_items": max_history_items,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_columns": prompt_columns,
            "column_labels": column_labels,
            "uses_item_stats": use_item_stats,
            "summary_visibility": summary_visibility,
            "item_summaries": item_summary_metadata,
            "extra_body": extra_body,
            "prompt_style": scoring,
        },
        "decision_rule": {
            "kind": "hard_binary_yes_no_parser",
            "threshold": None,
        },
        "limits": {
            "max_candidate_groups": max_candidate_groups,
            "max_llm_attempts": max_llm_attempts,
            "max_workers": max_workers,
            "max_error_rate": max_error_rate,
            "min_groups_before_check": min_groups_before_check,
        },
        "llm_errors": len(errors),
        "candidate_groups": {
            "requested": requested_candidate_groups,
            "scored": scored_candidate_groups,
        },
        "task": {
            "name": task.name,
            "manifest": task.manifest,
        },
        "git_commit": current_git_commit(root),
    }
    result = {
        "method": method_name,
        "task": task.name,
        "main_metric": POINTWISE_MAIN_METRIC,
        "test": parsed_only_metrics,
        "test_failure_as_negative": failure_as_negative_metrics,
        "coverage": coverage,
        "llm_errors": len(errors),
        "scored_rows": int(valid.sum()),
        "requested_rows": int(len(X_test)),
        "max_candidate_groups": max_candidate_groups,
        "max_workers": max_workers,
        "scoring": scoring,
        "candidate_groups": {
            "requested": requested_candidate_groups,
            "scored": scored_candidate_groups,
        },
    }
    ranking_result = {
        "method": method_name,
        "task": task.name,
        "protocol": "ranking",
        "main_metric": RANKING_MAIN_METRIC,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
        },
        "test": ranking_metrics,
        "test_failure_as_zero_group": failure_as_zero_ranking_metrics,
        "coverage": coverage,
        "llm_errors": len(errors),
        "scored_rows": int(valid.sum()),
        "requested_rows": int(len(X_test)),
        "max_candidate_groups": max_candidate_groups,
        "max_workers": max_workers,
        "scoring": scoring,
        "candidate_groups": {
            "requested": requested_candidate_groups,
            "scored": scored_candidate_groups,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / POINTWISE_METRICS_FILENAME, result)
    write_json(output_dir / RANKING_METRICS_FILENAME, ranking_result)
    return result


_ITEMWISE_GROUP_COLUMN = "_itemwise_group_"


def _assign_itemwise_groups(
    X: pd.DataFrame, *, candidate_group_column: str
) -> tuple[pd.DataFrame, str]:
    """Explode a candidate-group column into one row per group for itemwise scoring.

    Cold-start candidate rows are already unique per (candidate_group, item), so a
    per-row group id is sufficient — unlike policy_ranking_agreement, there is no
    cross-group duplication of the same (user, item) pair to dedupe here.
    """

    del candidate_group_column
    X_itemwise = X.copy()
    X_itemwise[_ITEMWISE_GROUP_COLUMN] = X_itemwise.index.astype(str)
    return X_itemwise, _ITEMWISE_GROUP_COLUMN


def _score_groups(
    scorer: Scorer,
    X: pd.DataFrame,
    *,
    candidate_group_column: str,
    max_attempts: int,
    max_workers: int = 1,
    method_name: str,
    task_name: str,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
    min_groups_before_check: int = DEFAULT_MIN_GROUPS_BEFORE_CHECK,
) -> tuple[pd.Series, list[dict[str, object]]]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    scores = pd.Series(index=X.index, dtype=float, name="score")
    errors: list[dict[str, object]] = []
    groups = list(X.groupby(candidate_group_column, sort=False))
    total = len(groups)

    def _check(attempted: int, *, force: bool = False) -> None:
        check_error_budget(
            errors=errors,
            attempted=attempted,
            total=total,
            method_name=method_name,
            task_name=task_name,
            max_error_rate=max_error_rate,
            min_groups_before_check=min_groups_before_check,
            force=force,
        )

    if max_workers == 1:
        progress = tqdm(groups, desc="llm groups", unit="group")
        for attempted, (group_id, group) in enumerate(progress, start=1):
            group_scores, error = _score_one_group(
                scorer,
                group_id,
                group,
                max_attempts=max_attempts,
            )
            if error is not None:
                errors.append(error)
                progress.set_postfix(errors=len(errors))
            else:
                scores.loc[group_scores.index] = group_scores
            _check(attempted)
        _check(total, force=True)
        return scores, errors

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _score_one_group,
                scorer,
                group_id,
                group,
                max_attempts=max_attempts,
            )
            for group_id, group in groups
        ]
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="llm groups",
            unit="group",
        )
        attempted = 0
        for future in progress:
            attempted += 1
            group_scores, error = future.result()
            if error is not None:
                errors.append(error)
                progress.set_postfix(errors=len(errors))
            else:
                scores.loc[group_scores.index] = group_scores
            try:
                _check(attempted)
            except LLMErrorRateExceededError:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        _check(total, force=True)
    return scores, errors


def _score_one_group(
    scorer: Scorer,
    group_id: object,
    group: pd.DataFrame,
    *,
    max_attempts: int,
) -> tuple[pd.Series, dict[str, object] | None]:
    attempt_errors: list[str] = []
    for _ in range(max_attempts):
        try:
            return scorer.score(group), None
        except Exception as error:  # noqa: BLE001 - keep long LLM runs alive.
            attempt_errors.append(repr(error))

    empty_scores = pd.Series(index=group.index, dtype=float, name="score")
    return empty_scores, {
        "candidate_group": str(group_id),
        "attempts": max_attempts,
        "errors": attempt_errors,
    }


def _write_errors(path: Path, errors: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, sort_keys=True) + "\n")
