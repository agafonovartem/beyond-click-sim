from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMRegressor
from beyond_click_sim.tasks import Task
from runners.in_distribution.llm_item_stats import (
    item_rating_column_labels,
    maybe_add_item_rating_prompt_columns,
)
from runners.in_distribution.regression_prediction.config import (
    DATASET_TARGET_REGRESSION_CONFIG,
    MAX_HISTORY_ITEMS,
)
from runners.in_distribution.regression_prediction.item_summaries import (
    ITEM_SUMMARY_COLUMN,
    ITEM_SUMMARY_COLUMN_LABEL,
    SummaryVisibility,
    maybe_add_item_summary_prompt_columns,
    resolve_item_summary_visibility,
    task_item_summary_metadata,
)
from runners.in_distribution.regression_prediction.methods.common import (
    current_git_commit,
    regression_metrics_for_split,
    score_frame,
    task_xy,
    write_json,
)
from runners.in_distribution.regression_prediction.metrics import (
    REGRESSION_MAIN_METRIC,
    REGRESSION_METRICS_FILENAME,
)
from runners.in_distribution.regression_prediction.task_builders import repo_root

OLLAMA_LLAMA31_8B_METHOD_NAME = "llm_regressor_ollama_llama31_8b"
OLLAMA_LLAMA31_8B_CLIENT = "ollama_local"
OLLAMA_LLAMA31_8B_MODEL = "llama3.1:8b"
VLLM_LLAMA33_70B_METHOD_NAME = "llm_regressor_vllm_llama33_70b"
VLLM_LLAMA33_70B_CLIENT = "vllm_local"
VLLM_LLAMA33_70B_MODEL = "llama-3.3-70b-instruct"
VLLM_QWEN3_8B_METHOD_NAME = "llm_regressor_vllm_qwen3_8b"
VLLM_QWEN3_8B_CLIENT = "vllm_local"
VLLM_QWEN3_8B_MODEL = "Qwen/Qwen3-8B"
VLLM_QWEN36_27B_METHOD_NAME = "llm_regressor_vllm_qwen36_27b"
VLLM_QWEN36_27B_CLIENT = "vllm_local"
VLLM_QWEN36_27B_MODEL = "Qwen/Qwen3.6-27B"
QWEN_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}
OPENAI_VK_GPT54_MINI_METHOD_NAME = "llm_regressor_openai_vk_gpt54_mini"
OPENAI_VK_GPT54_MINI_CLIENT = "openai_vk_proxy"
OPENAI_VK_GPT54_MINI_MODEL = "gpt-5.4-mini"
OPENAI_VK_GPT55_METHOD_NAME = "llm_regressor_openai_vk_gpt55"
OPENAI_VK_GPT55_CLIENT = "openai_vk_proxy"
OPENAI_VK_GPT55_MODEL = "gpt-5.5"
TEMPERATURE = 0.0
MAX_TOKENS = 64
MAX_LLM_ATTEMPTS = 5
SMOKE_ROWS = 25
SMOKE10_ROWS = 10
OLLAMA_MAX_WORKERS = 1
VLLM_MAX_WORKERS = 32
QWEN3_8B_MAX_WORKERS = 128
QWEN36_27B_MAX_WORKERS = 128
OPENAI_VK_MAX_WORKERS = 4


def run_llama31_8b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_smoke",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=OLLAMA_MAX_WORKERS,
    )


def run_llama31_8b_smoke10(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_smoke10",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_rows=SMOKE10_ROWS,
        max_workers=OLLAMA_MAX_WORKERS,
    )


def run_llama31_8b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_with_item_stats_smoke",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=OLLAMA_MAX_WORKERS,
        use_item_stats=True,
    )


def run_llama31_8b_with_item_stats_smoke10(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_with_item_stats_smoke10",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_rows=SMOKE10_ROWS,
        max_workers=OLLAMA_MAX_WORKERS,
        use_item_stats=True,
    )


def run_llama31_8b_with_item_stats_summary_smoke10(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_with_item_stats_summary_smoke10",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_rows=SMOKE10_ROWS,
        max_workers=OLLAMA_MAX_WORKERS,
        use_item_stats=True,
        summary_visibility="both",
    )


def run_llama31_8b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_full",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_rows=None,
        max_workers=OLLAMA_MAX_WORKERS,
    )


def run_llama31_8b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_with_item_stats_full",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_rows=None,
        max_workers=OLLAMA_MAX_WORKERS,
        use_item_stats=True,
    )


def run_llama33_70b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_smoke",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=VLLM_MAX_WORKERS,
    )


def run_llama33_70b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_with_item_stats_smoke",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=VLLM_MAX_WORKERS,
        use_item_stats=True,
    )


def run_llama33_70b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_full",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_rows=None,
        max_workers=VLLM_MAX_WORKERS,
    )


def run_llama33_70b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_with_item_stats_full",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_rows=None,
        max_workers=VLLM_MAX_WORKERS,
        use_item_stats=True,
    )


def run_llama33_70b_with_item_stats_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_with_item_stats_summary_full",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_rows=None,
        max_workers=VLLM_MAX_WORKERS,
        use_item_stats=True,
        summary_visibility="both",
    )


def run_qwen3_8b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_with_item_stats_smoke",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=QWEN3_8B_MAX_WORKERS,
        use_item_stats=True,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen3_8b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_with_item_stats_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        use_item_stats=True,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen3_8b_with_item_stats_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_with_item_stats_summary_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        use_item_stats=True,
        summary_visibility="both",
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen3_8b_with_item_stats_history_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_with_item_stats_history_summary_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        use_item_stats=True,
        summary_visibility="history",
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen3_8b_with_item_stats_candidate_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_with_item_stats_candidate_summary_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        use_item_stats=True,
        summary_visibility="candidate",
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen36_27b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_with_item_stats_smoke",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=QWEN36_27B_MAX_WORKERS,
        use_item_stats=True,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen36_27b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_with_item_stats_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_rows=None,
        max_workers=QWEN36_27B_MAX_WORKERS,
        use_item_stats=True,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen36_27b_with_item_stats_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_with_item_stats_summary_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_rows=None,
        max_workers=QWEN36_27B_MAX_WORKERS,
        use_item_stats=True,
        summary_visibility="both",
        extra_body=QWEN_EXTRA_BODY,
    )


def run_gpt54_mini_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OPENAI_VK_GPT54_MINI_METHOD_NAME}_smoke",
        client_name=OPENAI_VK_GPT54_MINI_CLIENT,
        model=OPENAI_VK_GPT54_MINI_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=OPENAI_VK_MAX_WORKERS,
    )


def run_gpt54_mini_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OPENAI_VK_GPT54_MINI_METHOD_NAME}_with_item_stats_smoke",
        client_name=OPENAI_VK_GPT54_MINI_CLIENT,
        model=OPENAI_VK_GPT54_MINI_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=OPENAI_VK_MAX_WORKERS,
        use_item_stats=True,
    )


def run_gpt54_mini_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OPENAI_VK_GPT54_MINI_METHOD_NAME}_full",
        client_name=OPENAI_VK_GPT54_MINI_CLIENT,
        model=OPENAI_VK_GPT54_MINI_MODEL,
        max_rows=None,
        max_workers=OPENAI_VK_MAX_WORKERS,
    )


def run_gpt54_mini_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OPENAI_VK_GPT54_MINI_METHOD_NAME}_with_item_stats_full",
        client_name=OPENAI_VK_GPT54_MINI_CLIENT,
        model=OPENAI_VK_GPT54_MINI_MODEL,
        max_rows=None,
        max_workers=OPENAI_VK_MAX_WORKERS,
        use_item_stats=True,
    )


def run_gpt55_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OPENAI_VK_GPT55_METHOD_NAME}_smoke",
        client_name=OPENAI_VK_GPT55_CLIENT,
        model=OPENAI_VK_GPT55_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=OPENAI_VK_MAX_WORKERS,
    )


def run_gpt55_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OPENAI_VK_GPT55_METHOD_NAME}_with_item_stats_smoke",
        client_name=OPENAI_VK_GPT55_CLIENT,
        model=OPENAI_VK_GPT55_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=OPENAI_VK_MAX_WORKERS,
        use_item_stats=True,
    )


def run_gpt55_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OPENAI_VK_GPT55_METHOD_NAME}_full",
        client_name=OPENAI_VK_GPT55_CLIENT,
        model=OPENAI_VK_GPT55_MODEL,
        max_rows=None,
        max_workers=OPENAI_VK_MAX_WORKERS,
    )


def run_gpt55_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OPENAI_VK_GPT55_METHOD_NAME}_with_item_stats_full",
        client_name=OPENAI_VK_GPT55_CLIENT,
        model=OPENAI_VK_GPT55_MODEL,
        max_rows=None,
        max_workers=OPENAI_VK_MAX_WORKERS,
        use_item_stats=True,
    )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_rows: int | None,
    max_history_items: int = MAX_HISTORY_ITEMS,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_llm_attempts: int = MAX_LLM_ATTEMPTS,
    max_workers: int = 1,
    use_item_stats: bool = False,
    summary_visibility: SummaryVisibility = "none",
    extra_body: dict | None = None,
) -> dict[str, object]:
    """Run an LLM discrete numeric scorer for regression prediction."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if task.schema.candidate_group_column is not None:
        raise ValueError("LLM regression method requires observed-only task rows")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")

    dataset_name = str(task.manifest["dataset"])
    target_source_column = str(task.manifest["target_source_column"])
    target_config = DATASET_TARGET_REGRESSION_CONFIG[dataset_name][target_source_column]
    resolved_summary_visibility = resolve_item_summary_visibility(summary_visibility)
    base_prompt_columns = {
        "history_description_columns": target_config["history_description_columns"],
        "candidate_description_columns": target_config["candidate_description_columns"],
    }
    prompt_columns = maybe_add_item_rating_prompt_columns(
        dataset_name,
        base_prompt_columns,
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

    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = xy["test"]
    if max_rows is not None:
        X_test = X_test.head(max_rows).copy()
        y_test = y_test.loc[X_test.index].copy()
    item_summary_metadata = task_item_summary_metadata(
        task,
        history=resolved_summary_visibility["history"],
        candidate=resolved_summary_visibility["candidate"],
    )

    scorer = LLMRegressor(
        client=make_llm_client(client_name),
        model=model,
        history_description_columns=prompt_columns["history_description_columns"],
        candidate_description_columns=prompt_columns["candidate_description_columns"],
        target_description=str(target_config["target_description"]),
        output_instructions=str(target_config["output_instructions"]),
        valid_values=target_config["valid_values"],
        column_labels=column_labels,
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    ).fit(X_train, y_train)

    scores, errors = _score_rows(
        scorer,
        X_test,
        max_attempts=max_llm_attempts,
        max_workers=max_workers,
    )
    valid = scores.notna()
    score_frame(
        split="test",
        X=X_test,
        y=y_test,
        scores=scores,
    ).to_parquet(output_dir / "predictions.parquet", index=False)
    _write_errors(output_dir / "llm_errors.jsonl", errors)

    if not valid.any():
        raise RuntimeError("LLM scorer did not produce any valid scores")

    valid_scores = scores.loc[valid]
    valid_X = X_test.loc[valid].copy()
    valid_y = y_test.loc[valid].copy()
    test_metrics = regression_metrics_for_split(
        X=valid_X,
        y=valid_y,
        scores=valid_scores,
    )
    requested_rows = int(len(X_test))
    scored_rows = int(valid.sum())
    coverage = scored_rows / requested_rows if requested_rows else 0.0

    root = repo_root()
    scorer_manifest = {
        "class": "LLMRegressor",
        "client_name": client_name,
        "model": model,
        "max_history_items": max_history_items,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": extra_body,
        "prompt_columns": prompt_columns,
        "column_labels": column_labels,
        "uses_item_stats": use_item_stats,
        "summary_visibility": summary_visibility,
        "item_summaries": item_summary_metadata,
        "target": {
            "name": target_config["target_name"],
            "description": target_config["target_description"],
            "valid_values": target_config["valid_values"],
            "output_format": target_config["output_format"],
        },
    }
    limits = {
        "max_rows": max_rows,
        "max_llm_attempts": max_llm_attempts,
        "max_workers": max_workers,
    }
    manifest = {
        "method": method_name,
        "protocol": "regression",
        "scorer": scorer_manifest,
        "regression_evaluation": {
            "main_metric": REGRESSION_MAIN_METRIC,
            "aggregations": ["micro", "macro_by_user_mean"],
            "metrics": ["mae", "rmse"],
        },
        "limits": limits,
        "evaluated_splits": ["test"],
        "llm_errors": len(errors),
        "requested_rows": requested_rows,
        "scored_rows": scored_rows,
        "coverage": coverage,
        "task": {
            "name": task.name,
            "manifest": task.manifest,
        },
        "git_commit": current_git_commit(root),
    }
    metrics = {
        "method": method_name,
        "task": task.name,
        "protocol": "regression",
        "main_metric": REGRESSION_MAIN_METRIC,
        "regression_evaluation": {
            "aggregations": ["micro", "macro_by_user_mean"],
            "metrics": ["mae", "rmse"],
        },
        "evaluated_splits": ["test"],
        "test": test_metrics,
        "llm_errors": len(errors),
        "requested_rows": requested_rows,
        "scored_rows": scored_rows,
        "coverage": coverage,
        "max_rows": max_rows,
        "max_workers": max_workers,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / REGRESSION_METRICS_FILENAME, metrics)
    return metrics


def _score_rows(
    scorer: LLMRegressor,
    X: pd.DataFrame,
    *,
    max_attempts: int,
    max_workers: int = 1,
) -> tuple[pd.Series, list[dict[str, object]]]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    scores = pd.Series(index=X.index, dtype=float, name="score")
    errors: list[dict[str, object]] = []
    rows = list(X.iterrows())

    if max_workers == 1:
        progress = tqdm(rows, desc="llm rows", unit="row")
        for row_index, row in progress:
            row_scores, error = _score_one_row(
                scorer,
                row_index,
                row,
                max_attempts=max_attempts,
            )
            if error is not None:
                errors.append(error)
                progress.set_postfix(errors=len(errors))
            else:
                scores.loc[row_scores.index] = row_scores
        return scores, errors

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _score_one_row,
                scorer,
                row_index,
                row,
                max_attempts=max_attempts,
            )
            for row_index, row in rows
        ]
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="llm rows",
            unit="row",
        )
        for future in progress:
            row_scores, error = future.result()
            if error is not None:
                errors.append(error)
                progress.set_postfix(errors=len(errors))
            else:
                scores.loc[row_scores.index] = row_scores
    return scores, errors


def _score_one_row(
    scorer: LLMRegressor,
    row_index: object,
    row: pd.Series,
    *,
    max_attempts: int,
) -> tuple[pd.Series, dict[str, object] | None]:
    row_frame = pd.DataFrame([row], index=[row_index])
    attempt_errors: list[str] = []
    for _ in range(max_attempts):
        try:
            return scorer.score(row_frame), None
        except Exception as error:  # noqa: BLE001 - keep long LLM runs alive.
            attempt_errors.append(repr(error))

    empty_scores = pd.Series(index=row_frame.index, dtype=float, name="score")
    return empty_scores, {
        "row_index": str(row_index),
        "user_id": str(row.get("user_id", "")),
        "item_id": str(row.get("item_id", "")),
        "attempts": max_attempts,
        "errors": attempt_errors,
    }


def _write_errors(path: Path, errors: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, sort_keys=True) + "\n")
