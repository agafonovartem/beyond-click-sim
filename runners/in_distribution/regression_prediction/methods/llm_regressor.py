from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMRegressor
from beyond_click_sim.tasks import Task
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
MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 64
MAX_LLM_ATTEMPTS = 5
SMOKE_ROWS = 25
OLLAMA_MAX_WORKERS = 1
VLLM_MAX_WORKERS = 32

DATASET_TARGET_PROMPT_CONFIG = {
    "ml-1m": {
        "target_rating": {
            "history_description_columns": ("item_title", "item_genres", "rating"),
            "candidate_description_columns": ("item_title", "item_genres"),
            "target_name": "rating",
            "target_description": (
                "Predict the integer MovieLens rating this user would give to "
                "the candidate movie on a 1 to 5 scale."
            ),
            "output_instructions": (
                "Return exactly one integer: 1, 2, 3, 4, or 5. "
                "Return no other text."
            ),
            "valid_values": (1, 2, 3, 4, 5),
            "output_format": "bare_integer",
        },
    },
}


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
) -> dict[str, object]:
    """Run an LLM discrete numeric scorer for regression prediction."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if task.schema.candidate_group_column is not None:
        raise ValueError("LLM regression method requires observed-only task rows")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")

    dataset_name = str(task.manifest["dataset"])
    target_source_column = str(task.manifest["target_source_column"])
    target_config = DATASET_TARGET_PROMPT_CONFIG[dataset_name][target_source_column]

    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = xy["test"]
    if max_rows is not None:
        X_test = X_test.head(max_rows).copy()
        y_test = y_test.loc[X_test.index].copy()

    scorer = LLMRegressor(
        client=make_llm_client(client_name),
        model=model,
        history_description_columns=target_config["history_description_columns"],
        candidate_description_columns=target_config["candidate_description_columns"],
        target_description=str(target_config["target_description"]),
        output_instructions=str(target_config["output_instructions"]),
        valid_values=target_config["valid_values"],
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
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
    prompt_columns = {
        "history_description_columns": target_config["history_description_columns"],
        "candidate_description_columns": target_config["candidate_description_columns"],
    }
    scorer_manifest = {
        "class": "LLMRegressor",
        "client_name": client_name,
        "model": model,
        "max_history_items": max_history_items,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_columns": prompt_columns,
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
