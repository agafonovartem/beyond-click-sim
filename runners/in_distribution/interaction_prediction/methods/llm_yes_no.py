from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import pandas as pd

from beyond_click_sim.evaluation import (
    binary_classification_metrics,
    grouped_binary_classification_metrics,
)
from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMInteractionYesNoScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods.common import (
    candidate_group_summary,
    current_git_commit,
    limit_candidate_groups,
    prediction_frame,
    task_xy,
    write_json,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root


OLLAMA_LLAMA31_8B_METHOD_NAME = "llm_yes_no_ollama_llama31_8b"
OLLAMA_LLAMA31_8B_CLIENT = "ollama_local"
OLLAMA_LLAMA31_8B_MODEL = "llama3.1:8b"
VLLM_LLAMA33_70B_METHOD_NAME = "llm_yes_no_vllm_llama33_70b"
VLLM_LLAMA33_70B_CLIENT = "vllm_local"
VLLM_LLAMA33_70B_MODEL = "llama-3.3-70b-instruct"
MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_LLM_ERRORS = 3
OLLAMA_MAX_WORKERS = 1
VLLM_MAX_WORKERS = 32

DATASET_PROMPT_COLUMNS = {
    "ml-1m": {
        "history_description_columns": ("item_title", "item_genres", "rating"),
        "candidate_description_columns": ("item_title", "item_genres"),
    },
    "steam": {
        "history_description_columns": (
            "item_title",
            "item_genres_json",
            "item_tags_json",
            "playtime_forever",
        ),
        "candidate_description_columns": (
            "item_title",
            "item_genres_json",
            "item_tags_json",
        ),
    },
}


def run_llama31_8b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_smoke",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=25,
        max_workers=OLLAMA_MAX_WORKERS,
    )


def run_llama31_8b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_full",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=None,
        max_workers=OLLAMA_MAX_WORKERS,
    )


def run_llama33_70b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_smoke",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
    )


def run_llama33_70b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_full",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
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
    max_llm_errors: int = MAX_LLM_ERRORS,
    max_workers: int = 1,
) -> dict[str, object]:
    """Run the yes/no LLM scorer for pointwise interaction alignment."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    prompt_columns = DATASET_PROMPT_COLUMNS[dataset_name]
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("LLM yes/no method requires candidate_group_column")

    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = limit_candidate_groups(
        *xy["test"],
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )

    scorer = LLMInteractionYesNoScorer(
        client=make_llm_client(client_name),
        model=model,
        history_description_columns=prompt_columns["history_description_columns"],
        candidate_description_columns=prompt_columns["candidate_description_columns"],
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
    ).fit(X_train, y_train)

    scores, errors = _score_groups(
        scorer,
        X_test,
        candidate_group_column=candidate_group_column,
        max_errors=max_llm_errors,
        max_workers=max_workers,
    )
    valid = scores.notna()
    if not valid.any():
        raise RuntimeError("LLM scorer did not produce any valid scores")

    valid_scores = scores.loc[valid]
    valid_X = X_test.loc[valid].copy()
    valid_y = y_test.loc[valid].copy()
    predictions = valid_scores.astype(bool).rename("prediction")
    macro_metrics = grouped_binary_classification_metrics(
        valid_y,
        predictions,
        valid_X[candidate_group_column],
    )
    micro_metrics = binary_classification_metrics(valid_y, predictions)
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

    prediction_frame(
        split="test",
        X=valid_X,
        y=valid_y,
        scores=valid_scores,
        predictions=predictions,
    ).to_parquet(output_dir / "predictions.parquet", index=False)
    _write_errors(output_dir / "llm_errors.jsonl", errors)

    root = repo_root()
    manifest = {
        "method": method_name,
        "scorer": {
            "class": "LLMInteractionYesNoScorer",
            "client_name": client_name,
            "model": model,
            "max_history_items": max_history_items,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_columns": prompt_columns,
        },
        "decision_rule": {
            "kind": "hard_binary_yes_no_parser",
            "threshold": None,
        },
        "limits": {
            "max_candidate_groups": max_candidate_groups,
            "max_llm_errors": max_llm_errors,
            "max_workers": max_workers,
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
        "main_metric": "test.macro_by_group.f1",
        "test": {
            "macro_by_group": macro_metrics,
            "micro": micro_metrics,
        },
        "llm_errors": len(errors),
        "scored_rows": int(valid.sum()),
        "requested_rows": int(len(X_test)),
        "max_candidate_groups": max_candidate_groups,
        "max_workers": max_workers,
        "candidate_groups": {
            "requested": requested_candidate_groups,
            "scored": scored_candidate_groups,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "metrics.json", result)
    return result


def _score_groups(
    scorer: LLMInteractionYesNoScorer,
    X: pd.DataFrame,
    *,
    candidate_group_column: str,
    max_errors: int,
    max_workers: int = 1,
) -> tuple[pd.Series, list[dict[str, str]]]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    scores = pd.Series(index=X.index, dtype=float, name="score")
    errors: list[dict[str, str]] = []
    groups = list(X.groupby(candidate_group_column, sort=False))

    if max_workers == 1:
        for group_id, group in groups:
            group_scores, error = _score_one_group(scorer, group_id, group)
            if error is not None:
                errors.append(error)
                if len(errors) >= max_errors:
                    break
            else:
                scores.loc[group_scores.index] = group_scores
        return scores, errors

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_score_one_group, scorer, group_id, group)
            for group_id, group in groups
        ]
        for future in as_completed(futures):
            group_scores, error = future.result()
            if error is not None:
                errors.append(error)
                if len(errors) >= max_errors:
                    for pending in futures:
                        pending.cancel()
                    break
            else:
                scores.loc[group_scores.index] = group_scores
    return scores, errors


def _score_one_group(
    scorer: LLMInteractionYesNoScorer,
    group_id: object,
    group: pd.DataFrame,
) -> tuple[pd.Series, dict[str, str] | None]:
    try:
        return scorer.score(group), None
    except Exception as error:  # noqa: BLE001 - keep long LLM runs alive.
        empty_scores = pd.Series(index=group.index, dtype=float, name="score")
        return empty_scores, {"candidate_group": str(group_id), "error": repr(error)}


def _write_errors(path: Path, errors: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, sort_keys=True) + "\n")
