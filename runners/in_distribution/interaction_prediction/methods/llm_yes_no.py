from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMInteractionYesNoScorer
from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.llm_item_stats import (
    item_rating_column_labels,
    maybe_add_item_rating_prompt_columns,
)
from runners.in_distribution.item_summaries import (
    ITEM_SUMMARY_COLUMN,
    ITEM_SUMMARY_COLUMN_LABEL,
    SummaryVisibility,
    maybe_add_item_summary_prompt_columns,
    resolve_item_summary_visibility,
    task_item_summary_metadata,
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
from runners.in_distribution.interaction_prediction.metrics import (
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
LITELLM_CLIENT_NAME = "litellm_local"
LITELLM_QWEN3_8B_METHOD_NAME = "llm_yes_no_litellm_qwen3_8b"
LITELLM_QWEN3_8B_MODEL = "Qwen/Qwen3-8B"
LITELLM_QWEN3_8B_MAX_WORKERS = 64
LITELLM_QWEN36_27B_METHOD_NAME = "llm_yes_no_litellm_qwen36_27b"
LITELLM_QWEN36_27B_MODEL = "Qwen/Qwen3.6-27B"
LITELLM_QWEN36_27B_MAX_WORKERS = 32
OPENAI_VK_GPT54_MINI_METHOD_NAME = "llm_yes_no_openai_vk_gpt54_mini"
OPENAI_VK_GPT54_MINI_CLIENT = "openai_vk_proxy"
OPENAI_VK_GPT54_MINI_MODEL = "gpt-5.4-mini"
OPENAI_VK_GPT55_METHOD_NAME = "llm_yes_no_openai_vk_gpt55"
OPENAI_VK_GPT55_CLIENT = "openai_vk_proxy"
OPENAI_VK_GPT55_MODEL = "gpt-5.5"
QWEN_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}
MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_LLM_ATTEMPTS = 5
OLLAMA_MAX_WORKERS = 1
VLLM_MAX_WORKERS = 32
OPENAI_VK_MAX_WORKERS = 4

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
DATASET_JSON_LIST_COLUMNS = {
    "ml-1m": (),
    "steam": ("item_genres_json", "item_tags_json"),
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
        max_candidate_groups=25,
        max_workers=OLLAMA_MAX_WORKERS,
        use_item_stats=True,
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
        max_candidate_groups=None,
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
        max_candidate_groups=25,
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
        max_candidate_groups=25,
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
        max_candidate_groups=None,
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
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        use_item_stats=True,
    )


def run_qwen36_27b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
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
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        use_item_stats=True,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen36_27b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
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
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        use_item_stats=True,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_litellm_qwen3_8b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_litellm_qwen_with_item_stats(
        task,
        output_dir,
        method_name=f"{LITELLM_QWEN3_8B_METHOD_NAME}_with_item_stats_smoke",
        model=LITELLM_QWEN3_8B_MODEL,
        max_candidate_groups=25,
        max_workers=LITELLM_QWEN3_8B_MAX_WORKERS,
    )


def run_litellm_qwen3_8b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_litellm_qwen_with_item_stats(
        task,
        output_dir,
        method_name=f"{LITELLM_QWEN3_8B_METHOD_NAME}_with_item_stats_full",
        model=LITELLM_QWEN3_8B_MODEL,
        max_candidate_groups=None,
        max_workers=LITELLM_QWEN3_8B_MAX_WORKERS,
    )


def run_litellm_qwen36_27b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_litellm_qwen_with_item_stats(
        task,
        output_dir,
        method_name=f"{LITELLM_QWEN36_27B_METHOD_NAME}_with_item_stats_smoke",
        model=LITELLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=LITELLM_QWEN36_27B_MAX_WORKERS,
    )


def run_litellm_qwen36_27b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_litellm_qwen_with_item_stats(
        task,
        output_dir,
        method_name=f"{LITELLM_QWEN36_27B_METHOD_NAME}_with_item_stats_full",
        model=LITELLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=LITELLM_QWEN36_27B_MAX_WORKERS,
    )


def _run_litellm_qwen_with_item_stats(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    model: str,
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


def run_gpt54_mini_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OPENAI_VK_GPT54_MINI_METHOD_NAME}_smoke",
        client_name=OPENAI_VK_GPT54_MINI_CLIENT,
        model=OPENAI_VK_GPT54_MINI_MODEL,
        max_candidate_groups=25,
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
        max_candidate_groups=25,
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
        max_candidate_groups=None,
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
        max_candidate_groups=None,
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
        max_candidate_groups=25,
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
        max_candidate_groups=25,
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
        max_candidate_groups=None,
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
        max_candidate_groups=None,
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
    max_candidate_groups: int | None,
    max_history_items: int = MAX_HISTORY_ITEMS,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_llm_attempts: int = MAX_LLM_ATTEMPTS,
    max_workers: int = 1,
    use_item_stats: bool = False,
    summary_visibility: SummaryVisibility = "none",
    extra_body: dict | None = None,
    scorer_class: type[LLMInteractionYesNoScorer] = LLMInteractionYesNoScorer,
    scorer_kwargs: dict[str, object] | None = None,
    serving_metadata: dict[str, object] | None = None,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run a history-conditioned yes/no LLM scorer on candidate groups."""

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

    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = limit_candidate_groups(
        *xy["test"],
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    history_user_ids = X_test["user_id"].drop_duplicates().tolist()
    item_summary_metadata = task_item_summary_metadata(
        task,
        history=resolved_summary_visibility["history"],
        candidate=resolved_summary_visibility["candidate"],
    )

    scorer = scorer_class(
        client=make_llm_client(client_name),
        model=model,
        history_description_columns=prompt_columns["history_description_columns"],
        candidate_description_columns=prompt_columns["candidate_description_columns"],
        column_labels=column_labels,
        json_list_columns=DATASET_JSON_LIST_COLUMNS[dataset_name],
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        **({} if scorer_kwargs is None else scorer_kwargs),
    ).fit(X_train, y_train, history_user_ids=history_user_ids)

    scores, errors = _score_groups(
        scorer,
        X_test,
        candidate_group_column=candidate_group_column,
        max_attempts=max_llm_attempts,
        max_workers=max_workers,
    )
    valid = scores.notna()
    predictions = scores.astype("boolean").rename("prediction")
    prediction_frame(
        split="test",
        X=X_test,
        y=y_test,
        scores=scores,
        predictions=predictions,
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
    coverage = score_coverage_summary(scores)

    root = repo_root()
    manifest = {
        "method": method_name,
        "scorer": {
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
            "summary_visibility": summary_visibility,
            "item_summaries": item_summary_metadata,
            "extra_body": extra_body,
            "scorer_kwargs": scorer_kwargs,
            "serving": serving_metadata,
        },
        "decision_rule": {
            "kind": "hard_binary_yes_no_parser",
            "threshold": None,
        },
        "limits": {
            "max_candidate_groups": max_candidate_groups,
            "max_llm_attempts": max_llm_attempts,
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
        "source": source_metadata,
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
        "candidate_groups": {
            "requested": requested_candidate_groups,
            "scored": scored_candidate_groups,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / POINTWISE_METRICS_FILENAME, result)
    write_json(output_dir / RANKING_METRICS_FILENAME, ranking_result)
    return result


def _score_groups(
    scorer: Scorer,
    X: pd.DataFrame,
    *,
    candidate_group_column: str,
    max_attempts: int,
    max_workers: int = 1,
) -> tuple[pd.Series, list[dict[str, object]]]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    scores = pd.Series(index=X.index, dtype=float, name="score")
    errors: list[dict[str, object]] = []
    groups = list(X.groupby(candidate_group_column, sort=False))

    if max_workers == 1:
        progress = tqdm(groups, desc="llm groups", unit="group")
        for group_id, group in progress:
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
        for future in progress:
            group_scores, error = future.result()
            if error is not None:
                errors.append(error)
                progress.set_postfix(errors=len(errors))
            else:
                scores.loc[group_scores.index] = group_scores
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


def _serving_metadata() -> dict[str, object]:
    return {
        "backend": "litellm_proxy_over_vllm",
        "litellm_base_url": os.environ.get(
            "BEYOND_CLICK_SIM_LITELLM_LOCAL_BASE_URL",
            "http://127.0.0.1:8080/v1",
        ),
        "litellm_version": os.environ.get(
            "BEYOND_CLICK_SIM_LITELLM_VERSION",
            "unknown",
        ),
        "routing_strategy": os.environ.get(
            "BEYOND_CLICK_SIM_LITELLM_ROUTING_STRATEGY",
            "simple-shuffle",
        ),
        "vllm_version": os.environ.get(
            "BEYOND_CLICK_SIM_VLLM_VERSION",
            "unknown",
        ),
        "vllm_replicas": int(
            os.environ.get("BEYOND_CLICK_SIM_VLLM_REPLICAS", "4")
        ),
        "vllm_ports": [8000, 8001, 8002, 8003],
        "tensor_parallel_size_per_replica": 1,
        "max_model_len": int(
            os.environ.get("BEYOND_CLICK_SIM_VLLM_MAX_MODEL_LEN", "4096")
        ),
        "gpu_memory_utilization": float(
            os.environ.get(
                "BEYOND_CLICK_SIM_VLLM_GPU_MEMORY_UTILIZATION",
                "0.90",
            )
        ),
        "model_revision": os.environ.get(
            "BEYOND_CLICK_SIM_MODEL_REVISION",
            "unknown",
        ),
        "thinking_enabled": False,
    }


def _source_metadata() -> dict[str, object]:
    return {
        "base_git_commit": os.environ.get(
            "BEYOND_CLICK_SIM_SOURCE_BASE_GIT_COMMIT"
        ),
        "snapshot_sha256": os.environ.get(
            "BEYOND_CLICK_SIM_SOURCE_SNAPSHOT_SHA256"
        ),
        "diff_sha256": os.environ.get("BEYOND_CLICK_SIM_SOURCE_DIFF_SHA256"),
    }
