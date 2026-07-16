from __future__ import annotations

from pathlib import Path

import pandas as pd

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMInteractionListwiseRankingScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods._listwise import (
    evaluate_listwise_scorer,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    limit_candidate_groups,
    task_xy,
)
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    DATASET_JSON_LIST_COLUMNS,
    DATASET_PROMPT_COLUMNS,
    QWEN_EXTRA_BODY,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root
from runners.in_distribution.llm_item_stats import (
    item_rating_column_labels,
    maybe_add_item_rating_prompt_columns,
)


VLLM_QWEN36_27B_METHOD_NAME = "llm_listwise_ranking_vllm_qwen36_27b"
VLLM_QWEN36_27B_CLIENT = "vllm_local"
VLLM_QWEN36_27B_MODEL = "Qwen/Qwen3.6-27B"
MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 512
MAX_LLM_ATTEMPTS = 5
VLLM_MAX_WORKERS = 32


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
    extra_body: dict | None = None,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run direct listwise interaction ranking with a validation threshold."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    prompt_columns = maybe_add_item_rating_prompt_columns(
        dataset_name,
        DATASET_PROMPT_COLUMNS[dataset_name],
        use_item_stats=use_item_stats,
    )
    column_labels = item_rating_column_labels(
        dataset_name,
        use_item_stats=use_item_stats,
    )
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("LLM listwise ranking method requires candidate_group_column")

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
        [X_val["user_id"], X_test["user_id"]],
        ignore_index=True,
    ).drop_duplicates()

    scorer = LLMInteractionListwiseRankingScorer(
        client=make_llm_client(client_name),
        model=model,
        history_description_columns=prompt_columns["history_description_columns"],
        candidate_description_columns=prompt_columns[
            "candidate_description_columns"
        ],
        column_labels=column_labels,
        json_list_columns=DATASET_JSON_LIST_COLUMNS[dataset_name],
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    ).fit(X_train, y_train, history_user_ids=history_user_ids.tolist())

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
        },
        repo_root=repo_root(),
        source_metadata=source_metadata,
    )
