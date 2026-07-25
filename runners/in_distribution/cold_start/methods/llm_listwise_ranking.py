from __future__ import annotations

from pathlib import Path

import pandas as pd

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMPreferenceListwiseRankingScorer
from beyond_click_sim.tasks import split_xy
from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.cold_start.methods._listwise import (
    evaluate_listwise_scorer,
)
from runners.in_distribution.cold_start.methods.llm_yes_no import (
    DATASET_PROMPT_COLUMNS,
    QWEN_EXTRA_BODY,
    VLLM_MAX_WORKERS,
    VLLM_QWEN36_27B_CLIENT,
    VLLM_QWEN36_27B_MODEL,
    VLLM_QWEN36_35B_A3B_CLIENT,
    VLLM_QWEN36_35B_A3B_MODEL,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    limit_candidate_groups,
    task_xy,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root
from runners.in_distribution.llm_error_budget import (
    DEFAULT_MAX_ERROR_RATE,
    DEFAULT_MIN_GROUPS_BEFORE_CHECK,
)
from runners.in_distribution.llm_item_stats import (
    item_rating_column_labels,
    maybe_add_item_rating_prompt_columns,
)


VLLM_QWEN36_27B_METHOD_NAME = "llm_preference_listwise_ranking_vllm_qwen36_27b"
VLLM_QWEN36_35B_A3B_METHOD_NAME = (
    "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b"
)
TARGET_DESCRIPTIONS = {
    "ml-1m": "The user would rate the candidate movie at least 4 out of 5.",
}
MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 512
MAX_LLM_ATTEMPTS = 5


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
    )


def run_qwen36_35b_a3b_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen36_35b_a3b_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
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
    extra_body: dict | None = None,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
    min_groups_before_check: int = DEFAULT_MIN_GROUPS_BEFORE_CHECK,
) -> dict[str, object]:
    """Run direct listwise preference ranking on cold-start val/test groups."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    try:
        target_description = TARGET_DESCRIPTIONS[dataset_name]
        base_prompt_columns = DATASET_PROMPT_COLUMNS[dataset_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported cold-start listwise dataset: {dataset_name!r}"
        ) from error
    prompt_columns = maybe_add_item_rating_prompt_columns(
        dataset_name,
        base_prompt_columns,
        use_item_stats=False,
    )
    column_labels = item_rating_column_labels(dataset_name, use_item_stats=False)
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("LLM listwise ranking method requires candidate_group_column")

    # Cold-start fit contract: use each cold user's k-item online history only.
    X_history, y_history = split_xy(
        task.online_session_history,
        target_column=task.schema.target_column,
    )
    xy = task_xy(task)
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

    scorer = LLMPreferenceListwiseRankingScorer(
        client=make_llm_client(client_name),
        model=model,
        target_description=target_description,
        history_description_columns=prompt_columns["history_description_columns"],
        candidate_description_columns=prompt_columns["candidate_description_columns"],
        column_labels=column_labels,
        json_list_columns=(),
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    ).fit(X_history, y_history, history_user_ids=history_user_ids.tolist())

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
        max_error_rate=max_error_rate,
        min_groups_before_check=min_groups_before_check,
        scorer_manifest={
            "class": scorer.__class__.__name__,
            "fit_on": "online_session_history",
            "k": task.k,
            "client_name": client_name,
            "model": model,
            "target_description": target_description,
            "max_history_items": max_history_items,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_columns": prompt_columns,
            "column_labels": column_labels,
            "json_list_columns": [],
            "extra_body": extra_body,
        },
        repo_root=repo_root(),
    )
