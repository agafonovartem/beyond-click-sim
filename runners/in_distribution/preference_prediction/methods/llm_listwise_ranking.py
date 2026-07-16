from __future__ import annotations

from pathlib import Path

import pandas as pd

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMPreferenceListwiseRankingScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.llm_item_stats import (
    item_rating_column_labels,
    maybe_add_item_rating_prompt_columns,
)
from runners.in_distribution.preference_prediction.methods import (
    _grouped_llm_yes_no,
)
from runners.in_distribution.preference_prediction.methods._listwise import (
    evaluate_listwise_scorer,
)
from runners.in_distribution.preference_prediction.methods.common import (
    limit_candidate_groups,
    task_xy,
)
from runners.in_distribution.preference_prediction.methods.llm_yes_no import (
    QWEN_EXTRA_BODY,
    QWEN36_27B_MAX_WORKERS,
    QWEN36_27B_MODEL,
    SMOKE_CANDIDATE_GROUPS,
    TARGET_DESCRIPTIONS,
    _source_metadata,
)
from runners.in_distribution.preference_prediction.task_builders import repo_root


VLLM_CLIENT_NAME = "vllm_local"
QWEN36_27B_METHOD_NAME = "llm_preference_listwise_ranking_vllm_qwen36_27b"
MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 512
MAX_LLM_ATTEMPTS = 5


def run_qwen36_27b_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{QWEN36_27B_METHOD_NAME}_smoke",
        client_name=VLLM_CLIENT_NAME,
        model=QWEN36_27B_MODEL,
        max_candidate_groups=SMOKE_CANDIDATE_GROUPS,
        max_workers=QWEN36_27B_MAX_WORKERS,
        use_item_stats=False,
        extra_body=QWEN_EXTRA_BODY,
        source_metadata=_source_metadata(),
    )


def run_qwen36_27b_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{QWEN36_27B_METHOD_NAME}_full",
        client_name=VLLM_CLIENT_NAME,
        model=QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=QWEN36_27B_MAX_WORKERS,
        use_item_stats=False,
        extra_body=QWEN_EXTRA_BODY,
        source_metadata=_source_metadata(),
    )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    model: str,
    max_candidate_groups: int | None,
    max_workers: int,
    client_name: str = VLLM_CLIENT_NAME,
    max_history_items: int = MAX_HISTORY_ITEMS,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_llm_attempts: int = MAX_LLM_ATTEMPTS,
    use_item_stats: bool = False,
    extra_body: dict | None = None,
    serving_metadata: dict[str, object] | None = None,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run direct listwise preference ranking with a validation threshold."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    try:
        target_description = TARGET_DESCRIPTIONS[dataset_name]
        base_prompt_columns = _grouped_llm_yes_no.DATASET_PROMPT_COLUMNS[dataset_name]
        json_list_columns = _grouped_llm_yes_no.DATASET_JSON_LIST_COLUMNS[
            dataset_name
        ]
    except KeyError as error:
        raise ValueError(
            f"Unsupported listwise preference dataset: {dataset_name!r}"
        ) from error
    prompt_columns = maybe_add_item_rating_prompt_columns(
        dataset_name,
        base_prompt_columns,
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
    scorer = LLMPreferenceListwiseRankingScorer(
        client=make_llm_client(client_name),
        model=model,
        target_description=target_description,
        history_description_columns=prompt_columns["history_description_columns"],
        candidate_description_columns=prompt_columns[
            "candidate_description_columns"
        ],
        column_labels=column_labels,
        json_list_columns=json_list_columns,
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
            "target_description": target_description,
            "max_history_items": max_history_items,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_columns": prompt_columns,
            "column_labels": column_labels,
            "json_list_columns": list(json_list_columns),
            "uses_item_stats": use_item_stats,
            "extra_body": extra_body,
            "serving": serving_metadata,
        },
        repo_root=repo_root(),
        source_metadata=source_metadata,
    )
