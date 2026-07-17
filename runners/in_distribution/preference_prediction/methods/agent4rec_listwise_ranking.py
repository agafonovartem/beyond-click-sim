from __future__ import annotations

from pathlib import Path

import pandas as pd

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import (
    Agent4RecPreferenceListwiseRankingScorer,
    Agent4RecProfileGenerator,
)
from beyond_click_sim.tasks import Task
from runners.in_distribution.item_summaries import (
    Agent4RecSummaryUsage,
    ITEM_SUMMARY_COLUMN,
    canonical_agent4rec_summary_usage,
    resolve_agent4rec_summary_usage,
    task_item_summary_metadata,
)
from runners.in_distribution.preference_prediction.methods import (
    _grouped_agent4rec_yes_no as grouped_agent4rec,
)
from runners.in_distribution.preference_prediction.methods import (
    agent4rec_yes_no as preference_agent4rec,
)
from runners.in_distribution.preference_prediction.methods._listwise import (
    evaluate_listwise_scorer,
)
from runners.in_distribution.preference_prediction.methods.common import (
    limit_candidate_groups,
    task_xy,
)
from runners.in_distribution.preference_prediction.methods.llm_yes_no import (
    CLIENT_NAME,
    QWEN_EXTRA_BODY,
    QWEN3_8B_MAX_WORKERS,
    QWEN3_8B_MODEL,
    QWEN36_27B_MAX_WORKERS,
    QWEN36_27B_MODEL,
    SMOKE_CANDIDATE_GROUPS,
    TARGET_DESCRIPTIONS,
    _serving_metadata,
    _source_metadata,
)
from runners.in_distribution.preference_prediction.task_builders import repo_root


VLLM_CLIENT_NAME = "vllm_local"
QWEN36_27B_METHOD_NAME = (
    "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits"
)
QWEN3_8B_TRAITS_TASTE_METHOD_NAME = (
    "agent4rec_preference_listwise_ranking_litellm_qwen3_8b_"
    "traits_taste_gpt4o_mini_candidate_summary"
)


def run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary(
        task,
        output_dir,
        max_candidate_groups=SMOKE_CANDIDATE_GROUPS,
        suffix="smoke",
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary(
        task,
        output_dir,
        max_candidate_groups=None,
        suffix="full",
    )


def _run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary(
    task: Task,
    output_dir: Path,
    *,
    max_candidate_groups: int | None,
    suffix: str,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{QWEN3_8B_TRAITS_TASTE_METHOD_NAME}_{suffix}",
        client_name=CLIENT_NAME,
        model=QWEN3_8B_MODEL,
        max_candidate_groups=max_candidate_groups,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=grouped_agent4rec.OPENAI_CLIENT,
        taste_model=grouped_agent4rec.GPT4O_MINI_TASTE_MODEL,
        taste_temperature=grouped_agent4rec.TASTE_TEMPERATURE,
        taste_max_tokens=grouped_agent4rec.TASTE_MAX_TOKENS,
        summary_usage="candidate",
        serving_metadata=_serving_metadata(),
        source_metadata=_source_metadata(),
    )


def run_qwen36_27b_traits_smoke(
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
        summary_usage=canonical_agent4rec_summary_usage(task),
        extra_body=QWEN_EXTRA_BODY,
        source_metadata=_source_metadata(),
    )


def run_qwen36_27b_traits_full(
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
        summary_usage=canonical_agent4rec_summary_usage(task),
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
    temperature: float = grouped_agent4rec.TEMPERATURE,
    max_tokens: int = grouped_agent4rec.MAX_TOKENS,
    max_history_items: int | None = grouped_agent4rec.MAX_HISTORY_ITEMS,
    max_llm_attempts: int = grouped_agent4rec.MAX_LLM_ATTEMPTS,
    extra_body: dict | None = None,
    profile_components: tuple[str, ...] = ("traits",),
    taste_client_name: str | None = None,
    taste_model: str | None = None,
    taste_temperature: float = grouped_agent4rec.TASTE_TEMPERATURE,
    taste_max_tokens: int | None = grouped_agent4rec.TASTE_MAX_TOKENS,
    taste_max_attempts: int = grouped_agent4rec.MAX_LLM_ATTEMPTS,
    taste_prompt_version: str | None = None,
    taste_cache_path: Path | None = None,
    serving_metadata: dict[str, object] | None = None,
    source_metadata: dict[str, object] | None = None,
    summary_usage: Agent4RecSummaryUsage = "candidate",
) -> dict[str, object]:
    """Run Agent4Rec direct preference ranking with validation calibration."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    try:
        target_description = TARGET_DESCRIPTIONS[dataset_name]
        base_candidate_columns = preference_agent4rec.DATASET_CANDIDATE_COLUMNS[
            dataset_name
        ]
        base_column_labels = preference_agent4rec.DATASET_COLUMN_LABELS[
            dataset_name
        ]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Agent4Rec listwise preference dataset: {dataset_name!r}"
        ) from error
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("Agent4Rec listwise method requires candidate_group_column")

    uses_taste = "taste" in profile_components
    resolved_summary_usage = resolve_agent4rec_summary_usage(summary_usage)
    if resolved_summary_usage["profile"] and not uses_taste:
        raise ValueError("Agent4Rec profile summaries require taste profiles")
    if resolved_summary_usage["any"] and dataset_name != "ml-1m":
        raise ValueError(
            "Agent4Rec movie summaries are configured only for ml-1m, got "
            f"{dataset_name!r}"
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
    X_eval = pd.concat([X_val, X_test], ignore_index=True)
    profile_user_ids = X_eval["user_id"].drop_duplicates().tolist()
    candidate_columns = grouped_agent4rec._candidate_columns(
        base_candidate_columns,
        use_item_summaries=resolved_summary_usage["candidate"],
    )
    column_labels = grouped_agent4rec._column_labels(
        base_column_labels,
        use_item_summaries=resolved_summary_usage["candidate"],
    )
    grouped_agent4rec._require_columns(X_eval, list(candidate_columns))
    item_summary_metadata = task_item_summary_metadata(
        task,
        profile=resolved_summary_usage["profile"],
        candidate=resolved_summary_usage["candidate"],
    )

    if uses_taste:
        if not taste_client_name or not taste_model:
            raise ValueError(
                "taste_client_name and taste_model are required for taste profiles"
            )
        if taste_prompt_version is None:
            taste_prompt_version = grouped_agent4rec.DATASET_TASTE_PROMPT_VERSION[
                dataset_name
            ]
        if taste_cache_path is None:
            taste_cache_path = grouped_agent4rec.agent4rec_taste_cache_path(
                task,
                taste_model=taste_model,
                taste_prompt_version=taste_prompt_version,
                use_history_item_summaries=resolved_summary_usage["profile"],
            )
        taste_client = make_llm_client(taste_client_name)
    else:
        taste_prompt_version = (
            taste_prompt_version
            or grouped_agent4rec.DATASET_TASTE_PROMPT_VERSION[dataset_name]
        )
        taste_client = None

    profile_generator = Agent4RecProfileGenerator(
        profile_components=profile_components,
        taste_client=taste_client,
        taste_client_name=taste_client_name,
        taste_model=taste_model,
        taste_cache_path=taste_cache_path,
        taste_prompt_version=taste_prompt_version,
        taste_temperature=taste_temperature,
        taste_max_tokens=taste_max_tokens,
        taste_max_attempts=taste_max_attempts,
        summary_column=(
            ITEM_SUMMARY_COLUMN if resolved_summary_usage["profile"] else None
        ),
        **grouped_agent4rec.DATASET_PROFILE_GENERATOR_KWARGS[dataset_name],
    )
    scorer = Agent4RecPreferenceListwiseRankingScorer(
        client=make_llm_client(client_name),
        model=model,
        target_description=target_description,
        profile_generator=profile_generator,
        candidate_description_columns=candidate_columns,
        column_labels=column_labels,
        json_list_columns=grouped_agent4rec.DATASET_JSON_LIST_COLUMNS[dataset_name],
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        **grouped_agent4rec.DATASET_PROMPT_KWARGS[dataset_name],
    ).fit(X_train, y_train, profile_user_ids=profile_user_ids)
    if uses_taste:
        scorer.build_taste(X_eval)

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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_history_items": max_history_items,
            "candidate_description_columns": list(candidate_columns),
            "column_labels": column_labels,
            "json_list_columns": list(
                grouped_agent4rec.DATASET_JSON_LIST_COLUMNS[dataset_name]
            ),
            "profile_generator": scorer.profile_generator.manifest(),
            "extra_body": extra_body,
            "prompt": grouped_agent4rec.DATASET_PROMPT_KWARGS[dataset_name],
            "summary_usage": summary_usage,
            "item_summaries": item_summary_metadata,
            "serving": serving_metadata,
        },
        repo_root=repo_root(),
        source_metadata=source_metadata,
    )
