from __future__ import annotations

from pathlib import Path

import pandas as pd

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import (
    Agent4RecPreferenceListwiseRankingScorer,
    Agent4RecProfileGenerator,
)
from beyond_click_sim.scorers.agent4rec.prompts import AGENT4REC_TASTE_PROMPT_VERSION
from beyond_click_sim.tasks import split_xy
from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.cold_start.methods._listwise import (
    evaluate_listwise_scorer,
)
from runners.in_distribution.cold_start.methods.agent4rec_yes_no import (
    DATASET_CANDIDATE_COLUMNS,
    DATASET_COLUMN_LABELS,
    GPT4O_MINI_TASTE_MODEL,
    MAX_HISTORY_ITEMS,
    MAX_LLM_ATTEMPTS,
    MAX_TOKENS,
    OPENAI_CLIENT,
    QWEN_EXTRA_BODY,
    TASTE_MAX_TOKENS,
    TASTE_TEMPERATURE,
    TEMPERATURE,
    VLLM_MAX_WORKERS,
    VLLM_QWEN36_27B_CLIENT,
    VLLM_QWEN36_27B_MODEL,
    VLLM_QWEN36_35B_A3B_CLIENT,
    VLLM_QWEN36_35B_A3B_MODEL,
    _require_columns,
    agent4rec_taste_cache_path,
)
from runners.in_distribution.cold_start.methods.llm_listwise_ranking import (
    TARGET_DESCRIPTIONS,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    limit_candidate_groups,
    task_xy,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root
from runners.in_distribution.item_summaries import (
    Agent4RecSummaryUsage,
    ITEM_SUMMARY_COLUMN,
    ITEM_SUMMARY_COLUMN_LABEL,
    canonical_agent4rec_summary_usage,
    resolve_agent4rec_summary_usage,
    task_item_summary_metadata,
)
from runners.in_distribution.llm_error_budget import (
    DEFAULT_MAX_ERROR_RATE,
    DEFAULT_MIN_GROUPS_BEFORE_CHECK,
)


VLLM_QWEN36_27B_TRAITS_METHOD_NAME = (
    "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits"
)
VLLM_QWEN36_27B_TRAITS_TASTE_METHOD_NAME = (
    "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_"
    "traits_taste_gpt4o_mini_candidate_summary"
)
VLLM_QWEN36_27B_TRAITS_TASTE_SUMBOTH_METHOD_NAME = (
    "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_"
    "traits_taste_gpt4o_mini_sumboth"
)
VLLM_QWEN36_35B_A3B_TRAITS_METHOD_NAME = (
    "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_traits"
)
VLLM_QWEN36_35B_A3B_TRAITS_TASTE_METHOD_NAME = (
    "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_"
    "traits_taste_gpt4o_mini_candidate_summary"
)
VLLM_QWEN36_35B_A3B_TRAITS_TASTE_SUMBOTH_METHOD_NAME = (
    "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_"
    "traits_taste_gpt4o_mini_sumboth"
)
VLLM_QWEN3_8B_CLIENT = "vllm_local"
VLLM_QWEN3_8B_MODEL = "Qwen/Qwen3-8B"
VLLM_QWEN3_8B_METHOD_BASE = "agent4rec_preference_listwise_ranking_vllm_qwen3_8b"
VLLM_LLAMA33_70B_CLIENT = "vllm_local"
VLLM_LLAMA33_70B_MODEL = "llama-3.3-70b-instruct"
VLLM_LLAMA33_70B_METHOD_BASE = "agent4rec_preference_listwise_ranking_vllm_llama33_70b"


def run_qwen36_27b_traits_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_TRAITS_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits",),
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen36_27b_traits_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_TRAITS_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits",),
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_base=VLLM_QWEN36_27B_TRAITS_TASTE_METHOD_NAME,
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        suffix="smoke",
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_base=VLLM_QWEN36_27B_TRAITS_TASTE_METHOD_NAME,
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        suffix="full",
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_sumboth_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_base=VLLM_QWEN36_27B_TRAITS_TASTE_SUMBOTH_METHOD_NAME,
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        suffix="smoke",
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_base=VLLM_QWEN36_27B_TRAITS_TASTE_SUMBOTH_METHOD_NAME,
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        suffix="full",
    )


def run_qwen36_35b_a3b_traits_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_TRAITS_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits",),
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen36_35b_a3b_traits_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_TRAITS_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits",),
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen36_35b_a3b_traits_taste_gpt4o_mini_candidate_summary_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_base=VLLM_QWEN36_35B_A3B_TRAITS_TASTE_METHOD_NAME,
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        suffix="smoke",
    )


def run_qwen36_35b_a3b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_base=VLLM_QWEN36_35B_A3B_TRAITS_TASTE_METHOD_NAME,
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        suffix="full",
    )


def run_qwen36_35b_a3b_traits_taste_gpt4o_mini_sumboth_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_base=VLLM_QWEN36_35B_A3B_TRAITS_TASTE_SUMBOTH_METHOD_NAME,
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        suffix="smoke",
    )


def run_qwen36_35b_a3b_traits_taste_gpt4o_mini_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_base=VLLM_QWEN36_35B_A3B_TRAITS_TASTE_SUMBOTH_METHOD_NAME,
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        suffix="full",
    )


def run_qwen3_8b_traits_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_BASE,
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_base=f"{VLLM_QWEN3_8B_METHOD_BASE}_traits_taste_gpt4o_mini_candidate_summary",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_candidate_groups=None,
        suffix="full",
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_base=f"{VLLM_QWEN3_8B_METHOD_BASE}_traits_taste_gpt4o_mini_sumboth",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_candidate_groups=None,
        suffix="full",
        extra_body=QWEN_EXTRA_BODY,
    )


def run_llama33_70b_traits_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_BASE,
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
    )


def run_llama33_70b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_base=f"{VLLM_LLAMA33_70B_METHOD_BASE}_traits_taste_gpt4o_mini_candidate_summary",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None,
        suffix="full",
        extra_body=None,
    )


def run_llama33_70b_traits_taste_gpt4o_mini_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_base=f"{VLLM_LLAMA33_70B_METHOD_BASE}_traits_taste_gpt4o_mini_sumboth",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None,
        suffix="full",
        extra_body=None,
    )


def _run_traits_full(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_base: str,
    client_name: str,
    model: str,
    extra_body: dict | None,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{method_base}_traits_full",
        client_name=client_name,
        model=model,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=extra_body,
        profile_components=("traits",),
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def _run_traits_taste_candidate_summary(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_base: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
    suffix: str,
    extra_body: dict | None = QWEN_EXTRA_BODY,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{method_base}_{suffix}",
        client_name=client_name,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=extra_body,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        summary_usage="candidate",
    )


def _run_traits_taste_sumboth(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_base: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
    suffix: str,
    extra_body: dict | None = QWEN_EXTRA_BODY,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{method_base}_{suffix}",
        client_name=client_name,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=extra_body,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        summary_usage="both",
    )


def run_method(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
    max_workers: int,
    profile_components: tuple[str, ...] = ("traits",),
    taste_client_name: str | None = None,
    taste_model: str | None = None,
    taste_temperature: float = TASTE_TEMPERATURE,
    taste_max_tokens: int | None = TASTE_MAX_TOKENS,
    taste_max_attempts: int = MAX_LLM_ATTEMPTS,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_history_items: int = MAX_HISTORY_ITEMS,
    max_llm_attempts: int = MAX_LLM_ATTEMPTS,
    extra_body: dict | None = None,
    summary_usage: Agent4RecSummaryUsage = "candidate",
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
    min_groups_before_check: int = DEFAULT_MIN_GROUPS_BEFORE_CHECK,
) -> dict[str, object]:
    """Run Agent4Rec direct preference ranking for cold-start val/test groups."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    try:
        target_description = TARGET_DESCRIPTIONS[dataset_name]
        candidate_columns = DATASET_CANDIDATE_COLUMNS[dataset_name]
        column_labels = dict(DATASET_COLUMN_LABELS[dataset_name])
    except KeyError as error:
        raise ValueError(
            f"Unsupported cold-start Agent4Rec listwise dataset: {dataset_name!r}"
        ) from error

    uses_taste = "taste" in profile_components
    resolved_summary_usage = resolve_agent4rec_summary_usage(summary_usage)
    if resolved_summary_usage["profile"] and not uses_taste:
        raise ValueError("Agent4Rec profile summaries require taste profiles")
    if resolved_summary_usage["candidate"]:
        candidate_columns = (*candidate_columns, ITEM_SUMMARY_COLUMN)
        column_labels[ITEM_SUMMARY_COLUMN] = ITEM_SUMMARY_COLUMN_LABEL
    item_summary_metadata = task_item_summary_metadata(
        task,
        profile=resolved_summary_usage["profile"],
        candidate=resolved_summary_usage["candidate"],
        extra_frames={"online_session_history": task.online_session_history},
    )

    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("Agent4Rec listwise method requires candidate_group_column")

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
    X_eval = pd.concat([X_val, X_test], ignore_index=True)
    profile_user_ids = X_eval["user_id"].drop_duplicates().tolist()
    _require_columns(X_eval, list(candidate_columns))

    if uses_taste:
        if not taste_client_name or not taste_model:
            raise ValueError(
                "taste_client_name and taste_model are required for taste profiles"
            )
        taste_cache_path = agent4rec_taste_cache_path(
            task,
            taste_model=taste_model,
            taste_prompt_version=AGENT4REC_TASTE_PROMPT_VERSION,
            use_history_item_summaries=resolved_summary_usage["profile"],
        )
        taste_client = make_llm_client(taste_client_name)
    else:
        taste_cache_path = None
        taste_client = None

    profile_generator = Agent4RecProfileGenerator(
        profile_components=profile_components,
        activity_counts=None,
        taste_client=taste_client,
        taste_client_name=taste_client_name,
        taste_model=taste_model,
        taste_cache_path=taste_cache_path,
        taste_prompt_version=AGENT4REC_TASTE_PROMPT_VERSION,
        taste_temperature=taste_temperature,
        taste_max_tokens=taste_max_tokens,
        taste_max_attempts=taste_max_attempts,
        summary_column=(
            ITEM_SUMMARY_COLUMN if resolved_summary_usage["profile"] else None
        ),
    )
    scorer = Agent4RecPreferenceListwiseRankingScorer(
        client=make_llm_client(client_name),
        model=model,
        target_description=target_description,
        profile_generator=profile_generator,
        candidate_description_columns=candidate_columns,
        column_labels=column_labels,
        json_list_columns=(),
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    ).fit(X_history, y_history, profile_user_ids=profile_user_ids)
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
        max_error_rate=max_error_rate,
        min_groups_before_check=min_groups_before_check,
        scorer_manifest={
            "class": scorer.__class__.__name__,
            "fit_on": "online_session_history",
            "k": task.k,
            "client_name": client_name,
            "model": model,
            "target_description": target_description,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_history_items": max_history_items,
            "candidate_description_columns": list(candidate_columns),
            "column_labels": column_labels,
            "json_list_columns": [],
            "profile_generator": scorer.profile_generator.manifest(),
            "extra_body": extra_body,
            "summary_usage": summary_usage,
            "item_summaries": item_summary_metadata,
        },
        repo_root=repo_root(),
    )
