from __future__ import annotations

import re
from pathlib import Path

from beyond_click_sim.evaluation.policy_ranking import policy_ranking_agreement_metrics
from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import Agent4RecProfileGenerator, Agent4RecYesNoScorer
from beyond_click_sim.scorers.agent4rec.profiles import (
    AGENT4REC_GAME_ACTIVITY_DESCRIPTIONS,
    AGENT4REC_GAME_DIVERSITY_DESCRIPTIONS,
)
from beyond_click_sim.scorers.agent4rec.prompts import (
    AGENT4REC_PLAYTIME_TASTE_PROMPT_VERSION,
    AGENT4REC_TASTE_PROMPT_VERSION,
)
from beyond_click_sim.tasks import Task

from runners.in_distribution.llm_error_budget import (
    DEFAULT_MAX_ERROR_RATE,
    DEFAULT_MIN_GROUPS_BEFORE_CHECK,
    LLMErrorRateExceededError,
)
from runners.in_distribution.item_summaries import (
    Agent4RecSummaryUsage,
    ITEM_SUMMARY_COLUMN,
    ITEM_SUMMARY_COLUMN_LABEL,
    canonical_agent4rec_summary_usage,
    resolve_agent4rec_summary_usage,
    task_item_summary_metadata,
)
from runners.in_distribution.policy_ranking_agreement.metrics import (
    MAIN_METRIC,
    METRICS_FILENAME,
    UTILITY_AGGREGATION,
)
from runners.in_distribution.policy_ranking_agreement.methods.common import (
    compute_policy_utilities,
    current_git_commit,
    json_safe,
    task_xy,
    write_json,
    write_policy_metrics,
)
from runners.in_distribution.policy_ranking_agreement.methods.llm_yes_no import (
    _score_groups,
    _write_errors,
    majority_vote,
)
from runners.in_distribution.policy_ranking_agreement.task_builders import repo_root


VLLM_QWEN36_27B_METHOD_NAME = "agent4rec_yes_no_vllm_qwen36_27b"
VLLM_QWEN36_27B_CLIENT = "vllm_local"
VLLM_QWEN36_27B_MODEL = "Qwen/Qwen3.6-27B"
QWEN_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}
OPENAI_CLIENT = "openai"
GPT4O_MINI_TASTE_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 1000
MAX_HISTORY_ITEMS = 20
MAX_LLM_ATTEMPTS = 5
TASTE_TEMPERATURE = 0.0
TASTE_MAX_TOKENS = None
VLLM_MAX_WORKERS = 128

DATASET_CANDIDATE_COLUMNS = {
    "ml-1m": ("item_title", "item_genres"),
    "steam": ("item_title", "item_genres_json", "item_tags_json"),
}
DATASET_COLUMN_LABELS = {
    "ml-1m": {
        "item_title": "movie title",
        "item_genres": "genres",
    },
    "steam": {
        "item_title": "game title",
        "item_genres_json": "genres",
        "item_tags_json": "tags",
    },
}
DATASET_PROFILE_GENERATOR_KWARGS = {
    "ml-1m": {},
    "steam": {
        "genre_column": "item_genres_json",
        "tag_column": "item_tags_json",
        "title_column": "item_title",
        "playtime_column": "playtime_forever",
        "include_conformity": False,
        "taste_prompt_kind": "playtime",
        "activity_descriptions": AGENT4REC_GAME_ACTIVITY_DESCRIPTIONS,
        "diversity_descriptions": AGENT4REC_GAME_DIVERSITY_DESCRIPTIONS,
    },
}
DATASET_PROMPT_KWARGS = {
    "ml-1m": {},
    "steam": {
        "domain_name": "game",
        "taste_label": "game tastes",
        "entity_field": "GAME",
        "entity_name": "game",
        "entity_plural": "games",
    },
}
DATASET_SUPPORTS_TASTE = {
    "ml-1m": True,
    "steam": True,
}
DATASET_TASTE_PROMPT_VERSION = {
    "ml-1m": AGENT4REC_TASTE_PROMPT_VERSION,
    "steam": AGENT4REC_PLAYTIME_TASTE_PROMPT_VERSION,
}


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
        summary_usage=canonical_agent4rec_summary_usage(task),
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
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=(
            f"{VLLM_QWEN36_27B_METHOD_NAME}_traits_taste_gpt4o_mini_smoke"
        ),
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=(
            f"{VLLM_QWEN36_27B_METHOD_NAME}_traits_taste_gpt4o_mini_full"
        ),
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
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
    max_workers: int = 1,
    extra_body: dict | None = None,
    summary_usage: Agent4RecSummaryUsage = "candidate",
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
    min_groups_before_check: int = DEFAULT_MIN_GROUPS_BEFORE_CHECK,
) -> dict[str, object]:
    """Run the Agent4Rec profile-based yes/no scorer for policy ranking agreement."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    if dataset_name not in DATASET_CANDIDATE_COLUMNS:
        raise ValueError(
            "Agent4Rec yes/no policy ranking has no dataset config for "
            f"dataset: {dataset_name!r}"
        )
    resolved_summary_usage = resolve_agent4rec_summary_usage(summary_usage)
    if resolved_summary_usage["profile"]:
        raise ValueError(
            "Agent4Rec policy ranking supports summaries only for candidates"
        )
    item_summary_metadata = task_item_summary_metadata(
        task,
        candidate=resolved_summary_usage["candidate"],
    )
    candidate_columns = DATASET_CANDIDATE_COLUMNS[dataset_name]
    column_labels = dict(DATASET_COLUMN_LABELS[dataset_name])
    if resolved_summary_usage["candidate"]:
        candidate_columns = (*candidate_columns, ITEM_SUMMARY_COLUMN)
        column_labels[ITEM_SUMMARY_COLUMN] = ITEM_SUMMARY_COLUMN_LABEL

    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = xy["test"]

    # One LLM group per (user, policy) — the natural unit for policy ranking.
    X_test = X_test.copy()
    X_test["_llm_group_"] = (
        X_test["user_id"].astype(str) + "::" + X_test["policy"].astype(str)
    )

    # Proportional smoke cap per policy so every policy stays represented.
    if max_candidate_groups is not None:
        policy_to_groups = (
            X_test.groupby("policy")["_llm_group_"].unique().to_dict()
        )
        n_policies = len(policy_to_groups)
        per_policy = max(1, max_candidate_groups // n_policies)
        kept_groups: set[str] = set()
        for groups in policy_to_groups.values():
            kept_groups.update(groups[:per_policy])
        X_test = X_test[X_test["_llm_group_"].isin(kept_groups)].copy()
        y_test = y_test.loc[X_test.index]

    profile_user_ids = X_test["user_id"].drop_duplicates().tolist()

    uses_taste = "taste" in profile_components
    if uses_taste and not DATASET_SUPPORTS_TASTE[dataset_name]:
        raise ValueError(
            "Agent4Rec taste profiles are not configured for "
            f"dataset: {dataset_name!r}"
        )
    if uses_taste:
        if not taste_client_name:
            raise ValueError("taste_client_name is required for taste profiles")
        if not taste_model:
            raise ValueError("taste_model is required for taste profiles")
        taste_prompt_version = DATASET_TASTE_PROMPT_VERSION[dataset_name]
        taste_cache_path = agent4rec_taste_cache_path(
            task,
            taste_model=taste_model,
            taste_prompt_version=taste_prompt_version,
        )
        taste_client = make_llm_client(taste_client_name)
    else:
        taste_prompt_version = DATASET_TASTE_PROMPT_VERSION[dataset_name]
        taste_cache_path = None
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
        **DATASET_PROFILE_GENERATOR_KWARGS[dataset_name],
    )
    scorer = Agent4RecYesNoScorer(
        client=make_llm_client(client_name),
        model=model,
        profile_generator=profile_generator,
        candidate_description_columns=candidate_columns,
        column_labels=column_labels,
        candidate_group_column="_llm_group_",
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        **DATASET_PROMPT_KWARGS[dataset_name],
    ).fit(X_train, y_train, profile_user_ids=profile_user_ids)

    if uses_taste:
        scorer.build_taste(X_test)

    try:
        scores, errors = _score_groups(
            scorer,
            X_test,
            candidate_group_column="_llm_group_",
            max_attempts=max_llm_attempts,
            max_workers=max_workers,
            run_count=1,
            tie_break_fn=majority_vote,
            method_name=method_name,
            task_name=task.name,
            max_error_rate=max_error_rate,
            min_groups_before_check=min_groups_before_check,
        )
    except LLMErrorRateExceededError as error:
        _write_errors(output_dir / "llm_errors.jsonl", error.errors)
        raise

    valid = scores.notna()
    X_scored = X_test.loc[valid].drop(columns=["_llm_group_"])
    y_scored = y_test.loc[valid]
    scores_scored = scores.loc[valid]

    simulated_utilities, real_utilities = compute_policy_utilities(
        X_scored, y_scored, scores_scored, policy_column="policy",
    )
    policy_names = sorted(simulated_utilities)
    agreement = policy_ranking_agreement_metrics(
        policy_names,
        [simulated_utilities[p] for p in policy_names],
        [real_utilities[p] for p in policy_names],
    )

    predictions = X_test.drop(columns=["_llm_group_"]).copy()
    predictions.insert(0, "split", "test")
    predictions["target"] = y_test.to_numpy()
    predictions["score"] = scores.to_numpy()
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _write_errors(output_dir / "llm_errors.jsonl", errors)

    requested_groups = int(X_test["_llm_group_"].nunique())

    metrics = {
        "method": method_name,
        "task": task.name,
        "protocol": "policy_ranking",
        "main_metric": MAIN_METRIC,
        "utility_aggregation": UTILITY_AGGREGATION,
        "test": agreement,
        "llm_errors": len(errors),
        "scored_rows": int(valid.sum()),
        "requested_rows": int(len(scores)),
        "requested_groups": requested_groups,
        "max_candidate_groups": max_candidate_groups,
    }
    manifest = {
        "method": method_name,
        "protocol": "policy_ranking",
        "scorer": {
            "class": "Agent4RecYesNoScorer",
            "fit_on": "task.train",
            "client_name": client_name,
            "model": model,
            "candidate_group": "user_id::policy",
            "candidate_description_columns": list(candidate_columns),
            "column_labels": column_labels,
            "max_history_items": max_history_items,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "profile_generator": scorer.profile_generator.manifest(),
            "extra_body": extra_body,
            "prompt": DATASET_PROMPT_KWARGS[dataset_name],
            "summary_usage": summary_usage,
            "item_summaries": item_summary_metadata,
        },
        "utility_aggregation": UTILITY_AGGREGATION,
        "limits": {
            "max_candidate_groups": max_candidate_groups,
            "max_llm_attempts": max_llm_attempts,
            "max_workers": max_workers,
            "max_error_rate": max_error_rate,
            "min_groups_before_check": min_groups_before_check,
        },
        "llm_errors": len(errors),
        "groups": {"requested": requested_groups},
        "task": {"name": task.name, "manifest": json_safe(task.manifest)},
        "git_commit": current_git_commit(repo_root()),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / METRICS_FILENAME, metrics)
    write_policy_metrics(task, output_dir)

    return metrics


def agent4rec_taste_cache_path(
    task: Task,
    *,
    taste_model: str,
    taste_prompt_version: str,
) -> Path:
    dataset_name = str(task.manifest["dataset"])
    dataset_version = str(task.manifest["dataset_version"])
    split_seed = task.manifest["splitter"]["seed"]
    model_slug = _cache_slug(taste_model)
    return (
        repo_root()
        / "outputs"
        / "agent4rec_taste_cache"
        / (
            f"{dataset_name}_{dataset_version}_seed{split_seed}_"
            f"{model_slug}_{taste_prompt_version}.jsonl"
        )
    )


def _cache_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not slug:
        raise ValueError(f"Cannot build cache slug from value: {value!r}")
    return slug
