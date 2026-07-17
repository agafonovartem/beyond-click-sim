from __future__ import annotations

from pathlib import Path
import re

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import Agent4RecProfileGenerator, Agent4RecYesNoScorer
from beyond_click_sim.scorers.agent4rec.profiles import (
    AGENT4REC_GAME_ACTIVITY_DESCRIPTIONS,
    AGENT4REC_GAME_DIVERSITY_DESCRIPTIONS,
)
from beyond_click_sim.scorers.agent4rec.prompts import AGENT4REC_TASTE_PROMPT_VERSION
from beyond_click_sim.scorers.agent4rec.prompts import (
    AGENT4REC_PLAYTIME_TASTE_PROMPT_VERSION,
)
from beyond_click_sim.tasks import PREFIXED_ITEM_RATING_MEAN_COLUMN, Task
from runners.in_distribution.item_summaries import (
    Agent4RecSummaryUsage,
    ITEM_SUMMARY_COLUMN,
    ITEM_SUMMARY_COLUMN_LABEL,
    canonical_agent4rec_summary_usage,
    resolve_agent4rec_summary_usage,
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
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    _score_groups,
    _write_errors,
)
from runners.in_distribution.interaction_prediction.methods.llm_listwise_ranking import (
    LITELLM_CLIENT_NAME,
    QWEN3_8B_MAX_WORKERS,
    QWEN3_8B_MODEL,
    _serving_metadata,
    _source_metadata,
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


OLLAMA_LLAMA31_8B_METHOD_NAME = "agent4rec_yes_no_ollama_llama31_8b"
OLLAMA_LLAMA31_8B_CLIENT = "ollama_local"
OLLAMA_LLAMA31_8B_MODEL = "llama3.1:8b"
VLLM_LLAMA33_70B_METHOD_NAME = "agent4rec_yes_no_vllm_llama33_70b"
VLLM_LLAMA33_70B_CLIENT = "vllm_local"
VLLM_LLAMA33_70B_MODEL = "llama-3.3-70b-instruct"
VLLM_QWEN36_27B_METHOD_NAME = "agent4rec_yes_no_vllm_qwen36_27b"
VLLM_QWEN36_27B_CLIENT = "vllm_local"
VLLM_QWEN36_27B_PORT8001_METHOD_NAME = (
    "agent4rec_yes_no_vllm_qwen36_27b_port8001"
)
VLLM_QWEN36_27B_PORT8001_CLIENT = "vllm_local_8001"
VLLM_QWEN36_27B_PORT8002_METHOD_NAME = (
    "agent4rec_yes_no_vllm_qwen36_27b_port8002"
)
VLLM_QWEN36_27B_PORT8002_CLIENT = "vllm_local_8002"
VLLM_QWEN36_27B_MODEL = "Qwen/Qwen3.6-27B"
QWEN_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}
OPENAI_CLIENT = "openai"
GPT4O_MINI_TASTE_MODEL = "gpt-4o-mini"
LITELLM_QWEN3_8B_TRAITS_TASTE_METHOD_NAME = (
    "agent4rec_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_"
    "candidate_summary"
)
TEMPERATURE = 0.0
MAX_TOKENS = 1000
MAX_HISTORY_ITEMS = 20
MAX_LLM_ATTEMPTS = 5
TASTE_TEMPERATURE = 0.0
TASTE_MAX_TOKENS = None
OLLAMA_MAX_WORKERS = 1
VLLM_MAX_WORKERS = 32

DATASET_CANDIDATE_COLUMNS = {
    "ml-1m": ("item_title", PREFIXED_ITEM_RATING_MEAN_COLUMN, "item_genres"),
    "steam": ("item_title", "item_genres_json", "item_tags_json"),
}
DATASET_COLUMN_LABELS = {
    "ml-1m": {
        "item_title": "movie title",
        PREFIXED_ITEM_RATING_MEAN_COLUMN: "History ratings",
        "item_genres": "genres",
    },
    "steam": {
        "item_title": "game title",
        "item_genres_json": "genres",
        "item_tags_json": "tags",
    },
}
DATASET_JSON_LIST_COLUMNS = {
    "ml-1m": (),
    "steam": ("item_genres_json", "item_tags_json"),
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


def run_llama31_8b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_smoke",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=25,
        max_workers=OLLAMA_MAX_WORKERS,
        summary_usage=canonical_agent4rec_summary_usage(task),
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
        summary_usage=canonical_agent4rec_summary_usage(task),
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
        summary_usage=canonical_agent4rec_summary_usage(task),
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
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
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
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary(
        task,
        output_dir,
        max_candidate_groups=25,
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
        method_name=f"{LITELLM_QWEN3_8B_TRAITS_TASTE_METHOD_NAME}_{suffix}",
        client_name=LITELLM_CLIENT_NAME,
        model=QWEN3_8B_MODEL,
        max_candidate_groups=max_candidate_groups,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
        summary_usage="candidate",
        serving_metadata=_serving_metadata(),
        source_metadata=_source_metadata(),
    )


def run_qwen36_27b_port8001_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_PORT8001_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_27B_PORT8001_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen36_27b_port8001_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_PORT8001_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_27B_PORT8001_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen36_27b_port8002_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_PORT8002_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_27B_PORT8002_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        summary_usage=canonical_agent4rec_summary_usage(task),
    )


def run_qwen36_27b_port8002_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_PORT8002_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_27B_PORT8002_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
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
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_history_items: int | None = MAX_HISTORY_ITEMS,
    max_llm_attempts: int = MAX_LLM_ATTEMPTS,
    max_workers: int = 1,
    extra_body: dict | None = None,
    profile_components: tuple[str, ...] = ("traits",),
    taste_client_name: str | None = None,
    taste_model: str | None = None,
    taste_temperature: float = TASTE_TEMPERATURE,
    taste_max_tokens: int | None = TASTE_MAX_TOKENS,
    taste_max_attempts: int = MAX_LLM_ATTEMPTS,
    taste_prompt_version: str | None = None,
    taste_cache_path: Path | None = None,
    scorer_class: type[Agent4RecYesNoScorer] = Agent4RecYesNoScorer,
    scorer_kwargs: dict[str, object] | None = None,
    candidate_description_columns: tuple[str, ...] | None = None,
    column_labels: dict[str, str] | None = None,
    parser_contract: str = "agent4rec_labeled_id_movie_watch_reason",
    serving_metadata: dict[str, object] | None = None,
    source_metadata: dict[str, object] | None = None,
    summary_usage: Agent4RecSummaryUsage = "candidate",
) -> dict[str, object]:
    """Run the Agent4Rec profile-based yes/no scorer."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    if dataset_name not in DATASET_CANDIDATE_COLUMNS:
        raise ValueError(
            "Agent4Rec yes/no v1 has no dataset prompt config for "
            f"dataset: {dataset_name!r}"
        )
    if candidate_description_columns is None:
        candidate_description_columns = DATASET_CANDIDATE_COLUMNS[dataset_name]
    if column_labels is None:
        column_labels = DATASET_COLUMN_LABELS[dataset_name]
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("Agent4Rec yes/no method requires candidate_group_column")

    uses_taste = "taste" in profile_components
    resolved_summary_usage = resolve_agent4rec_summary_usage(summary_usage)
    if resolved_summary_usage["profile"] and not uses_taste:
        raise ValueError(
            "Agent4Rec summary_usage='profile' or 'both' requires "
            "'taste' in profile_components"
        )
    if resolved_summary_usage["any"] and dataset_name != "ml-1m":
        raise ValueError(
            "Agent4Rec movie summaries are configured only for ml-1m, got "
            f"{dataset_name!r}"
        )

    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = limit_candidate_groups(
        *xy["test"],
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    item_summary_metadata = task_item_summary_metadata(
        task,
        profile=resolved_summary_usage["profile"],
        candidate=resolved_summary_usage["candidate"],
    )
    candidate_columns = _candidate_columns(
        candidate_description_columns,
        use_item_summaries=resolved_summary_usage["candidate"],
    )
    column_labels = _column_labels(
        column_labels,
        use_item_summaries=resolved_summary_usage["candidate"],
    )
    profile_user_ids = X_test["user_id"].drop_duplicates().tolist()
    _require_columns(X_test, list(candidate_columns))

    if uses_taste and not DATASET_SUPPORTS_TASTE[dataset_name]:
        raise ValueError(
            "Agent4Rec taste profiles are currently configured only for "
            "MovieLens-style 1-5 rating histories. "
            f"Got dataset: {dataset_name!r}"
        )
    if uses_taste:
        if not taste_client_name:
            raise ValueError("taste_client_name is required for taste profiles")
        if not taste_model:
            raise ValueError("taste_model is required for taste profiles")
        if taste_prompt_version is None:
            taste_prompt_version = DATASET_TASTE_PROMPT_VERSION[dataset_name]
        if taste_cache_path is None:
            taste_cache_path = agent4rec_taste_cache_path(
                task,
                taste_model=taste_model,
                taste_prompt_version=taste_prompt_version,
                use_history_item_summaries=resolved_summary_usage["profile"],
            )
        taste_client = make_llm_client(taste_client_name)
    else:
        if taste_prompt_version is None:
            taste_prompt_version = DATASET_TASTE_PROMPT_VERSION[dataset_name]
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
        **DATASET_PROFILE_GENERATOR_KWARGS[dataset_name],
    )
    scorer = scorer_class(
        client=make_llm_client(client_name),
        model=model,
        profile_generator=profile_generator,
        candidate_description_columns=candidate_columns,
        column_labels=column_labels,
        json_list_columns=DATASET_JSON_LIST_COLUMNS[dataset_name],
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        **({} if scorer_kwargs is None else scorer_kwargs),
        **DATASET_PROMPT_KWARGS[dataset_name],
    ).fit(X_train, y_train, profile_user_ids=profile_user_ids)
    if uses_taste:
        scorer.build_taste(X_test)

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
        raise RuntimeError("Agent4Rec scorer did not produce any valid scores")

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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_history_items": max_history_items,
            "candidate_description_columns": list(candidate_columns),
            "column_labels": column_labels,
            "json_list_columns": list(DATASET_JSON_LIST_COLUMNS[dataset_name]),
            "profile_generator": scorer.profile_generator.manifest(),
            "extra_body": extra_body,
            "prompt": DATASET_PROMPT_KWARGS[dataset_name],
            "scorer_kwargs": scorer_kwargs,
            "serving": serving_metadata,
            "summary_usage": summary_usage,
            "item_summaries": item_summary_metadata,
        },
        "decision_rule": {
            "kind": "hard_binary_yes_no_parser",
            "parser_contract": parser_contract,
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
    }
    if source_metadata is not None:
        manifest["source"] = source_metadata
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


def _candidate_columns(
    columns: tuple[str, ...],
    *,
    use_item_summaries: bool,
) -> tuple[str, ...]:
    if use_item_summaries and ITEM_SUMMARY_COLUMN not in columns:
        return (*columns, ITEM_SUMMARY_COLUMN)
    return columns


def _column_labels(
    base_labels: dict[str, str],
    *,
    use_item_summaries: bool,
) -> dict[str, str]:
    labels = dict(base_labels)
    if use_item_summaries:
        labels[ITEM_SUMMARY_COLUMN] = ITEM_SUMMARY_COLUMN_LABEL
    return labels


def _require_columns(frame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Agent4Rec yes/no requires item-stats task columns for the candidate "
            f"prompt. Missing columns: {missing}"
        )


def agent4rec_taste_cache_path(
    task: Task,
    *,
    taste_model: str,
    taste_prompt_version: str,
    use_history_item_summaries: bool = False,
) -> Path:
    dataset_name = str(task.manifest["dataset"])
    dataset_version = str(task.manifest["dataset_version"])
    split_seed = task.manifest["splitter"]["seed"]
    model_slug = _cache_slug(taste_model)
    summary_slug = "_summary" if use_history_item_summaries else ""
    return (
        repo_root()
        / "outputs"
        / "agent4rec_taste_cache"
        / (
            f"{dataset_name}_{dataset_version}_seed{split_seed}_"
            f"{model_slug}_{taste_prompt_version}{summary_slug}.jsonl"
        )
    )


def _cache_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not slug:
        raise ValueError(f"Cannot build cache slug from value: {value!r}")
    return slug
